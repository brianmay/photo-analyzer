//! ONNX-based face detection using ORT with Non-Maximum Suppression.
//!
//! Downloads the `deepghs/face_detect_onnx` ONNX model from HuggingFace on
//! first use. If the model cannot be downloaded or loaded, face detection is
//! disabled gracefully — the rest of the pipeline (captioning, categorisation)
//! continues to work normally.
//!
//! Expected output format: [batch, num_detections, 6] where the last
//! dimension is [x1, y1, x2, y2, confidence, class_id].

use anyhow::{Context, Result};
use hf_hub::api::sync::Api;
use image::DynamicImage;
use ndarray::Array4;
use ort::{session::Session, value::TensorRef};

use crate::models::{BoundingBox, Person};

const MODEL_ID: &str = "deepghs/face_detect_onnx";
const MODEL_FILE: &str = "face_detect.onnx";
const DETECTION_SIZE: u32 = 640;
const CONFIDENCE_THRESHOLD: f32 = 0.5;
const IOU_THRESHOLD: f32 = 0.4;

pub struct FaceDetector {
    session: Option<Session>,
}

impl FaceDetector {
    /// Attempt to download and initialise the face-detection model.
    ///
    /// If the model is unavailable (network error, authentication required,
    /// etc.) a warning is logged and the detector is returned in a disabled
    /// state.  All subsequent calls to [`detect_faces`] will return an empty
    /// list in that case.
    pub fn load() -> Result<Self> {
        match Self::try_load() {
            Ok(session) => Ok(Self { session: Some(session) }),
            Err(e) => {
                tracing::warn!(
                    "Face detection model could not be loaded — face detection \
                     is disabled for this session. Cause: {e:#}"
                );
                Ok(Self { session: None })
            }
        }
    }

    fn try_load() -> Result<Session> {
        let api = Api::new().context("Failed to create HuggingFace API client")?;
        let repo = api.model(MODEL_ID.to_string());

        let model_path = repo
            .get(MODEL_FILE)
            .context("Failed to download face detection ONNX model")?;

        let session = Session::builder()
            .context("Failed to create ORT session builder")?
            .commit_from_file(model_path)
            .context("Failed to load ONNX face detection model")?;

        Ok(session)
    }

    pub fn detect_faces(&mut self, img: &DynamicImage) -> Result<Vec<Person>> {
        let session = match &mut self.session {
            Some(s) => s,
            None => return Ok(vec![]),
        };

        match Self::run_inference(session, img) {
            Ok(people) => Ok(people),
            Err(e) => {
                tracing::warn!("Face detection inference failed — skipping faces for this image. Cause: {e:#}");
                Ok(vec![])
            }
        }
    }

    fn run_inference(session: &mut Session, img: &DynamicImage) -> Result<Vec<Person>> {
        let (orig_w, orig_h) = (img.width(), img.height());
        let input_arr = preprocess_image(img, DETECTION_SIZE, DETECTION_SIZE)?;
        let input_view = input_arr.view();
        let ort_input = TensorRef::<f32>::from_array_view(input_view)
            .context("Failed to create ORT input tensor")?;

        let outputs = session
            .run(ort::inputs![ort_input])
            .context("Face detection inference failed")?;

        if outputs.len() == 0 {
            return Ok(vec![]);
        }

        // Expected output: [batch, num_detections, 6]
        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .context("Failed to extract detection output tensor")?;

        let detections = parse_detections(shape, data, orig_w, orig_h)?;
        Ok(non_maximum_suppression(detections, IOU_THRESHOLD))
    }
}

fn preprocess_image(img: &DynamicImage, w: u32, h: u32) -> Result<Array4<f32>> {
    use image::imageops::FilterType;

    let resized = img.resize_exact(w, h, FilterType::Lanczos3);
    let rgb = resized.to_rgb8();

    let mut arr = Array4::<f32>::zeros((1, 3, h as usize, w as usize));
    for (y, row) in rgb.rows().enumerate() {
        for (x, pixel) in row.enumerate() {
            arr[[0, 0, y, x]] = pixel[0] as f32 / 255.0;
            arr[[0, 1, y, x]] = pixel[1] as f32 / 255.0;
            arr[[0, 2, y, x]] = pixel[2] as f32 / 255.0;
        }
    }
    Ok(arr)
}

#[derive(Clone)]
struct Detection {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    confidence: f32,
}

fn parse_detections(
    shape: &ort::value::Shape,
    data: &[f32],
    _orig_w: u32,
    _orig_h: u32,
) -> Result<Vec<Detection>> {
    let mut detections = Vec::new();

    // shape: [batch, num_detections, 6] or [num_detections, 6]
    if shape.len() < 2 || data.len() < 6 {
        return Ok(detections);
    }

    let stride = 6usize;
    let n = data.len() / stride;

    for i in 0..n {
        let base = i * stride;
        if base + 5 >= data.len() {
            break;
        }
        let confidence = data[base + 4];
        if confidence < CONFIDENCE_THRESHOLD {
            continue;
        }
        // Coordinates are in pixel space of the resized (DETECTION_SIZE x DETECTION_SIZE) image.
    // Divide by DETECTION_SIZE to normalize to [0, 1].
    let x1n = data[base] / DETECTION_SIZE as f32;
        let y1n = data[base + 1] / DETECTION_SIZE as f32;
        let x2n = data[base + 2] / DETECTION_SIZE as f32;
        let y2n = data[base + 3] / DETECTION_SIZE as f32;

        detections.push(Detection {
            x: x1n.clamp(0.0, 1.0),
            y: y1n.clamp(0.0, 1.0),
            w: (x2n - x1n).clamp(0.0, 1.0),
            h: (y2n - y1n).clamp(0.0, 1.0),
            confidence,
        });
    }

    Ok(detections)
}

fn non_maximum_suppression(mut detections: Vec<Detection>, iou_threshold: f32) -> Vec<Person> {
    detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

    let mut keep = vec![true; detections.len()];
    for i in 0..detections.len() {
        if !keep[i] {
            continue;
        }
        for j in (i + 1)..detections.len() {
            if !keep[j] {
                continue;
            }
            if iou(&detections[i], &detections[j]) > iou_threshold {
                keep[j] = false;
            }
        }
    }

    detections
        .into_iter()
        .enumerate()
        .filter(|(i, _)| keep[*i])
        .map(|(_, d)| Person {
            name: "Unknown".to_string(),
            confidence: d.confidence,
            bounding_box: BoundingBox {
                x: d.x,
                y: d.y,
                width: d.w,
                height: d.h,
            },
        })
        .collect()
}

fn iou(a: &Detection, b: &Detection) -> f32 {
    let ix1 = a.x.max(b.x);
    let iy1 = a.y.max(b.y);
    let ix2 = (a.x + a.w).min(b.x + b.w);
    let iy2 = (a.y + a.h).min(b.y + b.h);

    if ix2 <= ix1 || iy2 <= iy1 {
        return 0.0;
    }

    let intersection = (ix2 - ix1) * (iy2 - iy1);
    let union = a.w * a.h + b.w * b.h - intersection;
    if union <= 0.0 { 0.0 } else { intersection / union }
}
