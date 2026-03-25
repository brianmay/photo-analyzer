//! ONNX-based face detection using Ultra-Light-Fast-Generic-Face-Detector.
//!
//! Downloads `version-RFB-640.onnx` from GitHub on first use (MIT licence,
//! <https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB>)
//! and caches it in the OS temporary directory.  If the model cannot be
//! downloaded or loaded, face detection is disabled gracefully — the rest of
//! the pipeline (captioning, categorisation) continues to work normally.
//!
//! Model I/O contract
//! ------------------
//! Input  : `input`  — shape `[1, 3, 480, 640]` (NCHW), pixel values in
//!                      `[-1, 1]` via `(v − 127) / 128`.
//! Output 0: `scores` — shape `[1, N, 2]` — `[background_prob, face_prob]`
//!                      for each of the N anchor boxes.
//! Output 1: `boxes`  — shape `[1, N, 4]` — `[x1, y1, x2, y2]` normalised
//!                      to `[0, 1]` relative to the 640 × 480 input frame.

use std::io::Read;

use anyhow::{Context, Result};
use image::DynamicImage;
use ndarray::Array4;
use ort::{session::Session, value::TensorRef};

use crate::models::{BoundingBox, Person};

/// Public GitHub raw URL for version-RFB-640.onnx (MIT licence).
const MODEL_URL: &str = concat!(
    "https://raw.githubusercontent.com/",
    "Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/",
    "master/models/onnx/version-RFB-640.onnx"
);
const CACHE_FILENAME: &str = "photo_analyzer_ultraface_rfb640.onnx";

/// Input width/height expected by this model variant.
const INPUT_W: u32 = 640;
const INPUT_H: u32 = 480;

/// Minimum face-class probability to keep a detection.
const CONFIDENCE_THRESHOLD: f32 = 0.7;
/// IoU threshold for non-maximum suppression.
const IOU_THRESHOLD: f32 = 0.4;

// ---------------------------------------------------------------------------

pub struct FaceDetector {
    session: Option<Session>,
}

impl FaceDetector {
    /// Attempt to download and initialise the face-detection model.
    ///
    /// On any failure a `WARN` is logged and the detector is returned in a
    /// disabled state; [`detect_faces`] will then return an empty list.
    pub fn load() -> Result<Self> {
        match Self::try_load() {
            Ok(session) => Ok(Self {
                session: Some(session),
            }),
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
        let cache_path = std::env::temp_dir().join(CACHE_FILENAME);
        if !cache_path.exists() {
            Self::download_model(&cache_path)?;
        }

        Session::builder()
            .context("Failed to create ORT session builder")?
            .commit_from_file(&cache_path)
            .context("Failed to load ONNX face detection model")
    }

    fn download_model(dest: &std::path::Path) -> Result<()> {
        tracing::info!(
            "Downloading face detection model (Ultra-Light-Fast-Generic-Face-Detector) …"
        );

        let response = ureq::get(MODEL_URL)
            .call()
            .map_err(|e| anyhow::anyhow!("HTTP request for face detection model failed: {e}"))?;

        let status = response.status();
        anyhow::ensure!(
            status == 200,
            "Unexpected HTTP status {} while downloading face detection model from {}",
            status,
            MODEL_URL
        );

        let mut bytes = Vec::new();
        response
            .into_reader()
            .read_to_end(&mut bytes)
            .context("Failed to read face detection model response body")?;

        // Write to a temp file first, then rename atomically to avoid leaving
        // a partially-written file if the process is interrupted or two
        // instances run concurrently.
        let tmp_path = dest.with_extension("tmp");
        std::fs::write(&tmp_path, &bytes)
            .context("Failed to write face detection model to temp file")?;
        std::fs::rename(&tmp_path, dest)
            .context("Failed to move face detection model to cache location")?;

        tracing::info!("Face detection model cached at {:?}", dest);
        Ok(())
    }

    pub fn detect_faces(&mut self, img: &DynamicImage) -> Result<Vec<Person>> {
        let session = match &mut self.session {
            Some(s) => s,
            None => return Ok(vec![]),
        };

        match Self::run_inference(session, img) {
            Ok(people) => Ok(people),
            Err(e) => {
                tracing::warn!(
                    "Face detection inference failed — skipping faces for this image. \
                     Cause: {e:#}"
                );
                Ok(vec![])
            }
        }
    }

    fn run_inference(session: &mut Session, img: &DynamicImage) -> Result<Vec<Person>> {
        let input_arr = preprocess_image(img)?;
        let input_view = input_arr.view();
        let ort_input = TensorRef::<f32>::from_array_view(input_view)
            .context("Failed to create ORT input tensor")?;

        let outputs = session
            .run(ort::inputs![ort_input])
            .context("Face detection inference failed")?;

        if outputs.len() < 2 {
            return Ok(vec![]);
        }

        // UltraFace output layout:
        //   outputs[0] = scores — shape [1, N, 2]: [bkg_prob, face_prob]
        //   outputs[1] = boxes  — shape [1, N, 4]: [x1, y1, x2, y2] in [0,1]
        let (_, scores_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .context("Failed to extract face-scores tensor")?;
        let (_, boxes_data) = outputs[1]
            .try_extract_tensor::<f32>()
            .context("Failed to extract face-boxes tensor")?;

        let n = scores_data.len() / 2;

        // Validate that both tensors are consistent before indexing.
        anyhow::ensure!(
            scores_data.len() % 2 == 0,
            "UltraFace scores tensor has unexpected length {} (expected multiple of 2)",
            scores_data.len()
        );
        anyhow::ensure!(
            boxes_data.len() == scores_data.len() * 2,
            "UltraFace tensor size mismatch: scores.len()={} but boxes.len()={} (expected {})",
            scores_data.len(),
            boxes_data.len(),
            scores_data.len() * 2
        );

        let detections = collect_detections(n, scores_data, boxes_data);
        Ok(non_maximum_suppression(detections, IOU_THRESHOLD))
    }
}

// ---------------------------------------------------------------------------
// Pre-processing

fn preprocess_image(img: &DynamicImage) -> Result<Array4<f32>> {
    use image::imageops::FilterType;

    let resized = img.resize_exact(INPUT_W, INPUT_H, FilterType::Lanczos3);
    let rgb = resized.to_rgb8();

    let mut arr = Array4::<f32>::zeros((1, 3, INPUT_H as usize, INPUT_W as usize));
    for (y, row) in rgb.rows().enumerate() {
        for (x, pixel) in row.enumerate() {
            // UltraFace normalisation: (pixel − 127) / 128  →  roughly [−1, 1]
            arr[[0, 0, y, x]] = (pixel[0] as f32 - 127.0) / 128.0;
            arr[[0, 1, y, x]] = (pixel[1] as f32 - 127.0) / 128.0;
            arr[[0, 2, y, x]] = (pixel[2] as f32 - 127.0) / 128.0;
        }
    }
    Ok(arr)
}

// ---------------------------------------------------------------------------
// Post-processing

#[derive(Clone)]
struct Detection {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    confidence: f32,
}

fn collect_detections(n: usize, scores: &[f32], boxes: &[f32]) -> Vec<Detection> {
    let mut detections = Vec::new();
    for i in 0..n {
        let face_score = scores[i * 2 + 1]; // index 1 = face probability
        if face_score < CONFIDENCE_THRESHOLD {
            continue;
        }
        let x1 = boxes[i * 4].clamp(0.0, 1.0);
        let y1 = boxes[i * 4 + 1].clamp(0.0, 1.0);
        let x2 = boxes[i * 4 + 2].clamp(0.0, 1.0);
        let y2 = boxes[i * 4 + 3].clamp(0.0, 1.0);
        detections.push(Detection {
            x: x1,
            y: y1,
            w: (x2 - x1).max(0.0),
            h: (y2 - y1).max(0.0),
            confidence: face_score,
        });
    }
    detections
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
    if union <= 0.0 {
        0.0
    } else {
        intersection / union
    }
}
