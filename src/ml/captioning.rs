//! BLIP-based image captioning using candle-transformers.
//!
//! Downloads `Salesforce/blip-image-captioning-large` from HuggingFace on first
//! use and generates captions via greedy decoding with KV-cache.

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::blip;
use hf_hub::api::sync::Api;
use image::DynamicImage;
use tokenizers::Tokenizer;

const MODEL_ID: &str = "Salesforce/blip-image-captioning-large";
// BLIP uses the BERT tokenizer; download from bert-base-uncased to avoid
// relative-URL issues in the BLIP repo's tokenizer_config.json.
const TOKENIZER_MODEL_ID: &str = "bert-base-uncased";
const IMAGE_SIZE: usize = 384;
const SEP_TOKEN_ID: u32 = 102;
const MAX_TOKENS: usize = 64;

const IMAGENET_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const IMAGENET_STD: [f32; 3] = [0.26862954, 0.261_302_6, 0.275_777_1];

pub struct CaptioningModel {
    model: blip::BlipForConditionalGeneration,
    tokenizer: Tokenizer,
    device: Device,
}

impl CaptioningModel {
    pub fn load() -> Result<Self> {
        let device = Device::Cpu;
        let api = Api::new().context("Failed to create HuggingFace API client")?;
        let repo = api.model(MODEL_ID.to_string());

        let model_file = repo
            .get("model.safetensors")
            .context("Failed to download BLIP model weights")?;

        // The BLIP tokenizer is a BERT tokenizer; fetch it from bert-base-uncased
        // to avoid relative-URL resolution failures in the BLIP repo's tokenizer metadata.
        let tokenizer_file = api
            .model(TOKENIZER_MODEL_ID.to_string())
            .get("tokenizer.json")
            .context("Failed to download BERT tokenizer for BLIP")?;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, &device)
                .context("Failed to load BLIP model weights")?
        };

        let config = blip::Config::image_captioning_large();
        let model = blip::BlipForConditionalGeneration::new(&config, vb)
            .context("Failed to build BLIP model")?;

        let tokenizer =
            Tokenizer::from_file(tokenizer_file).map_err(|e| anyhow::anyhow!("{e}"))?;

        Ok(Self { model, tokenizer, device })
    }

    pub fn caption(&mut self, img: &DynamicImage) -> Result<(String, String)> {
        // Reset the KV cache so that previous images don't pollute this run.
        self.model.reset_kv_cache();

        let pixel_values = preprocess_image(img, &self.device)?;
        let image_embeds = self
            .model
            .vision_model()
            .forward(&pixel_values)
            .map_err(|e| anyhow::anyhow!("Vision forward failed: {e}"))?;

        // Greedy decoding with incremental (token-by-token) KV-cache feeding.
        //
        // The BLIP text decoder maintains a KV cache and tracks `past_kv_len`
        // for positional embeddings.  On every call only the *latest* token is
        // fed; the model appends its key/value pair to the cache and attends to
        // all previous positions through it.  Feeding the full accumulated
        // sequence instead would make the actual KV length (cache + new tokens)
        // diverge from the causal-mask size, causing a shape mismatch.
        let mut token_ids: Vec<u32> = vec![101]; // start with CLS (id=101)
        let mut generated = String::new();

        for _ in 0..MAX_TOKENS {
            // Pass only the most recently added token.
            let last_id = *token_ids.last().unwrap();
            let input = Tensor::new(&[last_id], &self.device)
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| anyhow::anyhow!("Tensor error: {e}"))?;

            let logits = self
                .model
                .text_decoder()
                .forward(&input, &image_embeds)
                .map_err(|e| anyhow::anyhow!("Text decoder error: {e}"))?;

            // logits shape: [1, 1, vocab_size] — take the single position.
            let last_logits = logits
                .i((0, 0, ..))
                .map_err(|e| anyhow::anyhow!("Logit indexing error: {e}"))?;
            let next_token = last_logits
                .argmax(0)
                .map_err(|e| anyhow::anyhow!("Argmax error: {e}"))?
                .to_scalar::<u32>()
                .map_err(|e| anyhow::anyhow!("Scalar error: {e}"))?;

            if next_token == SEP_TOKEN_ID {
                break;
            }
            token_ids.push(next_token);

            if let Ok(piece) = self.tokenizer.decode(&[next_token], true) {
                generated.push_str(&piece);
                generated.push(' ');
            }
        }

        let caption = generated.trim().to_string();
        let title = caption
            .split_once('.')
            .map(|(s, _)| capitalize(s.trim()))
            .unwrap_or_else(|| capitalize(&caption));
        let description = if caption.is_empty() {
            "A photograph.".to_string()
        } else {
            capitalize(&caption)
        };

        Ok((title, description))
    }
}

fn preprocess_image(img: &DynamicImage, device: &Device) -> Result<Tensor> {
    use image::imageops::FilterType;

    let resized = img.resize_exact(IMAGE_SIZE as u32, IMAGE_SIZE as u32, FilterType::Lanczos3);
    let rgb = resized.to_rgb8();
    let (w, h) = rgb.dimensions();
    let pixels = w as usize * h as usize;

    let mut data = vec![0f32; 3 * pixels];
    for (i, pixel) in rgb.pixels().enumerate() {
        data[i] = (pixel[0] as f32 / 255.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
        data[pixels + i] = (pixel[1] as f32 / 255.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
        data[2 * pixels + i] = (pixel[2] as f32 / 255.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
    }

    Tensor::from_vec(data, (1, 3, h as usize, w as usize), device)
        .map_err(|e| anyhow::anyhow!("Failed to create image tensor: {e}"))
}

fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
    }
}
