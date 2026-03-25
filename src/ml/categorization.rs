//! CLIP-based zero-shot image categorization.
//!
//! Downloads `openai/clip-vit-base-patch32` from HuggingFace on first use and
//! classifies images against a fixed set of category labels.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::VarBuilder;
use candle_transformers::models::clip;
use hf_hub::api::sync::Api;
use image::DynamicImage;
use tokenizers::{AddedToken, Tokenizer};
use tokenizers::models::bpe::BPE;
use tokenizers::processors::template::TemplateProcessing;

use crate::models::Category;

const MODEL_ID: &str = "openai/clip-vit-base-patch32";
const IMAGE_SIZE: usize = 224;

const IMAGENET_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const IMAGENET_STD: [f32; 3] = [0.26862954, 0.261_302_6, 0.275_777_1];

/// Default photo category labels used for zero-shot classification.
const CATEGORIES: &[&str] = &[
    "landscape",
    "portrait",
    "wildlife",
    "architecture",
    "street photography",
    "macro photography",
    "sports",
    "travel",
    "food",
    "abstract",
    "night photography",
    "black and white",
    "underwater",
    "aerial photography",
    "family",
];

pub struct CategorizationModel {
    model: clip::ClipModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl CategorizationModel {
    pub fn load() -> Result<Self> {
        let device = Device::Cpu;
        let api = Api::new().context("Failed to create HuggingFace API client")?;
        let repo = api.model(MODEL_ID.to_string());

        // Try safetensors first; fall back to the PyTorch bin format if not available.
        let (model_file, use_safetensors) = match repo.get("model.safetensors") {
            Ok(path) => (path, true),
            Err(_) => {
                let path = repo
                    .get("pytorch_model.bin")
                    .context("Failed to download CLIP model weights")?;
                (path, false)
            }
        };
        // Build the CLIP BPE tokenizer from raw vocab/merges files.
        //
        // The openai/clip-vit-base-patch32 repo triggers hf-hub's "relative URL
        // without base" error for every file (including vocab.json and merges.txt),
        // so we cannot use it as a download source at all.
        //
        // Instead we fetch from the gpt2 repo (which downloads cleanly) and then
        // derive CLIP's vocabulary from it.  CLIP's BPE is a strict prefix of
        // GPT-2's BPE:
        //
        //   IDs 0-255:     256 byte-level tokens  (identical in both)
        //   IDs 256-49405: 49150 BPE merge tokens (first 49150 of GPT-2's 50000)
        //   ID  49406:     <|startoftext|>
        //   ID  49407:     <|endoftext|>
        //
        // Total: 49408 tokens — exactly CLIP's embedding table size.
        let gpt2_repo = api.model("gpt2".to_string());
        let gpt2_vocab_path = gpt2_repo
            .get("vocab.json")
            .context("Failed to download CLIP vocabulary")?;
        let gpt2_merges_path = gpt2_repo
            .get("merges.txt")
            .context("Failed to download CLIP merge rules")?;

        // --- Filter vocab.json to CLIP's 49408-token vocabulary ---
        let gpt2_vocab_bytes = std::fs::read(&gpt2_vocab_path)
            .context("Failed to read GPT-2 vocab.json")?;
        let mut vocab: std::collections::HashMap<String, u32> =
            serde_json::from_slice(&gpt2_vocab_bytes)
            .context("Failed to parse GPT-2 vocab.json")?;
        // Keep IDs 0-49405 and replace GPT-2's special token with CLIP's two.
        vocab.retain(|_, id| *id < 49406);
        vocab.insert("<|startoftext|>".to_string(), 49406);
        vocab.insert("<|endoftext|>".to_string(), 49407);

        let clip_vocab_path = std::env::temp_dir().join("photo_analyzer_clip_vocab.json");
        std::fs::write(
            &clip_vocab_path,
            serde_json::to_vec(&vocab).context("Failed to serialize CLIP vocab")?,
        )
        .context("Failed to write CLIP vocab.json")?;

        // --- Trim merges.txt to CLIP's 49150 merge rules ---
        // GPT-2 merges.txt layout: one header comment line followed by 50000 rules.
        // CLIP needs only the first 49150 rules (256 byte tokens + 49150 merges =
        // 49406 base tokens, then +2 special tokens = 49408 total).
        let gpt2_merges = std::fs::read_to_string(&gpt2_merges_path)
            .context("Failed to read GPT-2 merges.txt")?;
        // GPT-2 merges.txt: line 0 is a header (#version: …), lines 1+ are rules.
        // We keep the header plus the first 49150 rules → 49151 lines total.
        let clip_merges: String = gpt2_merges
            .lines()
            .take(49151)
            .flat_map(|l| [l, "\n"])
            .collect();
        let clip_merges_path = std::env::temp_dir().join("photo_analyzer_clip_merges.txt");
        std::fs::write(&clip_merges_path, &clip_merges)
            .context("Failed to write CLIP merges.txt")?;

        let bpe_model = BPE::from_file(
            clip_vocab_path.to_str().context("CLIP vocab path is not valid UTF-8")?,
            clip_merges_path.to_str().context("CLIP merges path is not valid UTF-8")?,
        )
        .unk_token("<|endoftext|>".to_string())
        .build()
        .map_err(|e| anyhow::anyhow!("Failed to build CLIP BPE model: {}", e))?;

        let mut tokenizer = Tokenizer::new(bpe_model);

        // Register CLIP's special tokens. Since <|startoftext|> and <|endoftext|>
        // are already present in the vocab at IDs 49406 and 49407, this call just
        // marks them as "special" (bypassing the normal BPE segmentation) without
        // reassigning their IDs.
        tokenizer.add_special_tokens(&[
            AddedToken::from("<|startoftext|>".to_string(), true),
            AddedToken::from("<|endoftext|>".to_string(), true),
        ]);

        // Wrap every sequence with BOS / EOS, matching CLIP's text-encoder contract.
        let bos_id = tokenizer
            .token_to_id("<|startoftext|>")
            .context("BOS token not found in CLIP vocabulary")?;
        let eos_id = tokenizer
            .token_to_id("<|endoftext|>")
            .context("EOS token not found in CLIP vocabulary")?;

        let post_processor = TemplateProcessing::builder()
            .try_single("<|startoftext|> $A <|endoftext|>")
            .map_err(|e| anyhow::anyhow!("CLIP template error: {}", e))?
            .special_tokens(vec![
                ("<|startoftext|>", bos_id),
                ("<|endoftext|>", eos_id),
            ])
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build CLIP post-processor: {}", e))?;

        tokenizer.with_post_processor(Some(post_processor));

        let vb = if use_safetensors {
            unsafe {
                VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, &device)
                    .context("Failed to load CLIP model weights")?
            }
        } else {
            VarBuilder::from_pth(model_file, DType::F32, &device)
                .context("Failed to load CLIP model weights")?
        };

        let config = clip::ClipConfig::vit_base_patch32();
        let model = clip::ClipModel::new(vb, &config)
            .map_err(|e| anyhow::anyhow!("Failed to build CLIP model: {}", e))?;

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    pub fn categorize(&mut self, img: &DynamicImage, threshold: f32) -> Result<Vec<Category>> {
        let pixel_values = preprocess_image(img, &self.device)?;
        let image_features = self
            .model
            .get_image_features(&pixel_values)
            .map_err(|e| anyhow::anyhow!("Image features error: {}", e))?;
        let image_features = clip::div_l2_norm(&image_features)
            .map_err(|e| anyhow::anyhow!("L2 norm error: {}", e))?;

        // Encode each category label.
        let mut text_embeddings: Vec<Tensor> = Vec::with_capacity(CATEGORIES.len());
        for &label in CATEGORIES {
            let prompt = format!("a photo of {}", label);
            let encoding = self
                .tokenizer
                .encode(prompt, true)
                .map_err(|e| anyhow::anyhow!("Tokenizer error: {}", e))?;
            let ids: Vec<u32> = encoding.get_ids().to_vec();
            let input = Tensor::new(ids.as_slice(), &self.device)
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| anyhow::anyhow!("Tensor error: {}", e))?;
            let feat = self
                .model
                .get_text_features(&input)
                .map_err(|e| anyhow::anyhow!("Text features error: {}", e))?;
            let feat = clip::div_l2_norm(&feat)
                .map_err(|e| anyhow::anyhow!("L2 norm error: {}", e))?;
            text_embeddings.push(feat);
        }

        // Stack text embeddings and compute cosine similarities.
        let text_stack = Tensor::cat(&text_embeddings, 0)
            .map_err(|e| anyhow::anyhow!("Stack error: {}", e))?;

        let image_t = image_features
            .transpose(0, 1)
            .map_err(|e| anyhow::anyhow!("Transpose error: {}", e))?;
        let logits = text_stack
            .matmul(&image_t)
            .map_err(|e| anyhow::anyhow!("Matmul error: {}", e))?;
        let logits = logits
            .squeeze(1)
            .map_err(|e| anyhow::anyhow!("Squeeze error: {}", e))?;

        // Softmax to get probabilities.
        let probs = softmax(&logits)?;
        let probs_vec: Vec<f32> = probs
            .to_vec1()
            .map_err(|e| anyhow::anyhow!("Vec conversion error: {}", e))?;

        let categories = CATEGORIES
            .iter()
            .zip(probs_vec.iter())
            .filter(|(_, &p)| p >= threshold)
            .map(|(&name, &confidence)| Category {
                name: name.to_string(),
                confidence,
            })
            .collect();

        Ok(categories)
    }
}

fn preprocess_image(img: &DynamicImage, device: &Device) -> Result<Tensor> {
    use image::imageops::FilterType;

    let resized = img.resize_exact(IMAGE_SIZE as u32, IMAGE_SIZE as u32, FilterType::Lanczos3);
    let rgb = resized.to_rgb8();
    let (w, h) = rgb.dimensions();

    let mut data = vec![0f32; (3 * h * w) as usize];
    for (i, pixel) in rgb.pixels().enumerate() {
        let r = pixel[0] as f32 / 255.0;
        let g = pixel[1] as f32 / 255.0;
        let b = pixel[2] as f32 / 255.0;

        data[i] = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
        data[(h * w) as usize + i] = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
        data[2 * (h * w) as usize + i] = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
    }

    Tensor::from_vec(data, (1, 3, h as usize, w as usize), device)
        .map_err(|e| anyhow::anyhow!("Failed to create image tensor: {}", e))
}

fn softmax(t: &Tensor) -> Result<Tensor> {
    let max = t
        .max(D::Minus1)
        .map_err(|e| anyhow::anyhow!("Max error: {}", e))?;
    let shifted = t
        .broadcast_sub(&max)
        .map_err(|e| anyhow::anyhow!("Broadcast sub error: {}", e))?;
    let exp = shifted
        .exp()
        .map_err(|e| anyhow::anyhow!("Exp error: {}", e))?;
    let sum = exp
        .sum_keepdim(D::Minus1)
        .map_err(|e| anyhow::anyhow!("Sum error: {}", e))?;
    exp.broadcast_div(&sum)
        .map_err(|e| anyhow::anyhow!("Broadcast div error: {}", e))
}
