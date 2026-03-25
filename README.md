# photo-analyzer

A command-line tool written in Rust that scans directories for photos (JPEG, CR3) and uses local ML models to generate titles, descriptions, quality scores, category metadata, and face detection — writing results as JSON sidecar files alongside each photo.

## Features

- **Photo scanning** – recursively or shallowly finds `.jpg`, `.jpeg`, and `.cr3` files
- **CR3 support** – decodes Canon CR3 raw files via `exiftool` or `dcraw` (optional)
- **Quality analysis** – sharpness (Laplacian variance), exposure (histogram), noise (Gaussian residual MAD), composition (rule-of-thirds face placement), and aesthetic scoring
- **ML pipeline**:
  - **Captioning** – BLIP (`Salesforce/blip-image-captioning-base`) generates titles and descriptions via greedy decoding
  - **Categorization** – CLIP (`openai/clip-vit-base-patch32`) zero-shot classification against 15 photo categories
  - **Face detection** – ONNX model (`Xenova/face-detection`) with Non-Maximum Suppression via ORT
- **Automatic model downloads** – models are downloaded from HuggingFace Hub on first run and cached locally (`~/.cache/huggingface/hub`)
- **JSON sidecar output** – results written to `<photo>.analysis.json` next to each image
- **Parallel processing** – uses Rayon for multi-core photo analysis

## Prerequisites

- **Rust 1.70+** – Install from [rustup.rs](https://rustup.rs)
- **exiftool** *(optional, for CR3 support)* – `sudo apt install libimage-exiftool-perl` or `brew install exiftool`
- **dcraw** *(optional, CR3 fallback)* – `sudo apt install dcraw` or `brew install dcraw`
- **Internet access** – required on first run to download ML models from HuggingFace Hub

## Build

```bash
cargo build --release
```

The binary will be at `target/release/photo-analyzer`.

## Usage

```bash
photo-analyzer [OPTIONS] <DIRECTORIES>...
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-r`, `--recursive` | `true` | Scan directories recursively |
| `-j`, `--parallel <N>` | *(all cores)* | Number of parallel worker threads |
| `--category-threshold <F>` | `0.1` | Minimum confidence to include a category |
| `--skip-existing` | off | Skip photos that already have `.analysis.json` |
| `--force` | off | Re-analyze even if `.analysis.json` exists |
| `-v`, `--verbose` | off | Enable verbose/debug logging |

### Examples

```bash
# Analyze a single directory
photo-analyzer ~/Photos/Vacation2024

# Analyze multiple directories, skip already-analyzed photos
photo-analyzer --skip-existing ~/Photos ~/Pictures

# Use 4 threads, lower confidence threshold
photo-analyzer -j 4 --category-threshold 0.05 /mnt/photos

# Verbose output, do not recurse
photo-analyzer --recursive false -v ~/Photos
```

## JSON Output Format

Each analyzed photo gets a `<photo>.analysis.json` sidecar:

```json
{
  "file": "DSC_0001.jpg",
  "title": "A couple walking through a park",
  "description": "A couple walking through a park on a sunny afternoon.",
  "categories": [
    { "name": "landscape", "confidence": 0.42 },
    { "name": "portrait",  "confidence": 0.31 }
  ],
  "people": [
    {
      "name": "Unknown",
      "confidence": 0.97,
      "bounding_box": { "x": 0.35, "y": 0.10, "width": 0.15, "height": 0.30 }
    }
  ],
  "quality": {
    "overall":     0.72,
    "sharpness":   0.84,
    "exposure":    0.79,
    "noise":       0.91,
    "composition": 0.65,
    "aesthetic":   0.58
  },
  "analyzed_at": "2024-06-01T14:22:05Z"
}
```

All quality scores are in the range `[0.0, 1.0]` (higher is better).  
Bounding-box coordinates are normalized to `[0.0, 1.0]` relative to image dimensions.

## Model Downloads

On first run the following models are downloaded from HuggingFace Hub and cached in `~/.cache/huggingface/hub`:

| Model | Usage | ~Size |
|-------|-------|-------|
| `Salesforce/blip-image-captioning-large` | Caption generation | ~1.9 GB |
| `openai/clip-vit-base-patch32` | Zero-shot categorization | ~600 MB |
| `Xenova/face-detection` | Face bounding boxes | ~6 MB |

## Notes

- CR3 decoding requires either `exiftool` or `dcraw` on `PATH`; without them CR3 files are skipped.
- All ML inference runs locally on CPU — no data leaves the machine after the initial model download.
- The `--skip-existing` flag is useful for incremental runs; `--force` overrides it.
