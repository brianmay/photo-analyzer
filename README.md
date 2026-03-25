# photo-analyzer

A command-line tool written in Rust that scans directories for photos (JPEG, CR3) and performs local analysis to generate titles, descriptions, quality scores, and category metadata — writing results as JSON sidecar files alongside each photo.

## Features

- **Photo scanning** – recursively or shallowly finds `.jpg`, `.jpeg`, and `.cr3` files
- **CR3 support** – decodes Canon CR3 raw files via `exiftool` or `dcraw` (optional)
- **Quality analysis** – sharpness (Laplacian variance), exposure (histogram), noise (Gaussian residual MAD), composition (rule-of-thirds), and aesthetic scoring
- **ML pipeline** – pluggable captioning, zero-shot categorization, and face detection (stub implementations included; replace with real models as needed)
- **JSON sidecar output** – results written to `<photo>.analysis.json` next to each image
- **Parallel processing** – uses Rayon for multi-core photo analysis

## Prerequisites

- **Rust 1.70+** – Install from [rustup.rs](https://rustup.rs)
- **exiftool** *(optional, for CR3 support)* – `sudo apt install libimage-exiftool-perl` or `brew install exiftool`
- **dcraw** *(optional, fallback for CR3)* – `sudo apt install dcraw` or `brew install dcraw`

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
```

## JSON Output Format

Each analyzed photo gets a `<photo>.analysis.json` sidecar:

```json
{
  "file": "DSC_0001.jpg",
  "title": "Untitled Photo",
  "description": "A photo requiring ML analysis.",
  "categories": [],
  "people": [],
  "quality": {
    "overall": 0.62,
    "sharpness": 0.74,
    "exposure": 0.81,
    "noise": 0.90,
    "composition": 0.50,
    "aesthetic": 0.55
  },
  "analyzed_at": "2024-01-15T10:30:00Z"
}
```

All quality scores are in the range `[0.0, 1.0]` (higher is better).

## Notes

- ML captioning, categorization, and face detection are currently stub implementations that return placeholder values. Replace the modules in `src/ml/` with real model integrations (e.g., BLIP via candle-transformers, CLIP for zero-shot classification, ONNX face detection via ORT) when running in a networked environment.
- CR3 decoding requires either `exiftool` or `dcraw` to be installed and available on `PATH`.
