use anyhow::{Context, Result};
use image::DynamicImage;
use std::io::Cursor;
use std::path::Path;
use std::process::Command;

pub fn decode_image(path: &Path) -> Result<DynamicImage> {
    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default();

    match ext.as_str() {
        "cr3" => decode_cr3(path),
        _ => {
            image::open(path).with_context(|| format!("Failed to open image: {}", path.display()))
        }
    }
}

fn decode_cr3(path: &Path) -> Result<DynamicImage> {
    let path_str = path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("Path contains invalid UTF-8: {}", path.display()))?;

    let output = Command::new("exiftool")
        .args(["-b", "-PreviewImage", path_str])
        .output();

    if let Ok(output) = output {
        if output.status.success() && !output.stdout.is_empty() {
            let cursor = Cursor::new(output.stdout);
            if let Ok(img) = image::load(cursor, image::ImageFormat::Jpeg) {
                return Ok(img);
            }
        }
    }

    let output = Command::new("dcraw")
        .args(["-e", "-c", path_str])
        .output();

    if let Ok(output) = output {
        if output.status.success() && !output.stdout.is_empty() {
            let cursor = Cursor::new(output.stdout);
            if let Ok(img) = image::load(cursor, image::ImageFormat::Jpeg) {
                return Ok(img);
            }
        }
    }

    anyhow::bail!(
        "Failed to decode CR3 file: {}. Please install exiftool or dcraw.",
        path.display()
    )
}
