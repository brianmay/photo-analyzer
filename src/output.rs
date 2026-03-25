use anyhow::{Context, Result};
use std::path::Path;
use crate::models::AnalysisResult;

pub fn write_json(image_path: &Path, result: &AnalysisResult) -> Result<()> {
    let stem = image_path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let json_path = image_path
        .with_file_name(format!("{stem}.analysis.json"));
    let json = serde_json::to_string_pretty(result)
        .context("Failed to serialize analysis result")?;
    std::fs::write(&json_path, json)
        .with_context(|| format!("Failed to write JSON to: {}", json_path.display()))?;
    Ok(())
}
