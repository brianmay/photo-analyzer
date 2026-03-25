use anyhow::Result;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

pub fn scan_directory(dir: &Path, recursive: bool) -> Result<Vec<PathBuf>> {
    let mut photos = Vec::new();
    let walker = if recursive {
        WalkDir::new(dir)
    } else {
        WalkDir::new(dir).max_depth(1)
    };

    for entry in walker.into_iter().filter_map(|e| e.ok()) {
        let path = entry.path().to_path_buf();
        if path.is_file() {
            if let Some(ext) = path.extension() {
                let ext = ext.to_string_lossy().to_lowercase();
                if matches!(ext.as_str(), "jpg" | "jpeg" | "cr3") {
                    photos.push(path);
                }
            }
        }
    }

    Ok(photos)
}
