use anyhow::Result;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// RAW formats that should be skipped when a JPEG counterpart exists.
const RAW_EXTENSIONS: &[&str] = &["cr2", "cr3", "nef", "arw", "raf", "orf", "dng", "rw2"];
/// JPEG extensions that are considered "preferred" over a RAW file.
const JPEG_EXTENSIONS: &[&str] = &["jpg", "jpeg"];

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
                if JPEG_EXTENSIONS.contains(&ext.as_str())
                    || RAW_EXTENSIONS.contains(&ext.as_str())
                {
                    photos.push(path);
                }
            }
        }
    }

    // Build a set of (parent_dir, stem) pairs for every JPEG found.
    // Any RAW file whose (parent_dir, stem) is already covered by a JPEG is skipped.
    let jpeg_keys: HashSet<(PathBuf, String)> = photos
        .iter()
        .filter(|p| {
            p.extension()
                .map(|e| JPEG_EXTENSIONS.contains(&e.to_string_lossy().to_lowercase().as_str()))
                .unwrap_or(false)
        })
        .filter_map(|p| {
            let stem = p.file_stem()?.to_string_lossy().to_string();
            let parent = p.parent()?.to_path_buf();
            Some((parent, stem))
        })
        .collect();

    photos.retain(|p| {
        let ext = p
            .extension()
            .map(|e| e.to_string_lossy().to_lowercase())
            .unwrap_or_default();
        if !RAW_EXTENSIONS.contains(&ext.as_str()) {
            return true; // always keep JPEGs
        }
        // For RAW files: keep only if no JPEG counterpart exists.
        // If we can't determine parent or stem, keep the file to be safe.
        let Some(stem) = p.file_stem().map(|s| s.to_string_lossy().to_string()) else {
            return true;
        };
        let Some(parent) = p.parent().map(|q| q.to_path_buf()) else {
            return true;
        };
        !jpeg_keys.contains(&(parent, stem))
    });

    Ok(photos)
}
