mod decoder;
mod ml;
mod models;
mod output;
mod quality;
mod scanner;

use anyhow::{Context, Result};
use clap::Parser;
use chrono::Utc;
use rayon::prelude::*;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tracing::{error, info, warn};

#[derive(Parser, Debug)]
#[command(
    name = "photo-analyzer",
    about = "Analyze photos using local ML models to generate titles, descriptions, categories, people detection, and quality scores"
)]
struct Args {
    /// Directories to scan for photos
    #[arg(required = true)]
    directories: Vec<PathBuf>,

    /// Scan directories recursively
    #[arg(short, long, default_value = "true")]
    recursive: bool,

    /// Number of parallel workers
    #[arg(short = 'j', long)]
    parallel: Option<usize>,

    /// Minimum confidence for category inclusion
    #[arg(long, default_value = "0.1")]
    category_threshold: f32,

    /// Skip photos that already have .analysis.json files
    #[arg(long, conflicts_with = "force")]
    skip_existing: bool,

    /// Re-analyze even if .analysis.json exists (overrides --skip-existing)
    #[arg(long)]
    force: bool,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let subscriber = tracing_subscriber::FmtSubscriber::builder()
        .with_env_filter(if args.verbose {
            "photo_analyzer=debug"
        } else {
            "photo_analyzer=info"
        })
        .finish();
    tracing::subscriber::set_global_default(subscriber)
        .context("Failed to set up logging")?;

    if let Some(n) = args.parallel {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .context("Failed to configure thread pool")?;
    }

    let mut all_photos = Vec::new();
    for dir in &args.directories {
        if !dir.exists() {
            warn!("Directory does not exist: {}", dir.display());
            continue;
        }
        match scanner::scan_directory(dir, args.recursive) {
            Ok(photos) => {
                info!("Found {} photos in {}", photos.len(), dir.display());
                all_photos.extend(photos);
            }
            Err(e) => {
                error!("Failed to scan directory {}: {}", dir.display(), e);
            }
        }
    }

    if all_photos.is_empty() {
        info!("No photos found to analyze");
        return Ok(());
    }

    if args.skip_existing {
        all_photos.retain(|p| {
            let stem = p.file_stem().map(|s| s.to_string_lossy()).unwrap_or_default();
            let json_path = p.with_file_name(format!("{stem}.analysis.json"));
            !json_path.exists()
        });
        info!("{} photos remaining after filtering existing", all_photos.len());
    }

    info!("Analyzing {} photos...", all_photos.len());

    let models = match ml::MlModels::load() {
        Ok(m) => Arc::new(Mutex::new(m)),
        Err(e) => {
            error!("Failed to load ML models: {}", e);
            return Err(e);
        }
    };

    let category_threshold = args.category_threshold;
    let success_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let error_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    let sc = Arc::clone(&success_count);
    let ec = Arc::clone(&error_count);

    all_photos.par_iter().for_each(|photo_path| {
        info!("Analyzing: {}", photo_path.display());

        let result = analyze_photo(photo_path, Arc::clone(&models), category_threshold);

        match result {
            Ok(_) => {
                sc.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                info!("Completed: {}", photo_path.display());
            }
            Err(e) => {
                ec.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                error!("Failed to analyze {}: {:#}", photo_path.display(), e);
            }
        }
    });

    info!(
        "Done! {} photos analyzed successfully, {} failed",
        success_count.load(std::sync::atomic::Ordering::Relaxed),
        error_count.load(std::sync::atomic::Ordering::Relaxed)
    );

    Ok(())
}

fn analyze_photo(
    path: &std::path::Path,
    models: Arc<Mutex<ml::MlModels>>,
    category_threshold: f32,
) -> Result<()> {
    let img = decoder::decode_image(path)
        .with_context(|| format!("Failed to decode image: {}", path.display()))?;

    let (title, description, categories, people) = {
        let mut locked_models = models.lock().unwrap();
        locked_models
            .analyze(&img, category_threshold)
            .with_context(|| format!("ML analysis failed for: {}", path.display()))?
    };

    let face_boxes: Vec<_> = people.iter().map(|p| p.bounding_box.clone()).collect();
    let quality = quality::compute_quality(&img, &face_boxes);

    let file_name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();

    let result = models::AnalysisResult {
        file: file_name,
        title,
        description,
        categories,
        people,
        quality,
        analyzed_at: Utc::now(),
    };

    output::write_json(path, &result)
        .with_context(|| format!("Failed to write JSON for: {}", path.display()))?;

    Ok(())
}
