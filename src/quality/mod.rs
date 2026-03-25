use crate::models::{BoundingBox, QualityScore};
use image::DynamicImage;

pub mod aesthetic;
pub mod composition;
pub mod exposure;
pub mod noise;
pub mod sharpness;

// Composite quality score weights (must sum to 1.0).
const SHARPNESS_WEIGHT: f32 = 0.25;
const EXPOSURE_WEIGHT: f32 = 0.15;
const NOISE_WEIGHT: f32 = 0.15;
const COMPOSITION_WEIGHT: f32 = 0.20;
const AESTHETIC_WEIGHT: f32 = 0.25;

pub fn compute_quality(img: &DynamicImage, faces: &[BoundingBox]) -> QualityScore {

    let sharpness = sharpness::compute_sharpness(img);
    let exposure = exposure::compute_exposure(img);
    let noise = noise::compute_noise(img);
    let composition = composition::compute_composition(faces);
    let aesthetic = aesthetic::compute_aesthetic(img, sharpness, exposure, noise);

    let overall = sharpness * SHARPNESS_WEIGHT
        + exposure * EXPOSURE_WEIGHT
        + noise * NOISE_WEIGHT
        + composition * COMPOSITION_WEIGHT
        + aesthetic * AESTHETIC_WEIGHT;

    QualityScore {
        overall: overall.clamp(0.0, 1.0),
        sharpness,
        exposure,
        noise,
        composition,
        aesthetic,
    }
}
