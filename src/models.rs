use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Category {
    pub name: String,
    pub confidence: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Person {
    pub name: String,
    pub confidence: f32,
    pub bounding_box: BoundingBox,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct QualityScore {
    pub overall: f32,
    pub sharpness: f32,
    pub exposure: f32,
    pub noise: f32,
    pub composition: f32,
    pub aesthetic: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AnalysisResult {
    pub file: String,
    pub title: String,
    pub description: String,
    pub categories: Vec<Category>,
    pub people: Vec<Person>,
    pub quality: QualityScore,
    pub analyzed_at: DateTime<Utc>,
}
