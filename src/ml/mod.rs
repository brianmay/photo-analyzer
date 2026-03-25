pub mod captioning;
pub mod categorization;
pub mod face_detection;

use anyhow::Result;
use image::DynamicImage;

use crate::models::{Category, Person};
use captioning::CaptioningModel;
use categorization::CategorizationModel;
use face_detection::FaceDetector;

pub struct MlModels {
    pub captioning: CaptioningModel,
    pub categorization: CategorizationModel,
    pub face_detector: FaceDetector,
}

impl MlModels {
    pub fn load() -> Result<Self> {
        tracing::info!("Loading ML models...");

        let captioning = CaptioningModel::load()?;
        tracing::info!("Captioning model loaded");

        let categorization = CategorizationModel::load()?;
        tracing::info!("Categorization model loaded");

        let face_detector = FaceDetector::load()?;
        tracing::info!("Face detection model loaded");

        Ok(Self {
            captioning,
            categorization,
            face_detector,
        })
    }

    pub fn analyze(
        &mut self,
        img: &DynamicImage,
        category_threshold: f32,
    ) -> Result<(String, String, Vec<Category>, Vec<Person>)> {
        let (title, description) = self.captioning.caption(img)?;
        let categories = self.categorization.categorize(img, category_threshold)?;
        let people = self.face_detector.detect_faces(img)?;

        Ok((title, description, categories, people))
    }
}
