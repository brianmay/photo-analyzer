use anyhow::Result;
use image::DynamicImage;
use crate::models::Person;

pub struct FaceDetector;

impl FaceDetector {
    pub fn load() -> Result<Self> {
        Ok(Self)
    }

    pub fn detect_faces(&mut self, _img: &DynamicImage) -> Result<Vec<Person>> {
        Ok(vec![])
    }
}
