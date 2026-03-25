use anyhow::Result;
use image::DynamicImage;
use crate::models::Category;

pub struct CategorizationModel;

impl CategorizationModel {
    pub fn load() -> Result<Self> {
        Ok(Self)
    }

    pub fn categorize(&mut self, _img: &DynamicImage, _threshold: f32) -> Result<Vec<Category>> {
        Ok(vec![])
    }
}
