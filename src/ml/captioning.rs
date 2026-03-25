use anyhow::Result;
use image::DynamicImage;

pub struct CaptioningModel;

impl CaptioningModel {
    pub fn load() -> Result<Self> {
        Ok(Self)
    }

    pub fn caption(&mut self, _img: &DynamicImage) -> Result<(String, String)> {
        Ok((
            "Untitled Photo".to_string(),
            "A photo requiring ML analysis.".to_string(),
        ))
    }
}
