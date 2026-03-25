use image::DynamicImage;

pub fn compute_aesthetic(img: &DynamicImage, sharpness: f32, exposure: f32, noise: f32) -> f32 {
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();

    if width == 0 || height == 0 {
        return 0.5;
    }

    let mut saturation_sum = 0.0f32;
    let mut pixel_count = 0u32;

    for pixel in rgb.pixels() {
        let r = pixel[0] as f32 / 255.0;
        let g = pixel[1] as f32 / 255.0;
        let b = pixel[2] as f32 / 255.0;

        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let delta = max - min;

        let sat = if max > 0.0 { delta / max } else { 0.0 };
        saturation_sum += sat;
        pixel_count += 1;
    }

    let avg_saturation = if pixel_count > 0 {
        saturation_sum / pixel_count as f32
    } else {
        0.0
    };

    let color_score = (1.0 - (avg_saturation - 0.4).abs() * 2.0).clamp(0.0, 1.0);

    (color_score * 0.3 + sharpness * 0.3 + exposure * 0.25 + noise * 0.15)
        .clamp(0.0, 1.0)
}
