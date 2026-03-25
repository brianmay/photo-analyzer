use image::DynamicImage;

pub fn compute_exposure(img: &DynamicImage) -> f32 {
    let rgb = img.to_rgb8();
    let total_pixels = rgb.width() * rgb.height();

    if total_pixels == 0 {
        return 0.5;
    }

    let mut histogram = [0u32; 256];
    for pixel in rgb.pixels() {
        let luma = (0.299 * pixel[0] as f32 + 0.587 * pixel[1] as f32 + 0.114 * pixel[2] as f32)
            as usize;
        histogram[luma.min(255)] += 1;
    }

    let black_clipped: u32 = histogram[..5].iter().sum();
    let white_clipped: u32 = histogram[251..].iter().sum();

    let black_ratio = black_clipped as f32 / total_pixels as f32;
    let white_ratio = white_clipped as f32 / total_pixels as f32;

    let black_penalty = if black_ratio > 0.05 {
        (black_ratio - 0.05) * 10.0
    } else {
        0.0
    };
    let white_penalty = if white_ratio > 0.05 {
        (white_ratio - 0.05) * 10.0
    } else {
        0.0
    };

    let mean_luma: f32 = histogram
        .iter()
        .enumerate()
        .map(|(i, &count)| i as f32 * count as f32)
        .sum::<f32>()
        / total_pixels as f32;

    let center_score = 1.0 - (mean_luma - 128.0).abs() / 128.0;

    (center_score - black_penalty - white_penalty).clamp(0.0, 1.0)
}
