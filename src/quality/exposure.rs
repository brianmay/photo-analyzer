use image::DynamicImage;

/// Luminance values 0–4 are treated as clipped blacks.
const BLACK_CLIP_END: usize = 5;
/// Luminance values 251–255 are treated as clipped whites.
const WHITE_CLIP_START: usize = 251;

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

    let black_clipped: u32 = histogram[..BLACK_CLIP_END].iter().sum();
    let white_clipped: u32 = histogram[WHITE_CLIP_START..].iter().sum();

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
