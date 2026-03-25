use image::DynamicImage;

pub fn compute_sharpness(img: &DynamicImage) -> f32 {
    let gray = img.to_luma8();
    let (width, height) = gray.dimensions();

    if width < 3 || height < 3 {
        return 0.0;
    }

    let kernel: [f32; 9] = [0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0];

    let mut values = Vec::with_capacity(((width - 2) * (height - 2)) as usize);

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let mut sum = 0.0f32;
            for ky in 0..3usize {
                for kx in 0..3usize {
                    let px = gray.get_pixel(x + kx as u32 - 1, y + ky as u32 - 1)[0] as f32;
                    sum += px * kernel[ky * 3 + kx];
                }
            }
            values.push(sum);
        }
    }

    if values.is_empty() {
        return 0.0;
    }

    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;

    (variance / 500.0).min(1.0)
}
