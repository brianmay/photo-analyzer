use image::DynamicImage;

pub fn compute_noise(img: &DynamicImage) -> f32 {
    let gray = img.to_luma8();
    let (width, height) = gray.dimensions();

    if width < 5 || height < 5 {
        return 0.5;
    }

    let blurred = apply_gaussian_blur(&gray);

    let mut residuals: Vec<f32> = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            let orig = gray.get_pixel(x, y)[0] as f32;
            let blur = blurred[((y * width) + x) as usize];
            residuals.push((orig - blur).abs());
        }
    }

    residuals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = residuals[residuals.len() / 2];

    let noise_level = (median / 20.0).min(1.0);
    1.0 - noise_level
}

fn apply_gaussian_blur(gray: &image::GrayImage) -> Vec<f32> {
    let (width, height) = gray.dimensions();
    let kernel: [f32; 9] = [
        1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0,
        2.0 / 16.0, 4.0 / 16.0, 2.0 / 16.0,
        1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0,
    ];

    let mut output = vec![0.0f32; (width * height) as usize];

    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0f32;
            for ky in 0..3i32 {
                for kx in 0..3i32 {
                    let nx = (x as i32 + kx - 1).clamp(0, width as i32 - 1) as u32;
                    let ny = (y as i32 + ky - 1).clamp(0, height as i32 - 1) as u32;
                    let px = gray.get_pixel(nx, ny)[0] as f32;
                    sum += px * kernel[(ky * 3 + kx) as usize];
                }
            }
            output[((y * width) + x) as usize] = sum;
        }
    }
    output
}
