use crate::models::BoundingBox;

pub fn compute_composition(faces: &[BoundingBox]) -> f32 {

    if faces.is_empty() {
        return 0.5;
    }

    let mut score = 0.5f32;

    let thirds_x = [1.0f32 / 3.0, 2.0 / 3.0];
    let thirds_y = [1.0f32 / 3.0, 2.0 / 3.0];

    for face in faces {
        let cx = face.x + face.width / 2.0;
        let cy = face.y + face.height / 2.0;

        let mut min_dist = f32::MAX;
        for &tx in &thirds_x {
            for &ty in &thirds_y {
                let dist = ((cx - tx).powi(2) + (cy - ty).powi(2)).sqrt();
                min_dist = min_dist.min(dist);
            }
        }

        if min_dist < 0.15 {
            score += 0.15;
        }

        let center_dist = ((cx - 0.5).powi(2) + (cy - 0.5).powi(2)).sqrt();
        if center_dist < 0.05 {
            score -= 0.1;
        }

        if face.y < 0.02 {
            score -= 0.15;
        } else if face.y > 0.05 && face.y < 0.3 {
            score += 0.1;
        }

        let right_edge = face.x + face.width;
        let bottom_edge = face.y + face.height;
        if face.x < 0.01 || right_edge > 0.99 || face.y < 0.01 || bottom_edge > 0.99 {
            score -= 0.2;
        }
    }

    score.clamp(0.0, 1.0)
}
