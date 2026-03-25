use crate::models::BoundingBox;

/// Max normalized distance from a rule-of-thirds intersection to earn a bonus.
const THIRDS_PROXIMITY: f32 = 0.15;
/// Bonus when a face is near a rule-of-thirds intersection point.
const THIRDS_BONUS: f32 = 0.15;
/// Max normalized distance from center to apply the "too centered" penalty.
const CENTER_PROXIMITY: f32 = 0.05;
/// Penalty for a face being at the exact center (static/dull composition).
const CENTER_PENALTY: f32 = 0.1;
/// Normalized Y threshold below which a face is considered cropped at the top.
const TOP_CROP_THRESHOLD: f32 = 0.02;
/// Penalty for a face being cropped at the top edge.
const TOP_CROP_PENALTY: f32 = 0.15;
/// Normalized Y range considered good vertical placement for portrait faces.
const GOOD_PLACEMENT_Y_MIN: f32 = 0.05;
const GOOD_PLACEMENT_Y_MAX: f32 = 0.3;
/// Bonus for a face in the preferred vertical zone.
const GOOD_PLACEMENT_BONUS: f32 = 0.1;
/// Tolerance for detecting faces that are clipped at any edge.
const EDGE_CLIP_MARGIN: f32 = 0.01;
/// Penalty for a face that is clipped at any image edge.
const EDGE_CLIP_PENALTY: f32 = 0.2;

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

        if min_dist < THIRDS_PROXIMITY {
            score += THIRDS_BONUS;
        }

        let center_dist = ((cx - 0.5).powi(2) + (cy - 0.5).powi(2)).sqrt();
        if center_dist < CENTER_PROXIMITY {
            score -= CENTER_PENALTY;
        }

        if face.y < TOP_CROP_THRESHOLD {
            score -= TOP_CROP_PENALTY;
        } else if (GOOD_PLACEMENT_Y_MIN..GOOD_PLACEMENT_Y_MAX).contains(&cy) {
            score += GOOD_PLACEMENT_BONUS;
        }

        let right_edge = face.x + face.width;
        let bottom_edge = face.y + face.height;
        if face.x < EDGE_CLIP_MARGIN
            || right_edge > 1.0 - EDGE_CLIP_MARGIN
            || face.y < EDGE_CLIP_MARGIN
            || bottom_edge > 1.0 - EDGE_CLIP_MARGIN
        {
            score -= EDGE_CLIP_PENALTY;
        }
    }

    score.clamp(0.0, 1.0)
}
