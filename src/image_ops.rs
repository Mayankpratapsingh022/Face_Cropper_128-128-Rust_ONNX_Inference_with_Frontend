// src/image_ops.rs

use image::{DynamicImage, GenericImageView};
use image::imageops::{FilterType, brighten, flip_horizontal_in_place, resize};
use rand::Rng;

/// Given:
///  - `orig`: the original full‐size image,
///  - `rect`: (x1, y1, x2, y2) in original coordinates,
///  - `brightness_jitter`: ±percentage to randomly adjust brightness,
/// Returns a 128×128 DynamicImage that has:
///  1) been cropped to `rect`,
///  2) randomly horizontally flipped (50% chance),
///  3) had brightness jitter (±%),
///  4) been resized (distorted if necessary) exactly to 128×128.
pub fn crop_and_augment(
    orig: &DynamicImage,
    rect: (u32, u32, u32, u32),
    brightness_jitter: i32,
) -> DynamicImage {
    let (x1, y1, x2, y2) = rect;
    let w = x2.saturating_sub(x1);
    let h = y2.saturating_sub(y1);

    // 1) Crop the face region
    let mut face_crop = orig.crop_imm(x1, y1, w, h);

    // 2) Random horizontal flip (50% chance)
    let mut rng = rand::thread_rng();
    if rng.gen_bool(0.5) {
        flip_horizontal_in_place(&mut face_crop);
    }

    // 3) Brightness jitter ± brightness_jitter%
    if brightness_jitter != 0 {
        let jitter_val = rng.gen_range(-brightness_jitter..=brightness_jitter);
        let bright_buf = brighten(&face_crop, jitter_val);
        face_crop = DynamicImage::ImageRgba8(bright_buf);
    }

    // 4) Resize (distort if needed) to exactly 128×128
    let resized = resize(&face_crop, 128, 128, FilterType::CatmullRom);
    DynamicImage::ImageRgba8(resized)
}
