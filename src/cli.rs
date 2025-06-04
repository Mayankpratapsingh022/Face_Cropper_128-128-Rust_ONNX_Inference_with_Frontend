// src/cli.rs

use std::path::PathBuf;
use clap::Parser; // ← needed so that `Args::parse()` is in scope

/// CLI arguments for the face dataset generator
#[derive(Parser, Debug)]
#[command(
    name = "face_cropper",
    version,
    about = "Generates 128×128 face crops from a folder of photos using an ONNX face‐detection model."
)]
pub struct Args {
    /// Path to the input directory containing images (JPG, JPEG, PNG, BMP)
    #[arg(long, short)]
    pub input_dir: PathBuf,

    /// Path to the output directory where cropped faces will be saved
    #[arg(long, short)]
    pub output_dir: PathBuf,

    /// Path to the ONNX face‐detection model (e.g., model.onnx)
    #[arg(long, short)]
    pub model: PathBuf,

    /// Batch size for ONNX inference (default: 4)
    #[arg(long, default_value_t = 4)]
    pub batch_size: usize,

    /// Confidence threshold (keep detections with conf ≥ this) (default: 0.5)
    #[arg(long, default_value_t = 0.5)]
    pub conf_threshold: f32,

    /// IoU threshold for Non‐Maximum Suppression (default: 0.7)
    #[arg(long, default_value_t = 0.7)]
    pub iou_threshold: f32,

    /// Brightness jitter percentage (± value). Set to 0 to disable (default: 20)
    #[arg(long, default_value_t = 20)]
    pub brightness_jitter: i32,

    /// Do NOT recurse into subdirectories (default: false). Provide this flag to disable recursion.
    #[arg(long, default_value_t = false, action = clap::ArgAction::SetTrue)]
    pub no_recursive: bool,
}
