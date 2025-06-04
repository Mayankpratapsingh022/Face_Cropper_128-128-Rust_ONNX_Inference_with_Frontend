// src/main.rs

mod cli;
mod detection;
mod image_ops;
mod nms;
mod utils;

use clap::Parser;               // ← so that `Args::parse()` is in scope
use image::{DynamicImage, GenericImageView};
use image::imageops::FilterType; // ← for `.resize_exact(...)`
use ndarray::{Array, Axis, IxDyn};
use ort::{Environment, Value};  // ← so that `Value::from_array(...)` is in scope
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use walkdir::WalkDir;

use log::{info, warn, error};
use rayon::prelude::*;

use cli::Args;
use detection::{run_onnx, load_model};
use image_ops::crop_and_augment;
use nms::iou;
use utils::is_image_file;

fn main() {
    // 1) Parse command‐line arguments
    let args = Args::parse();

    // 2) Initialize the logger (control verbosity with RUST_LOG)
    env_logger::init();

    // 3) Create the output directory if it doesn't exist
    if let Err(e) = fs::create_dir_all(&args.output_dir) {
        error!("Failed to create output directory {:?}: {}", args.output_dir, e);
        std::process::exit(1);
    }

    // 4) Initialize ONNX Runtime and load the face detection model
    let ort_env = Arc::new(
        Environment::builder()
            .with_name("face_cropper")
            .build()
            .expect("Failed to create ORT environment"),
    );
    let session = load_model(&args.model, &ort_env);
    info!("✅ Loaded ONNX model from {:?}", args.model);

    // 5) Collect all image paths (conditionally recursive)
    let mut image_paths: Vec<PathBuf> = Vec::new();
    if !args.no_recursive {
        for entry in WalkDir::new(&args.input_dir)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path: PathBuf = entry.path().to_path_buf();
            if path.is_file() && is_image_file(&path) {
                image_paths.push(path);
            }
        }
    } else {
        for entry in WalkDir::new(&args.input_dir)
            .min_depth(1)
            .max_depth(1)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path: PathBuf = entry.path().to_path_buf();
            if path.is_file() && is_image_file(&path) {
                image_paths.push(path);
            }
        }
    }

    if image_paths.is_empty() {
        warn!("No image files found under {:?}", args.input_dir);
        std::process::exit(0);
    }
    info!(
        "Found {} image(s) under {:?} (recursive={})",
        image_paths.len(),
        args.input_dir,
        !args.no_recursive
    );

    // 6) Process all images in parallel, in chunks of `batch_size`
    let total_faces: usize = image_paths
        .chunks(args.batch_size)
        .map(|chunk| chunk.to_vec())
        .collect::<Vec<_>>() // Vec<Vec<PathBuf>>
        .par_iter()
        .map(|chunk_vec| {
            process_image_batch(
                &session,
                chunk_vec,
                &args.output_dir,
                args.conf_threshold,
                args.iou_threshold,
                args.brightness_jitter,
            )
        })
        .sum();

    info!("✅ Finished. Total faces cropped: {}", total_faces);
}

/// Processes one batch of images (≤ `batch_paths.len()`) and returns
/// how many 128×128 face crops were saved.
fn process_image_batch(
    session: &ort::Session,
    batch_paths: &[PathBuf],
    output_dir: &PathBuf,
    conf_threshold: f32,
    iou_threshold: f32,
    brightness_jitter: i32,
) -> usize {
    // 1) Load + preprocess each image: resize to 640×640 and convert to [1,3,640,640] tensor
    let mut dyn_images: Vec<DynamicImage> = Vec::new();
    let mut orig_sizes: Vec<(u32, u32)> = Vec::new();
    let mut input_tensors: Vec<ndarray::Array<f32, IxDyn>> = Vec::new();

    for path in batch_paths {
        match image::open(path) {
            Ok(img) => {
                let (w, h) = (img.width(), img.height());
                let resized = img.resize_exact(640, 640, FilterType::CatmullRom);
                let mut arr = Array::zeros((1, 3, 640, 640)).into_dyn();
                for (x, y, pixel) in resized.pixels() {
                    let [r, g, b, _] = pixel.0;
                    arr[[0, 0, y as usize, x as usize]] = (r as f32) / 255.0;
                    arr[[0, 1, y as usize, x as usize]] = (g as f32) / 255.0;
                    arr[[0, 2, y as usize, x as usize]] = (b as f32) / 255.0;
                }
                dyn_images.push(img);
                orig_sizes.push((w, h));
                input_tensors.push(arr);
            }
            Err(e) => {
                warn!("Failed to open image {:?}: {}", path, e);
            }
        }
    }

    // If no valid images loaded, skip
    if dyn_images.is_empty() {
        return 0;
    }

    // 2) Combine single-image tensors [1,3,640,640] → batch [B,3,640,640]
    let actual_batch_size = dyn_images.len();
    let mut batched = Array::zeros((actual_batch_size, 3, 640, 640)).into_dyn();
    for (i, single) in input_tensors.into_iter().enumerate() {
        let slice = single.into_shape((3, 640, 640)).unwrap();
        for c in 0..3 {
            for y in 0..640 {
                for x in 0..640 {
                    batched[[i, c, y, x]] = slice[[c, y, x]];
                }
            }
        }
    }

    // 3) Run ONNX inference on the batch
    let standard_array = batched.as_standard_layout();
    let input_tensor =
        Value::from_array(session.allocator(), &standard_array).expect("Tensor conversion failed");
    let outputs = session
        .run(vec![input_tensor])
        .expect("ONNX model inference failed");

    // 4) Extract raw output: could be [N,5,1] or [N,5]
    let raw_all = outputs[0]
        .try_extract::<f32>()
        .unwrap()
        .view()
        .t()
        .to_owned();
    let shape = raw_all.shape().to_vec();
    let raw_2d: ndarray::Array<f32, IxDyn> = if shape.len() == 3 && shape[2] == 1 {
        let n = shape[0];
        let c = shape[1];
        raw_all
            .into_shape(IxDyn(&[n, c]))
            .expect("Failed to reshape/squeeze ONNX output")
    } else if shape.len() == 2 {
        raw_all
    } else {
        panic!("Unexpected ONNX output shape: {:?}", shape);
    };

    // 5) Split predictions per image
    let total_preds = raw_2d.shape()[0];
    let rows_per_image = if total_preds % actual_batch_size == 0 {
        total_preds / actual_batch_size
    } else {
        total_preds / actual_batch_size
    };

    let mut per_image_raw: Vec<Vec<Vec<f32>>> = vec![Vec::new(); actual_batch_size];
    for (idx, row) in raw_2d.axis_iter(Axis(0)).enumerate() {
        let img_index = idx / rows_per_image;
        if img_index < actual_batch_size {
            per_image_raw[img_index].push(row.iter().cloned().collect());
        }
    }

    // 6) For each image: NMS, clamp boxes, augment, crop & resize to 128×128
    let mut saved_count = 0;
    for (i, path) in batch_paths.iter().enumerate() {
        let raw_rows = &per_image_raw[i];
        let (orig_w, orig_h) = orig_sizes[i];

        // Collect candidate detections
        let mut candidates = Vec::new();
        for vals in raw_rows.iter() {
            if vals.len() < 5 {
                continue;
            }
            let cx = vals[0];
            let cy = vals[1];
            let w  = vals[2];
            let h  = vals[3];
            let conf = vals[4];

            if conf < conf_threshold {
                continue;
            }
            // Scale from 640×640 back to original image size
            let x1 = (cx - w / 2.0) / 640.0 * (orig_w as f32);
            let y1 = (cy - h / 2.0) / 640.0 * (orig_h as f32);
            let x2 = (cx + w / 2.0) / 640.0 * (orig_w as f32);
            let y2 = (cy + h / 2.0) / 640.0 * (orig_h as f32);
            candidates.push((x1, y1, x2, y2, conf));
        }

        if candidates.is_empty() {
            warn!("No faces detected in {:?}. Skipping.", path);
            continue;
        }

        // Non‐Maximum Suppression (NMS)
        candidates.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap());
        let mut final_faces = Vec::new();
        let mut buffer = candidates;
        while let Some((x1, y1, x2, y2, conf)) = buffer.first().cloned() {
            final_faces.push((x1, y1, x2, y2, conf));
            let mut keep = Vec::new();
            for det in buffer.iter().skip(1) {
                let iou_val = iou((x1, y1, x2, y2), (det.0, det.1, det.2, det.3));
                if iou_val < iou_threshold {
                    keep.push(*det);
                }
            }
            buffer = keep;
        }

        // Crop, augment, and resize each face → 128×128
        for (face_idx, &(x1, y1, x2, y2, conf)) in final_faces.iter().enumerate() {
            // Clamp to valid pixel coords
            let xi1 = x1.clamp(0.0, (orig_w - 1) as f32).round() as u32;
            let yi1 = y1.clamp(0.0, (orig_h - 1) as f32).round() as u32;
            let xi2 = x2.clamp(0.0, (orig_w - 1) as f32).round() as u32;
            let yi2 = y2.clamp(0.0, (orig_h - 1) as f32).round() as u32;

            if xi2 <= xi1 || yi2 <= yi1 {
                warn!(
                    "Degenerate box for {:?} face {}: [{}, {}, {}, {}]. Skipping.",
                    path, face_idx, xi1, yi1, xi2, yi2
                );
                continue;
            }

            // Crop + augment + resize → 128×128
            let face_128 = crop_and_augment(
                &dyn_images[i],
                (xi1, yi1, xi2, yi2),
                brightness_jitter,
            );

            // Save to disk
            let stem = path
                .file_stem()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown");
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("jpg");
            let out_name = format!("{}_{}.{}", stem, face_idx, ext);
            let mut out_path = output_dir.clone();
            out_path.push(out_name);

            if let Err(e) = face_128.save(&out_path) {
                error!("Failed to save {:?}: {}", out_path, e);
            } else {
                info!(
                    "Saved face #{} from {:?} → [{}, {}, {}, {}], conf {:.2} → {:?} (128×128)",
                    face_idx, path, xi1, yi1, xi2, yi2, conf, out_path
                );
                saved_count += 1;
            }
        }
    }

    saved_count
}
