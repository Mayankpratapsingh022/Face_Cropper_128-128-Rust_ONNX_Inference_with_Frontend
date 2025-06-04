// src/detection.rs

use ndarray::{Array, CowArray, IxDyn};
use ort::{Environment, Session, SessionBuilder, Value};
use std::path::Path;
use std::sync::Arc;

/// Run ONNX face‐detection on a batch of `[B,3,640,640]` tensors.
///
/// # Arguments
/// - `session`: an already‐built ONNX Runtime session
/// - `batch_arr`: an `IxDyn` array of shape `[B, 3, 640, 640]`
///
/// # Returns
/// A flat `Vec<f32>` containing the raw output. The caller can reshape / split per image.
pub fn run_onnx(
    session: &Session,
    batch_arr: &Array<f32, IxDyn>,
) -> Vec<f32> {
    // Convert our owned Array into a CowArray (owned variant).
    let cow: CowArray<'_, f32, IxDyn> = CowArray::from(batch_arr.clone());

    // Convert to ONNX Value
    let ort_value = Value::from_array(session.allocator(), &cow)
        .expect("Failed to convert ndarray to ONNX Value");

    // Run inference
    let outputs = session
        .run(vec![ort_value])
        .expect("ONNX model inference failed");

    // We only care about `outputs[0]` here (the face‐detection output).
    let tensor = outputs[0]
        .try_extract::<f32>()
        .expect("Failed to extract f32 from ONNX output");

    // Copy data into a flat Vec<f32> by first getting an ArrayView, then iterating.
    tensor.view().iter().cloned().collect()
}


/// Build an ONNX `Session` from a given model file path (panics on error).
pub fn load_model(model_path: &Path, env: &Arc<Environment>) -> Session {
    SessionBuilder::new(env)
        .unwrap()
        .with_model_from_file(model_path)
        .unwrap_or_else(|e| panic!("Failed to load ONNX model from {:?}: {:?}", model_path, e))
}
