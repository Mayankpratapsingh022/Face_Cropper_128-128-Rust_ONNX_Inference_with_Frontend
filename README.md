# Face Cropper: 128x128 Face Dataset Generator in Rust
![1](https://github.com/user-attachments/assets/17db8612-91ee-467c-a4f8-83ddf25382d7)
This repository provides a complete pipeline to generate face image datasets from a batch of images using YOLOv11-face-detection ONNX inference in Rust. It includes:


https://github.com/user-attachments/assets/71cac7ac-9a6a-4f74-bbdb-31bca9cc0927


- A high-performance Rust binary (`face_cropper`) for batch inference and face cropping.

- A portable Linux package (`face_cropper_linux_package`) with ONNX runtime pre-bundled.

- A Streamlit-based frontend UI to run face cropping from the browser.

Final Dataset [https://huggingface.co/datasets/Mayank022/Cropped_Face_Dataset_128x12](https://huggingface.co/datasets/Mayank022/Cropped_Face_Dataset_128x128)

## Quick UI Demonstration



https://github.com/user-attachments/assets/3af58bf1-6acb-4ee1-ac90-1cb992604575


## Project Structure

```bash
FACE_CROPPER_128x128_RUST_ONNX_INFERENCE_WITH_FRONTEND/
│
├── face_cropper_linux_package/        # Self-contained package for Linux users
│   ├── face_cropper                   # Compiled Rust binary (Linux)
│   ├── model.onnx                     # YOLOv11-face-detection ONNX face detection model
│   ├── input_images/                  # Directory for test input images
│   ├── faces_cropped/                 # Output directory for cropped face images
│   ├── runtimeLib/                    # Pre-bundled ONNX Runtime dependencies
│   └── run.sh                         # Shell script to execute the pipeline
│
├── ui_frontend/                       # Streamlit frontend for drag-and-drop UI
│   ├── app.py                         # Streamlit app entry point
│   └── README.md                      # Instructions for running the UI
│
├── src/                               # Core Rust backend source code
│   ├── main.rs                        # Main binary entry point
│   ├── detection.rs                   # Face detection logic using ONNX Runtime
│   ├── utils.rs                       # Utility functions
│   └── cli.rs                         # Defines and parses command-line arguments using clap.
│   └── nms.rs                         # Performs Non-Maximum Suppression to filter overlapping bounding boxes.
│
├── model.onnx                         # Shared ONNX model used by both backend/UI
├── Cargo.toml                         # Rust project dependencies and metadata
├── README.md                          # Project overview and setup instructions
```

## Edge Case Handling

The system is designed to robustly handle various edge cases during face detection and cropping. Below is a summary of handled cases, visual examples, and the corresponding strategies:

---

### 1. No Faces Detected
![2](https://github.com/user-attachments/assets/3f8713a8-44e9-44eb-89f8-cde57f4b8c93)

**Handling:** Skipped with a warning in logs
**Example:**


---

### 2. Multiple Faces in One Image
![3](https://github.com/user-attachments/assets/d76b6fcc-403e-448a-a382-99111056c642)

**Handling:** All faces are cropped and saved with indexed filenames
**Example:**


---

### 3. Bounding Boxes Partially Outside Image
![MacBook Air - 75](https://github.com/user-attachments/assets/9e9a49a5-48e0-48f9-ac61-44d3dc24e479)


**Handling:** Coordinates are clamped to stay within image bounds
**Example:**


---


### 4. Low Confidence Detections (< 0.5)
![4](https://github.com/user-attachments/assets/1e007c32-1759-47e5-a562-215b9ee099ba)

**Handling:** Filtered before cropping
**Example:**


---

### 5. Heavily Overlapping Boxes
![MacBook Air - 78](https://github.com/user-attachments/assets/d2ed2d06-179c-46ae-a6c1-cce480203080)

**Handling:** Removed using Non-Maximum Suppression (IoU > 0.7)
**Example:**

---

### 6. Invalid or Corrupt Input Files


**Handling:** Skipped silently unless log level is set to debug
**Example:**


---

## Using the Linux Binary Package (`face_cropper_linux_package`)
The `face_cropper_linux_package` folder contains everything you need to run the face detection and cropping tool without rebuilding from source.

###  Folder Structure
After cloning the repository or copying the package, the folder should look like:


```
face_cropper_linux_package/
├── face_cropper             # Precompiled Linux binary (128×128 face cropper)
├── model.onnx               # YOLOv8 ONNX face detection model
├── input_images/            # Folder containing input test images or folders as well
│   └── img1.jpg
│   └── img2.jpg
├── faces_cropped/           # Will be automatically filled with cropped face outputs
├── runtimeLib/              # ONNX Runtime shared libraries
│   ├── libonnxruntime.so.1.16.0
│   ├── libonnxruntime.so.1         ← symlink
│   └── libonnxruntime.so           ← symlink
└── run.sh                   # Bash script to run the face cropper
```



### One-Time Setup (First-Time Users)

> If you downloaded this repo on **Windows** and are now running it in **Linux**, line endings might be wrong.

Run this command **once** to fix the script line endings:

```bash
dos2unix run.sh
```

> If you're on WSL or using a GitHub clone inside Windows, this step is usually necessary.

---

### Make the Script Executable

The project provides two scripts (`run.sh` for single image processing & `run_multiple_images.sh` for batch processing) depending on whether you're processing a single image (from UI) or an entire folder of images (batch processing).



Run:

```bash
chmod +x run.sh
```

This gives the `run.sh` script permission to execute.

---

### Run the Face Cropper

Now simply run the script:

```bash
./run.sh
```

---

### What `run.sh` Does Internally

1. **Creates Symlinks** inside `runtimeLib/`:

   ```bash
   ln -sf libonnxruntime.so.1.16.0 libonnxruntime.so.1
   ln -sf libonnxruntime.so.1       libonnxruntime.so
   ```

   These help the binary dynamically load the correct shared object.


2. **Sets the Library Path**:

   ```bash
   export LD_LIBRARY_PATH=$(pwd)/runtimeLib:$LD_LIBRARY_PATH
   ```

   Ensures the binary can find the ONNX Runtime `.so` files.

3. **Executes the Binary** with CLI arguments:

   ```bash
   ./face_cropper \
     --input-dir ./input_images \
     --output-dir ./faces_cropped \
     --model ./model.onnx \
     --batch-size 10 \
     --conf-threshold 0.5 \
     --iou-threshold 0.5 \
     --brightness-jitter 30 \
     --no-recursive
   ```

---


###  `run_multiple_images.sh` – For Batch Processing





Use this script when you want to process an **entire folder of images** without using the UI. It runs inference on all valid images inside the `input_images/` directory and saves cropped faces to `faces_cropped/`.

#### Setup and Execution

```bash
dos2unix run_multiple_images.sh     # Optional, if cloned on Windows
chmod +x run_multiple_images.sh     # Make the script executable
./run_multiple_images.sh            # Run the script
```
https://github.com/user-attachments/assets/b9f35f14-4cd9-45c9-aba7-7414004fe263

Output images after running a script to rename in sequence
![image](https://github.com/user-attachments/assets/35d07fcd-a80f-4474-813c-2dbd731cdb0a)

## Frontend UI Setup on Linux Cloud (ONNX Runtime + Streamlit)

This section helps you set up and run the drag-and-drop Streamlit interface for the face cropper binary on a fresh Linux machine (tested on Amazon Linux 2023 EC2 instance).

---

### 1 Install Dependencies

```bash
sudo dnf install -y git wget dos2unix gcc gcc-c++ make libstdc++ python3 pip
sudo pip install --upgrade pip
pip install streamlit Pillow
```

---

### 2 Clone This Repository

```bash
git clone https://github.com/Mayankpratapsingh022/Face_Cropper_128-128-Rust_ONNX_Inference_with_Frontend
cd Face_Cropper_128-128-Rust_ONNX_Inference_with_Frontend
```

---

### 3 Prepare the Linux Binary Package

```bash
cd face_cropper_linux_package
dos2unix run.sh && chmod +x run.sh face_cropper
```

---

### 4 Install ONNX Runtime for Linux

```bash
cd runtimeLib
rm -f libonnxruntime.so libonnxruntime.so.1 libonnxruntime.so.1.16.0

wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-linux-x64-1.16.0.tgz
tar -xzf onnxruntime-linux-x64-1.16.0.tgz -C . \
  --strip-components=2 \
  onnxruntime-linux-x64-1.16.0/lib/libonnxruntime.so.1.16.0

# Create symlinks
ln -sf libonnxruntime.so.1.16.0 libonnxruntime.so.1
ln -sf libonnxruntime.so.1       libonnxruntime.so

cd ../..
```

---

### 5 Export Runtime Path

```bash
export LD_LIBRARY_PATH=$PWD/face_cropper_linux_package/runtimeLib:$LD_LIBRARY_PATH
```

---

### 6 Run CLI Test (Optional)

```bash
./face_cropper_linux_package/run.sh
```

---

### 7 Launch Streamlit Frontend

```bash
streamlit run ui_frontend/app.py --server.port 8501 --server.address 0.0.0.0
```

Make sure your EC2 security group allows inbound traffic on port `8501`.

---


## CLI Arguments – Full Control via Command Line


You can run the face cropper with granular control using CLI flags. Here’s an overview of all available options:

### Command

```bash
./face_cropper \
  --input-dir ./input_images \
  --output-dir ./faces_cropped \
  --model ./model.onnx \
  --batch-size 10 \
  --conf-threshold 0.5 \
  --iou-threshold 0.5 \
  --brightness-jitter 30 \
  --no-recursive
```
![image](https://github.com/user-attachments/assets/c32c99ed-380e-4024-9f23-1b548db2a9d2)
>  You can customize each parameter to control performance, detection accuracy, and output structure as needed.

---

###  Flag Reference Table

| Flag                  | Description                                                                          |
| --------------------- | ------------------------------------------------------------------------------------ |
| `--input-dir`         | Path to the directory containing the input images.                                   |
| `--output-dir`        | Path where cropped face images will be saved.                                        |
| `--model`             | Path to the ONNX model file used for face detection.                                 |
| `--batch-size`        | Number of images processed in one batch. Higher values may use more memory.          |
| `--conf-threshold`    | Minimum confidence score (0–1) to accept a detected face.                            |
| `--iou-threshold`     | IOU threshold for non-max suppression (NMS). Filters overlapping boxes.              |
| `--brightness-jitter` | Random brightness jitter added to output images (for augmentation).                  |
| `--no-recursive`      | Disables recursive folder traversal. Only files in the top-level directory are used. |

---

### Output

* Cropped faces are saved in the `faces_cropped/` folder as:

  ```
  img1_0.jpg, img1_1.jpg, img2_0.jpg, ...
  ```
* Each file is a `128×128` face crop.
* If no faces are detected, the original image is skipped.

---

### Troubleshooting

| Problem                                                          | Fix                                                                        |
| ---------------------------------------------------------------- | -------------------------------------------------------------------------- |
| `./run.sh: cannot execute: required file not found`              | Run `dos2unix run.sh` and `chmod +x run.sh`                                |
| `error while loading shared libraries: libonnxruntime.so.1.16.0` | Ensure you're running from inside the `face_cropper_linux_package` folder. |
| No output in `faces_cropped/`                                    | Check if faces are detectable and confidence thresholds are reasonable.    |

---


### Tip for Validation

You can test with 1–2 sample images inside `input_images/`, then inspect `faces_cropped/` to verify it works before running on larger batches.

---

### 3. Running the Fronted Streamlit UI
To launch the drag-and-drop face cropping interface:

https://github.com/user-attachments/assets/3af58bf1-6acb-4ee1-ac90-1cb992604575

#### Step 1: Navigate to the frontend directory

```bash
cd ui_frontend
```

#### Step 2: Install required Python dependencies

```bash
pip install streamlit>=1.25 Pillow>=9.0

```

#### Step 3: Start the Streamlit application

```bash
streamlit run app.py
```

Once started, open the following URL in your browser:

```
http://localhost:8501/
```

---

### Features Available in the UI:

* Upload one or more images, folders via drag-and-drop.
* Adjust detection thresholds to fine-tune face detection sensitivity.
* Automatically detect and crop faces to 128×128 resolution.
* Download the cropped faces directly from the interface.



## Technologies Used

The project leverages the following tools and libraries for efficient face detection, cropping, and visualization:

| Technology               | Description                                                                  |
| ------------------------ | ---------------------------------------------------------------------------- |
| **Rust**  | Core implementation for high-performance inference and image processing      |
| **ONNX Runtime 1.16.0**  | Used for model inference via the [`ort`](https://crates.io/crates/ort) crate |
| **YOLOv11-Face Model**    | Pre-trained face detection model exported in ONNX format                     |
| **Streamlit** (optional) | Web-based UI for drag-and-drop image upload and preview                      |

---

## Running the Face Cropper (Rust)

This section provides instructions for compiling and running the `face_cropper` binary from source using Rust.

### Prerequisites

To build and run the project, ensure the following tools are installed on your system:

- [Rust](https://www.rust-lang.org/tools/install) (with `cargo`)
- `libonnxruntime` shared libraries (Linux) or the ONNX Runtime DLLs (Windows)
- A working C++ toolchain (for `ort` crate)

You can verify Rust installation with:

```bash
rustc --version
cargo --version
````

### Project Structure

The binary entry point is defined in `src/main.rs` via the `[bin]` section in `Cargo.toml`:

```toml
[[bin]]
name = "face_cropper"
path = "src/main.rs"
```

### Cargo Dependencies

The project depends on the following crates:

```toml
[dependencies]
image        = "0.24.7"       # For image loading and resizing
ndarray      = "0.15.6"       # For constructing input tensors
ort          = "1.15.2"       # ONNX Runtime bindings
walkdir      = "2.3.3"        # Directory traversal
log          = "0.4"          # Logging macros
env_logger   = "0.10"         # Logger initialization
rayon        = "1.5"          # Parallel image processing
rand         = "0.8"          # Augmentations (e.g. flips)
clap         = { version = "4.1", features = ["derive"] } # Command-line interface
```

Make sure these are present in your `Cargo.toml`.

### Build Instructions

To build the optimized release binary:

```bash
cargo build --release
```

This will generate the binary at:

```
target/release/face_cropper
```

### Running the Binary

Once built, the binary can be executed using:

```bash
./target/release/face_cropper \
  --input-dir ./input_images \
  --output-dir ./faces_cropped \
  --model ./model.onnx \
  --batch-size 10 \
  --conf-threshold 0.5 \
  --iou-threshold 0.5 \
  --brightness-jitter 30 \
  --no-recursive
```




## Submission Notes (for Reviewers)

This submission delivers a complete face detection and cropping pipeline with a focus on performance, portability, and usability. The key components are summarized below:

### Core Features

* Full end-to-end pipeline implemented in Rust, including image loading, ONNX model inference, face detection, and image cropping.
* Integrated ONNX Runtime via the `ort` crate for efficient execution of the YOLOv11-Face model.
* Includes all critical steps: confidence thresholding, non-maximum suppression (NMS), bounding box clamping, and saving output crops.

### Packaging and Portability

* Portable Linux package:

  * Includes compiled binary, ONNX model, and bundled runtime.
  * Comes with symlinks and a `run.sh` script for seamless execution.

### Optional Frontend

* Streamlit-based user interface:

  * Provides a drag-and-drop experience for uploading images.
  * Allows users to adjust detection thresholds.
  * Displays and enables download of cropped face outputs.

### Edge Case Handling

* Robust handling of edge cases:

  * No faces detected
  * Multiple faces per image
  * Invalid or low-confidence boxes
  * Bounding boxes out of bounds
  * Corrupt or unsupported input files
* All cases are documented and, where appropriate, demonstrated with visual examples.

---


   
