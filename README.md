# Face Cropper: 128x128 Face Dataset Generator in Rust
![1](https://github.com/user-attachments/assets/17db8612-91ee-467c-a4f8-83ddf25382d7)
This repository provides a complete pipeline to generate face image datasets from a batch of images using YOLOv11-face-detection ONNX inference in Rust. It includes:

- A high-performance Rust binary (`face_cropper`) for batch inference and face cropping.

- A portable Linux package (`face_cropper_linux_package`) with ONNX runtime pre-bundled.

- A Streamlit-based frontend UI to run face cropping from the browser.

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
![MacBook Air - 79](https://github.com/user-attachments/assets/03d1cc8f-228c-4f47-856f-3c55c7ed1241)

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

#### Step 1: Navigate to the frontend directory

```bash
cd ui_frontend
```

#### Step 2: Install required Python dependencies

```bash
pip install -r requirements.txt
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

Here’s a more structured, clean, and professional version of the **Technologies Used** section for your `README.md`:

---

## Technologies Used

The project leverages the following tools and libraries for efficient face detection, cropping, and visualization:

| Technology               | Description                                                                  |
| ------------------------ | ---------------------------------------------------------------------------- |
| **Rust (2021 Edition)**  | Core implementation for high-performance inference and image processing      |
| **ONNX Runtime 1.16.0**  | Used for model inference via the [`ort`](https://crates.io/crates/ort) crate |
| **YOLOv11-Face Model**    | Pre-trained face detection model exported in ONNX format                     |
| **Streamlit** (optional) | Web-based UI for drag-and-drop image upload and preview                      |

---



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


   
