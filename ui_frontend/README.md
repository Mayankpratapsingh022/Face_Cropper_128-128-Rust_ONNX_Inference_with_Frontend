# Face Cropper Frontend (Streamlit)

This is the frontend UI for the **128×128 Face Cropper** 
It allows users to upload image files, run face detection via a Rust-based backend, and download cropped face images.

---

## 🎯 Features

- Upload multiple images (JPG, JPEG, PNG, BMP)
- Detect and crop faces at **128×128** resolution
- Real-time scrollable status for each file processed
- Automatically skips images with no faces or errors
- Download all cropped faces as a single ZIP or individually
- Simple, fast UI — no emojis or unnecessary clutter

---

## 🛠 Requirements

- Python 3.8+
- Streamlit
- A compiled Rust binary: `face_cropper`
- ONNX model: `model.onnx` (face detection model)

---

## 📦 Installation

1. Clone this repository or copy the UI folder into your project directory.

2. Create a Python environment (optional but recommended):

```bash
conda create -n facecrop python=3.10
conda activate facecrop
