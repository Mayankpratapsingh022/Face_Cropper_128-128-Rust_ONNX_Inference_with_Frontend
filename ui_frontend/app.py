import io
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
import streamlit as st

# Define binary and model paths
BINARY_PATH = "./face_cropper_linux_package/face_cropper"
MODEL_PATH = "./face_cropper_linux_package/model.onnx"

def save_uploaded_file(uploaded_file, input_path: Path):
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

# Initialize session state for persistent face storage
if "cropped_faces" not in st.session_state:
    st.session_state.cropped_faces = []
if "final_output_dir" not in st.session_state:
    st.session_state.final_output_dir = tempfile.mkdtemp(prefix="fc_all_output_")
if "skipped_files" not in st.session_state:
    st.session_state.skipped_files = []

st.set_page_config(page_title="Face Cropper UI", layout="wide")
st.title("Face Cropper (128×128) – Drag & Drop Interface")

st.markdown("""
Upload one or more image files (JPG, JPEG, PNG, BMP) to detect faces and produce 128×128 crops.  
Set thresholds and brightness jitter, then click **Run Face Cropper**.  
Cropped face thumbnails will appear below.
""")

# Sidebar controls
with st.sidebar:
    st.header("Configuration")
    uploaded_files = st.file_uploader(
        "Upload Image Files:",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
    )
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
    iou_threshold = st.slider("IoU Threshold (NMS)", 0.0, 1.0, 0.7, 0.01)
    brightness_jitter = st.slider("Brightness Jitter (%)", 0, 50, 20, 1)
    run_button = st.button("Run Face Cropper")

if run_button:
    if not uploaded_files:
        st.warning("No files uploaded.")
    else:
        total = len(uploaded_files)
        progress_placeholder = st.empty()

        for index, file in enumerate(uploaded_files):
            progress_placeholder.markdown(
                f"🟢 **Processing image {index + 1} of {total}...**", unsafe_allow_html=True
            )

            input_dir = Path(tempfile.mkdtemp(prefix="fc_input_"))
            output_dir = Path(tempfile.mkdtemp(prefix="fc_output_"))
            input_path = input_dir / file.name
            save_uploaded_file(file, input_path)

            # Setup Linux shared libs
            lib_dir = Path("face_cropper_linux_package/runtimeLib")
            try:
                (lib_dir / "libonnxruntime.so.1").symlink_to("libonnxruntime.so.1.16.0")
            except FileExistsError:
                pass
            try:
                (lib_dir / "libonnxruntime.so").symlink_to("libonnxruntime.so.1")
            except FileExistsError:
                pass

            cmd = [
                "bash", "-c",
                f"export LD_LIBRARY_PATH={lib_dir}:$LD_LIBRARY_PATH && "
                f"{BINARY_PATH} "
                f"--input-dir '{input_dir}' "
                f"--output-dir '{output_dir}' "
                f"--model '{MODEL_PATH}' "
                f"--conf-threshold {conf_threshold} "
                f"--iou-threshold {iou_threshold} "
                f"--brightness-jitter {brightness_jitter} "
                f"--no-recursive"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                st.session_state.skipped_files.append((file.name, result.stderr.strip()))
                shutil.rmtree(input_dir)
                shutil.rmtree(output_dir)
                continue

            face_files = list(output_dir.glob("*.jpg")) + list(output_dir.glob("*.jpeg")) + list(output_dir.glob("*.png"))
            if face_files:
                for face_file in face_files:
                    target = Path(st.session_state.final_output_dir) / f"{file.name}_{face_file.name}"
                    shutil.copy(face_file, target)
                    st.session_state.cropped_faces.append(target)
            else:
                st.session_state.skipped_files.append((file.name, "No faces detected"))

            shutil.rmtree(input_dir)
            shutil.rmtree(output_dir)

        progress_placeholder.empty()

# Skipped files
if st.session_state.skipped_files:
    st.warning("Some files were skipped:")
    for fname, reason in st.session_state.skipped_files:
        st.text(f"- {fname}: {reason}")

# Output display
if not st.session_state.cropped_faces:
    st.info("No face images detected yet.")
else:
    st.success(f"{len(st.session_state.cropped_faces)} face(s) cropped.")

    # ZIP creation
    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, mode="w") as zf:
        for face_path in st.session_state.cropped_faces:
            zf.write(face_path, arcname=face_path.name)
    mem_zip.seek(0)

    st.download_button(
        label="Download All Faces as ZIP",
        data=mem_zip,
        file_name="face_crops.zip",
        mime="application/zip",
    )

    # Grid of all images
    st.markdown("<hr><h4>Cropped Faces:</h4>", unsafe_allow_html=True)
    cols = st.columns(4)
    for i, face in enumerate(st.session_state.cropped_faces):
        col = cols[i % 4]
        with col:
            st.image(str(face), width=128)
            st.markdown(
                f"<div style='overflow-wrap: anywhere; font-size: 0.8rem;'>{face.name}</div>",
                unsafe_allow_html=True
            )
            with open(face, "rb") as f:
                img_bytes = f.read()
            st.download_button(
                label="Download",
                data=img_bytes,
                file_name=face.name,
                mime="image/jpeg",
                key=face.name,
            )
