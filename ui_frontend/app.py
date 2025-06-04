import io
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import streamlit as st

# Paths to your binary and model
BINARY_PATH = "./face_cropper"       # "./face_cropper.exe" on Windows
MODEL_PATH = "./model.onnx"

def save_uploaded_file(uploaded_file, input_path: Path):
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

st.set_page_config(page_title="Face Cropper UI", layout="wide")

st.title("Face Cropper (128×128) – Drag & Drop Interface")
st.markdown(
    """
    Upload one or more image files (JPG, JPEG, PNG, BMP) to detect faces and produce 128×128 crops.  
    Set thresholds and brightness jitter, then click **Run Face Cropper**.  
    Cropped face thumbnails will appear below.
    """
)

# Sidebar Controls
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

# When user clicks "Run Face Cropper"
if run_button:
    if not uploaded_files:
        st.warning("No files uploaded.")
    else:
        all_faces = []
        skipped_files = []
        final_output_dir = Path(tempfile.mkdtemp(prefix="fc_all_output_"))

        # Scrollable container for log
        with st.container():
            st.markdown(
                """
                <div style="max-height: 180px; overflow-y: auto; padding: 5px; border: 1px solid #ccc; border-radius: 5px;">
                """,
                unsafe_allow_html=True
            )
            for file in uploaded_files:
                st.markdown(f"<code>Processing: {file.name}</code><br>", unsafe_allow_html=True)

                input_dir = Path(tempfile.mkdtemp(prefix="fc_input_"))
                output_dir = Path(tempfile.mkdtemp(prefix="fc_output_"))
                input_path = input_dir / file.name
                save_uploaded_file(file, input_path)

                cmd = [
                    BINARY_PATH,
                    "--input-dir", str(input_dir),
                    "--output-dir", str(output_dir),
                    "--model", MODEL_PATH,
                    "--conf-threshold", str(conf_threshold),
                    "--iou-threshold", str(iou_threshold),
                    "--brightness-jitter", str(brightness_jitter),
                    "--no-recursive"
                ]

                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    skipped_files.append((file.name, result.stderr.strip()))
                    shutil.rmtree(input_dir)
                    shutil.rmtree(output_dir)
                    continue

                face_files = list(output_dir.glob("*.jpg")) + list(output_dir.glob("*.jpeg")) + list(output_dir.glob("*.png"))

                if face_files:
                    for face_file in face_files:
                        target = final_output_dir / f"{file.name}_{face_file.name}"
                        shutil.copy(face_file, target)
                        all_faces.append(target)
                else:
                    skipped_files.append((file.name, "No faces detected"))

                shutil.rmtree(input_dir)
                shutil.rmtree(output_dir)

            st.markdown("</div>", unsafe_allow_html=True)

        if skipped_files:
            st.warning("Some files were skipped:")
            for fname, reason in skipped_files:
                st.text(f"- {fname}: {reason}")

        if not all_faces:
            st.error("No faces were detected or saved from any image.")
        else:
            st.success(f"{len(all_faces)} face(s) cropped.")

            @st.experimental_fragment
            def download_faces():
                mem_zip = io.BytesIO()
                with zipfile.ZipFile(mem_zip, mode="w") as zf:
                    for face_path in all_faces:
                        zf.write(face_path, arcname=face_path.name)
                mem_zip.seek(0)

                st.download_button(
                    label="Download All Faces as ZIP",
                    data=mem_zip,
                    file_name="face_crops.zip",
                    mime="application/zip",
                )

                cols = st.columns(4)
                for i, face in enumerate(all_faces):
                    col = cols[i % 4]
                    with col:
                        st.image(str(face), caption=face.name, width=128)
                        with open(face, "rb") as f:
                            img_bytes = f.read()
                        st.download_button(
                            label=f"Download {face.name}",
                            data=img_bytes,
                            file_name=face.name,
                            mime="image/jpeg",
                            key=face.name
                        )

            download_faces()
