import streamlit as st
import cv2
import numpy as np
import os
import sys

# Attempt to import the OCR pipeline
try:
    # Adding current directory to sys.path to ensure src is visible
    sys.path.append(os.getcwd())
    from src.ocr_pipeline import extract_text
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)

# Page Configuration
st.set_page_config(
    page_title="Sindhi OCR System",
    page_icon="🪶",
    layout="centered"
)

def main():
    st.title("🪶 Sindhi OCR System")
    st.markdown("---")

    if not IMPORT_SUCCESS:
        st.error(f"Could not import OCR components: {IMPORT_ERROR}")
        st.info("Check if 'src/ocr_pipeline.py' exists and all dependencies (opencv, pytesseract) are installed.")
        st.stop() # Stop execution if import failed

    st.write("Upload an image containing Sindhi script to extract the text.")

    # File Uploader
    uploaded_file = st.file_uploader("Upload Image (JPG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        temp_dir = "data/test_images"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, "temp_upload.jpg")
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Layout with two columns
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Uploaded Image")
            st.image(uploaded_file, use_container_width=True)

        # Button to run OCR
        if st.button("Extract Sindhi Text"):
            with st.spinner("Processing image and running OCR..."):
                try:
                    # Run the OCR pipeline
                    text, processed, original = extract_text(temp_path, lang='snd')
                    
                    with col2:
                        st.subheader("Extracted Text")
                        if text.strip():
                            st.text_area("Copy Text:", value=text.strip(), height=250)
                            st.success("Extraction Successful!")
                        else:
                            st.warning("No text detected. Ensure the image is clear and the Sindhi language pack ('snd') is installed.")
                    
                    # Show preprocessed version
                    with st.expander("View Preprocessing Steps (Binarization)"):
                        st.image(processed, caption="The cleaned-up version sent to Tesseract", use_container_width=True)
                        
                except Exception as e:
                    st.error(f"An error occurred during OCR: {e}")
                finally:
                    # Clean up the temporary file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

    # Footer
    st.markdown("---")
    st.caption("Powered by Tesseract OCR & OpenCV | Built for Sindhi Handwritten Recognition")

if __name__ == "__main__":
    main()
