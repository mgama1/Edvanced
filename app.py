import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import tempfile
import os
import time
from bubble_sheet_processor import *

# Initialize the transformer model for OCR
# This assumes you've set up the model somewhere else or include the required imports
# from transformers import AutoProcessor, AutoModelForSeq2SeqLM
# model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/trocr-base-printed")
# processor = AutoProcessor.from_pretrained("microsoft/trocr-base-printed")

st.set_page_config(page_title="Bubble Sheet Scanner", layout="wide")

st.title("Bubble Sheet Scanner")
st.write("Upload a bubble sheet image to process and extract answers and student ID.")

uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_path = tmp_file.name
    
    try:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Process the image
        with st.spinner("Processing image..."):
            processor = BubbleSheetProcessor(temp_path)
            try:
                student_id, answer_matrix = processor.run_pipeline()
                
                # Display results
                st.success("Processing complete!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Student ID")
                    st.write(student_id)
                
                with col2:
                    st.subheader("Answer Matrix")
                    st.write(answer_matrix)
                
                # Display processed images
                st.subheader("Processing Steps")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(cv2.cvtColor(processor.warped, cv2.COLOR_BGR2RGB), 
                            caption="Warped Image", use_column_width=True)
                with col2:
                    st.image(cv2.cvtColor(processor.cropped, cv2.COLOR_BGR2RGB), 
                            caption="Cropped Bubbles Area", use_column_width=True)
                
                # Display threshold image
                st.image(processor.thresh, caption="Threshold Image", use_column_width=True)
                
                # Show bubble detection visualization
                if processor.bubble_contours is not None:
                    output = processor.cropped.copy()
                    cv2.drawContours(output, processor.bubble_contours, -1, (0, 255, 0), 2)
                    st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), 
                            caption="Detected Bubbles", use_column_width=True)
                
            except Exception as e:
                st.error(f"Error during processing: {str(e)}")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
    
    finally:
        # Clean up the temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)

# Add explanation of the application
with st.expander("About this app"):
    st.write("""
    This application processes bubble sheet scans to extract student IDs and answer selections.
    
    ### How it works:
    1. Upload a scanned bubble sheet image
    2. The app detects ArUco markers to align and warp the image
    3. It identifies bubble areas and analyzes which bubbles are filled
    4. It extracts the student ID using OCR
    5. Results are displayed showing the student ID and answer matrix
    
    The app uses computer vision techniques through OpenCV and text recognition with TrOCR.
    """)