import os
import json
import streamlit as st
from metadata import extract_metadata
from hash_calculator import calculate_hash
from frame_analyzer import analyze_frames

# Streamlit app
st.title('Video Forensic Analysis Tool')

# File uploader
uploaded_file = st.file_uploader('Upload a video file', type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    # Check if 'temp' folder exists, create if not
    temp_folder = 'temp'
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    # Save the uploaded file temporarily
    temp_file_path = os.path.join(temp_folder, uploaded_file.name)
    with open(temp_file_path, 'wb') as f:
        f.write(uploaded_file.read())

    st.write("**File uploaded successfully**")

    # Extract metadata
    st.write("**Extracting metadata...**")
    metadata = extract_metadata(temp_file_path)
    st.json(metadata)

    # Calculate hash
    st.write("**Calculating file hash (MD5)...**")
    video_hash = calculate_hash(temp_file_path)
    st.write(f"MD5 Hash: {video_hash}")

    # Analyze frames for alterations
    st.write("**Analyzing frames for alterations...**")
    altered_frames = analyze_frames(temp_file_path)
    st.write(f"Altered frames: {altered_frames}")

    # Generate report name based on video file name
    file_name = os.path.splitext(uploaded_file.name)[0]
    report_name = f'report-{file_name}.json'

    # Report creation
    report = {
        'metadata': metadata,
        'hash': video_hash,
        'altered_frames': altered_frames
    }

    st.write(f"**Video forensic analysis complete! Report saved as: {report_name}**")
    st.json(report)

    # Offer download of the JSON report
    report_json = json.dumps(report, indent=4)
    st.download_button(
        label="Download report as JSON",
        data=report_json,
        file_name=report_name,
        mime="application/json"
    )
