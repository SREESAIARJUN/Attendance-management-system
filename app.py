import streamlit as st
import cv2
import numpy as np
import pandas as pd
import datetime
from deepface import DeepFace
import os
from PIL import Image

# File to store attendance
attendance_file = "attendance.csv"

# Load or create attendance DataFrame
try:
    df = pd.read_csv(attendance_file)
except FileNotFoundError:
    df = pd.DataFrame(columns=["Name", "Faculty_ID", "In_Time", "Out_Time", "Date"])
    df.to_csv(attendance_file, index=False)

# Path to known faculty images
dataset_folder = "faculty_images"

# Streamlit UI
st.title("🎥 Face Attendance System")

# Upload an image from user
uploaded_file = st.file_uploader("Upload your image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Save uploaded file temporarily
    temp_path = "captured_image.jpg"
    image = Image.open(uploaded_file)
    image.save(temp_path)

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Loop through faculty images to compare
    recognized = False
    for filename in os.listdir(dataset_folder):
        faculty_id = os.path.splitext(filename)[0]  # Extract ID from filename
        image_path = os.path.join(dataset_folder, filename)

        try:
            result = DeepFace.verify(img1_path=image_path, img2_path=temp_path, model_name="VGG-Face")
            if result["verified"]:
                now = datetime.datetime.now()
                date_today = now.strftime("%Y-%m-%d")
                time_now = now.strftime("%H:%M:%S")

                if faculty_id not in df["Faculty_ID"].values:
                    df = df.append({"Name": faculty_id, "Faculty_ID": faculty_id, "In_Time": time_now, "Out_Time": "", "Date": date_today}, ignore_index=True)
                else:
                    df.loc[(df["Faculty_ID"] == faculty_id) & (df["Date"] == date_today), "Out_Time"] = time_now

                df.to_csv(attendance_file, index=False)
                st.success(f"✅ Attendance marked for {faculty_id} at {time_now}")
                recognized = True
                break

        except Exception as e:
            st.error(f"Error processing image: {e}")

    if not recognized:
        st.warning("❌ Face Not Recognized")
