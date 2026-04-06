import streamlit as st
import cv2
import time
import numpy as np
from ultralytics import YOLO

st.set_page_config(page_title="Traffic Monitoring System", layout="wide")

st.title("🚦 AI Traffic Monitoring Dashboard")

# Load YOLO model
model = YOLO("yolov8n.pt")

uploaded_file = st.file_uploader("Upload Traffic Video", type=["mp4","avi","mov"])

if uploaded_file is not None:

    tfile = open("temp_video.mp4","wb")
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture("temp_video.mp4")

    frame_placeholder = st.empty()
    density_placeholder = st.empty()
    congestion_placeholder = st.empty()

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        vehicle_count = 0

        for r in results:
            for box in r.boxes:

                cls = int(box.cls[0])

                # Vehicle classes (COCO)
                if cls in [2,3,5,7]:
                    vehicle_count += 1

                x1,y1,x2,y2 = map(int,box.xyxy[0])

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

        # Congestion Logic
        if vehicle_count < 5:
            congestion = "FREE"
        elif vehicle_count < 10:
            congestion = "LIGHT"
        elif vehicle_count < 20:
            congestion = "MODERATE"
        else:
            congestion = "HEAVY"

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_placeholder.image(frame,use_container_width=True)

        density_placeholder.metric("Vehicle Count", vehicle_count)
        congestion_placeholder.metric("Congestion Level", congestion)

        time.sleep(0.03)

    cap.release()
