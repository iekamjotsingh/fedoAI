# Fedo-like AI Health Risk Estimation Web Demo (Extended)

import streamlit as st
import cv2
import numpy as np
from scipy.signal import butter, filtfilt
import tempfile
import os
import time
from datetime import datetime

# === Signal Processing Helpers ===
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut=0.7, highcut=4.0, fs=30.0, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data, axis=0)
    return y

def extract_green_signal_from_video(video_path, fps=30):
    cap = cv2.VideoCapture(video_path)
    green_signals = []
    timestamps = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (480, 360))
        roi = frame[100:300, 150:350]  # approximate face region
        green_avg = np.mean(roi[:, :, 1])  # green channel average
        green_signals.append(green_avg)
        timestamps.append(datetime.now().timestamp())
    cap.release()
    return np.array(green_signals), np.array(timestamps)

def estimate_heart_rate(signal, fps=30):
    filtered = bandpass_filter(signal, fs=fps)
    fft = np.abs(np.fft.rfft(filtered))
    freqs = np.fft.rfftfreq(len(filtered), d=1/fps)
    peak_freq = freqs[np.argmax(fft)]
    bpm = peak_freq * 60.0
    return bpm

def estimate_stress(signal, timestamps):
    diff = np.diff(signal)
    stress_index = np.std(diff) * 10  # arbitrary stress proxy
    return min(10, max(0, stress_index))

def estimate_blood_pressure(hr, age):
    systolic = 100 + 0.5 * age + 0.1 * hr
    diastolic = 60 + 0.3 * age + 0.05 * hr
    return round(systolic), round(diastolic)

def estimate_risk_score(bpm, age=35):
    risk = 0
    if bpm > 90: risk += 1
    if bpm < 50: risk += 1
    if age > 45: risk += 1
    if bpm > 100 and age > 50: risk += 2
    return min(10, risk * 2)

# === Streamlit App ===
st.title("Fedo-like AI Health Risk Estimator (Extended)")
st.markdown("Upload a short facial video (10–15 seconds) or use webcam to analyze:")

age = st.slider("Select Age", 18, 80, 35)
use_webcam = st.checkbox("Use Webcam Instead of Upload")

if use_webcam:
    st.warning("Ensure your webcam is enabled and well-lit. Recording starts automatically.")
    if st.button("Start Webcam Recording"):
        cap = cv2.VideoCapture(0)
        output_path = os.path.join(tempfile.gettempdir(), "webcam_capture.avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (640, 480))
        start_time = time.time()
        while int(time.time() - start_time) < 10:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            cv2.imshow('Recording...', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        st.success("Webcam recording complete. Processing...")
        video_path = output_path
    else:
        video_path = None
else:
    uploaded_file = st.file_uploader("Upload MP4 Video", type=["mp4"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(uploaded_file.read())
            video_path = temp_video.name
    else:
        video_path = None

if video_path:
    with st.spinner("Processing video and estimating vitals..."):
        signal, timestamps = extract_green_signal_from_video(video_path)
        if len(signal) < 30:
            st.error("Video too short or invalid for signal extraction.")
        else:
            hr = estimate_heart_rate(signal)
            stress = estimate_stress(signal, timestamps)
            systolic, diastolic = estimate_blood_pressure(hr, age)
            risk = estimate_risk_score(hr, age=age)
            st.success(f"Estimated Heart Rate: {hr:.2f} bpm")
            st.info(f"Estimated Stress Index (0-10): {stress:.2f}")
            st.info(f"Estimated Blood Pressure: {systolic}/{diastolic} mmHg")
            st.warning(f"Risk Score (0-10): {risk}")
            st.line_chart(signal)

    if not use_webcam:
        os.remove(video_path)
