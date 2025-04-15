import cv2

def extract_key_frames(video_path, interval_sec=0.5, target_width=300):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_sec)
    
    count = 0
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            h, w = frame.shape[:2]
            aspect_ratio = h / w
            new_height = int(target_width * aspect_ratio)
            resized_frame = cv2.resize(frame, (target_width, new_height))
            frames.append(resized_frame)
        count += 1
        
    cap.release()
    return frames

import matplotlib.pyplot as plt

import base64
import requests
import json
from io import BytesIO
import numpy as np

def encode_frame_to_base64(frame):
    # Convert frame to JPEG format in memory
    _, buffer = cv2.imencode('.jpg', frame)
    # Convert to base64
    return base64.b64encode(buffer).decode('utf-8')


def send_frames_to_chatgpt(frames, api_key, skill):
    prompt = "you are an excellent ski coach. the skier is trying to practice" + skill + ". Please analyze these frames taken in series from a video and tell what the skier is doing. "

    # Convert frames to base64
    base64_frames = [encode_frame_to_base64(frame) for frame in frames]
    
    # Prepare the API request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # Prepare the messages with images
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                }
            ] + [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_frame}"
                    }
                }
                for base64_frame in base64_frames
            ]
        }
    ]
    
    payload = {
        "model": "gpt-4o",
        "messages": messages,
        "max_tokens": 10000
    }
    
    # Send request to ChatGPT API
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload
    )
    
    return response.json()

from openai import OpenAI

def websearch(prompt,_api_key,skill):
    client = OpenAI(api_key=_api_key)

    response = client.responses.create(
        model="gpt-4o",
        tools=[{
            "type": "web_search_preview",
        }],
        input = "you are an excellent ski coach. the skier is trying to practice" + skill + ". here is a description for a skier's action:" 
        + prompt 
        + "please search the internet for the best tutorial for the skier's actions and summarize it in detail. provide one best drill for the skier to practice and show illustration pictures for the drill."
    )

    return response.output_text

import openai
import time

from PIL import Image
def image_to_base64(img: Image.Image) -> str:
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    base64_str = base64.b64encode(img_bytes).decode()
    return base64_str


#main page

import streamlit as st
import os

api_key = st.secrets["api_key"]
ACCESS_CODE = st.secrets["access_code"]  # or hardcode as needed

logo = Image.open("logo.png")
logo_base64 = image_to_base64(logo)

st.markdown(
    f"""
    <div style="text-align: center;">
        <img src="data:image/png;base64,{logo_base64}" width="120">
        <h2>AI Snow Coach-Monti</h2>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("上传你的视频，AI 帮你分析动作并推荐训练方法！")

# Ask user for access code
user_code = st.text_input("请输入访问口令（Please enter the access code）", type="password")

if user_code == ACCESS_CODE:
    # Skill selection dropdown
    st.markdown("### 滑雪技术选择 (Skiing Technique Selection)")
    
    # Main skill selection
    main_skill = st.selectbox(
        "请选择主要滑雪技术 (Select main skiing technique)：",
        options=["Riding", "park"],
        index=0,
        help="选择你正在练习的主要滑雪技术类别"
    )

    # Define sub-skills per main skill
    sub_skill_options = {
        "Riding": ["snowplow", "parallel", "carving"],
        "park": ["jumps", "rail", "box 90"],
    }

    # Sub skill selection based on main skill
    sub_skill = st.selectbox(
        "请选择具体技术细节 (Select specific technique)：",
        options=sub_skill_options[main_skill],
        index=0,
        help="选择你想要改进的具体技术细节"
    )

    st.markdown("---")  # Add a separator line

    uploaded_video = st.file_uploader("请上传滑雪视频（一个动作，5秒以内）Please upload a skiing video (one action, 5 seconds or less)", type=["mp4", "mov"])

    if uploaded_video:
        temp_filename = "temp_video.mp4"

        with open(temp_filename, "wb") as f:
            f.write(uploaded_video.getvalue())

        # Check video duration
        cap = cv2.VideoCapture(temp_filename)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps
        cap.release()

        if duration > 5:
            st.error(f"上传的视频时长为 {duration:.2f} 秒，超过了 5 秒，请上传更短的视频，录制一个弯就可以。")
            os.remove(temp_filename)
        else:
            st.video(uploaded_video)
            try:
                frames = extract_key_frames(temp_filename)
                responseT = send_frames_to_chatgpt(frames, api_key,sub_skill)
                st.write(responseT['choices'][0]['message']['content']) 
                response = websearch(responseT['choices'][0]['message']['content'],api_key,sub_skill)
                st.write(response)
            finally:
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
else:
    if user_code != "":
        st.error("访问口令错误，请重新输入。")
    else:
        st.info("请输入访问口令以使用本功能。")
