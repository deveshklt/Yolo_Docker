# streamlit_app.py
import streamlit as st
import requests
import base64
import os

FASTAPI_URL = os.getenv("FASTAPI_URL")

st.title("🚀 Object Detection App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", width="stretch")

    if st.button("Run Detection"):
        # Use a tuple: (filename, file content, MIME type)
        files = {
            "upload_file": (
                uploaded_file.name,          # original filename
                uploaded_file.getvalue(),    # file bytes
                uploaded_file.type           # MIME type
            )
        }

        response = requests.post(FASTAPI_URL,files=files)

        if response.status_code == 200:
            data = response.json()
            st.write("API response:", data)  # Debugging

            if "image_base64" in data:
                output_b64 = data["image_base64"]
                st.image(base64.b64decode(output_b64), caption="Detected Image", width="stretch")
                st.json(data["detections"])
            else:
                st.error("No 'image_base64' found in API response")
        else:
            st.error(f"Error: {response.text}")
