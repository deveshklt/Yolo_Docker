from fastapi import FastAPI, UploadFile, File
import torch
import numpy as np
import cv2
import base64
from fastapi.responses import JSONResponse
from torchvision.ops import nms
import os
import json

app = FastAPI()

# Folder to save annotated images and JSON
SAVE_DIR = "annotated_images"
os.makedirs(SAVE_DIR, exist_ok=True)

# Load YOLOv5 model
model_weights = torch.hub.load(
    "ultralytics/yolov5", "custom",
    path="yolov5/runs/train/exp13/weights/best.pt"
)

@app.post("/detect")
async def detect(upload_file: UploadFile = File(...)):
    try:
        # Read uploaded image
        image_bytes = await upload_file.read()
        np_array = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        # Inference
        results = model_weights(img, size=640)

        # Confidence threshold
        conf_thres = 0.5
        results.xyxy[0] = results.xyxy[0][results.xyxy[0][:, 4] >= conf_thres]

        # Optional manual NMS
        iou_thres = 0.8
        if results.xyxy[0].shape[0] > 1:
            boxes = results.xyxy[0][:, :4]
            scores = results.xyxy[0][:, 4]
            keep = nms(boxes, scores, iou_thres)
            results.xyxy[0] = results.xyxy[0][keep]

        # JSON detections
        detections = results.pandas().xyxy[0].to_dict(orient="records")

        # Annotated image
        annotated_img = results.render()[0]

        # Prepare filenames with _detected
        filename, ext = os.path.splitext(upload_file.filename)
        annotated_image_name = f"{filename}_detected{ext}"
        json_name = f"{filename}_detected.json"

        # Save annotated image
        save_image_path = os.path.join(SAVE_DIR, annotated_image_name)
        cv2.imwrite(save_image_path, annotated_img)

        # Save JSON
        save_json_path = os.path.join(SAVE_DIR, json_name)
        with open(save_json_path, "w") as f:
            json.dump(detections, f, indent=4)

        # Encode annotated image for API response
        _, buffer = cv2.imencode(".jpg", annotated_img)
        img_base64 = base64.b64encode(buffer).decode("utf-8")

        return JSONResponse({
            "image_base64": img_base64,
            "detections": detections,
            "saved_image_path": save_image_path,
            "saved_json_path": save_json_path
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
