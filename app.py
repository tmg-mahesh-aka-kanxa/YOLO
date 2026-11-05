import os
import cv2
import numpy as np
import base64
from flask import Flask, render_template, request, redirect

# Make sure you have ultralytics installed
from ultralytics import YOLO 

app = Flask(__name__)

# Load both of your YOLO models at startup
face_model = YOLO('best.pt')           # Your face detection weights
object_model = YOLO('object_detection_model.pt')       # Replace with your object detection weights file

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)
        
        # Check which button the user clicked
        action = request.form.get('action')
        
        if file:
            # 1. Read image directly into memory
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # 2. Select the model based on the form action button clicked
            if action == 'detect_faces':
                results = face_model(img)
                title_prefix = "Face Detection"
            elif action == 'detect_objects':
                results = object_model(img)
                title_prefix = "Object Detection"
            else:
                return redirect(request.url)
            
            # 3. Extract the annotated frame array from results
            # For modern ultralytics, results can be an array if batch=1, so we handle results[0]
            if isinstance(results, list):
                annotated_frame = results[0].plot()
            else:
                annotated_frame = results.plot()
            
            # 4. Encode the processed numpy frame back into a JPEG buffer
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            
            # 5. Convert the JPEG bytes buffer to a base64 string format
            b64_encoded = base64.b64encode(buffer).decode('utf-8')
            b64_src = f"data:image/jpeg;base64,{b64_encoded}"
            
            # Send the base64 string source and what model ran back to the frontend
            return render_template('index.html', result_image=b64_src, model_used=title_prefix)

    return render_template('index.html', result_image=None, model_used=None)

if __name__ == '__main__':
    app.run(debug=True)
