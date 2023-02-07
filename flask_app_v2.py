from flask import Flask, jsonify, request
from app_v2 import detect
import cv2
import numpy as np
import json
import base64
import binascii
import shutil
import os

app = Flask(__name__)
frame_count = 0
directory_path = 'frame'
shutil.rmtree(directory_path)
os.mkdir(directory_path)

@app.route("/", methods=["GET", "POST"])
def index():
    global frame_count
    file = request.files.get('file')
    file_name = file.filename
    file_content = file.read()
    file_type = file.content_type
    
    frame_name = 'frame/test'+str(frame_count)+'.jpg'
    with open(frame_name, 'wb') as f:
        f.write(file_content)
    frame_count += 1
    
    result = detect(frame_name)
    
    response = {'name': str(result)}
    
    return jsonify(response)

    
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")