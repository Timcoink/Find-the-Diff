#!/usr/bin/env python3
import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file, render_template_string
import base64
import json
from io import BytesIO
import zipfile
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure maximum content length (100MB)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

def base64_to_cv2(base64_str):
    try:
        # Remove data URL prefix if present
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        # Decode base64
        img_data = base64.b64decode(base64_str)
        # Convert to numpy array
        nparr = np.frombuffer(img_data, np.uint8)
        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")
        return img
    except Exception as e:
        logger.error(f"Error in base64_to_cv2: {str(e)}")
        raise

def cv2_to_base64(img):
    try:
        _, buffer = cv2.imencode('.jpg', img)
        return 'data:image/jpeg;base64,' + base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        logger.error(f"Error in cv2_to_base64: {str(e)}")
        raise

def process_images(img1_base64, img2_base64, settings):
    try:
        # Convert base64 to OpenCV format
        img1 = base64_to_cv2(img1_base64)
        img2 = base64_to_cv2(img2_base64)

        # Resize if necessary
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Find differences
        diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(diff, settings['threshold'], 255, cv2.THRESH_BINARY)
        kernel = np.ones((5,5), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=settings['dilationIter'])

        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= settings['minArea']]

        # Create result image
        result_img = img2.copy()
        circles_created = 0

        # Convert colors
        circle_color = tuple(int(settings['circleColor'].lstrip('#')[i:i+2], 16) for i in (4, 2, 0))
        overlay_color = tuple(int(settings['overlayColor'].lstrip('#')[i:i+2], 16) for i in (4, 2, 0))

        # Apply overlay
        if settings['overlayOpacity'] > 0:
            overlay = result_img.copy()
            cv2.fillPoly(overlay, filtered_contours, overlay_color)
            alpha = settings['overlayOpacity'] / 255.0
            cv2.addWeighted(overlay, alpha, result_img, 1 - alpha, 0, result_img)

        # Draw circles and numbers
        for i, cnt in enumerate(filtered_contours, 1):
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            
            # Draw circle
            cv2.circle(result_img, center, radius, circle_color, settings['circleThickness'])
            circles_created += 1

            # Draw number with background
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = str(i)
            (text_width, text_height), _ = cv2.getTextSize(text, font, 0.8, 2)
            
            # White background circle
            bg_radius = max(text_width, text_height) // 2 + 5
            cv2.circle(result_img, center, bg_radius, (255, 255, 255), -1)
            circles_created += 1
            
            # Draw text
            text_x = int(x - text_width/2)
            text_y = int(y + text_height/2)
            cv2.putText(result_img, text, (text_x, text_y), font, 0.8, (0, 0, 0), 2)

        # Create combined image
        spacing = settings['imageSpacing']
        sep_thickness = settings['separatorThickness']
        total_spacing = spacing * 2 + sep_thickness
        
        combined_height = max(img1.shape[0], img2.shape[0])
        combined_width = img1.shape[1] + total_spacing + img2.shape[1]
        
        combined_img = np.full((combined_height, combined_width, 3), 255, dtype=np.uint8)
        
        # Place first image
        combined_img[0:img1.shape[0], 0:img1.shape[1]] = img1
        
        # Draw separator
        separator_color = tuple(int(settings['separatorColor'].lstrip('#')[i:i+2], 16) for i in (4, 2, 0))
        start_sep = img1.shape[1] + spacing
        combined_img[:, start_sep:start_sep+sep_thickness] = separator_color
        
        # Place second image
        start_img2 = start_sep + sep_thickness + spacing
        combined_img[0:img2.shape[0], start_img2:] = result_img

        return {
            'combined_image': cv2_to_base64(combined_img),
            'answer_image': cv2_to_base64(result_img),
            'diff_count': len(filtered_contours),
            'circles_created': circles_created
        }
    except Exception as e:
        logger.error(f"Error in process_images: {str(e)}")
        raise

@app.route('/')
def index():
    return render_template_string(open('templates/index.html').read())

@app.route('/process', methods=['POST'])
def process():
    try:
        original = request.form['original']
        modified = request.form['modified']
        settings = json.loads(request.form['settings'])
        
        result = process_images(original, modified, settings)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in /process route: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/download-zip', methods=['POST'])
def download_zip():
    try:
        data = request.json
        memory_file = BytesIO()
        
        with zipfile.ZipFile(memory_file, 'w') as zf:
            # Add images
            for image_name, image_data in [
                ('combined.jpg', data['combined_image']),
                ('answer.jpg', data['answer_image'])
            ]:
                image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                zf.writestr(image_name, image_bytes)
        
        memory_file.seek(0)
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name='results.zip'
        )
    except Exception as e:
        logger.error(f"Error in /download-zip route: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=85, debug=True)