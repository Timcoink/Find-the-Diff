#!/usr/bin/env python3
import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file, render_template_string
import base64
import json
import zipfile
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Tuple, Dict
import math

app = Flask(__name__)

@dataclass
class DifferenceRegion:
    contour: np.ndarray
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    area: float
    intensity: float
    shape_complexity: float
    edge_density: float
    texture_pattern: np.ndarray
    neighbors: List[int] = None
    grouped: bool = False

class SmartDifferenceDetector:
    def __init__(self):
        self.regions = []
        self.groups = []
        
    def analyze_texture(self, roi: np.ndarray) -> Tuple[float, np.ndarray]:
        """Analyze texture patterns in the region"""
        if roi.size == 0:
            return 0.0, np.array([])
            
        # Calculate gradient
        gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(gx, gy)
        
        edge_density = np.mean(magnitude)
        
        # Get texture pattern using LBP-like approach
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        pattern = np.zeros_like(gray)
        
        for i in range(1, gray.shape[0]-1):
            for j in range(1, gray.shape[1]-1):
                center = gray[i,j]
                code = 0
                code |= (gray[i-1,j] > center) << 0
                code |= (gray[i+1,j] > center) << 1
                code |= (gray[i,j-1] > center) << 2
                code |= (gray[i,j+1] > center) << 3
                pattern[i,j] = code
                
        return edge_density, pattern

    def analyze_region(self, img1: np.ndarray, img2: np.ndarray, contour: np.ndarray) -> DifferenceRegion:
        x, y, w, h = cv2.boundingRect(contour)
        center = (x + w//2, y + h//2)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        shape_complexity = perimeter * perimeter / (4 * np.pi * area) if area > 0 else float('inf')
        
        # Get ROIs
        roi1 = img1[y:y+h, x:x+w]
        roi2 = img2[y:y+h, x:x+w]
        
        # Calculate difference intensity
        diff_intensity = np.mean(cv2.absdiff(roi1, roi2))
        
        # Analyze texture
        edge_density, texture_pattern = self.analyze_texture(roi1)
        
        return DifferenceRegion(
            contour=contour,
            bbox=(x, y, w, h),
            center=center,
            area=area,
            intensity=diff_intensity,
            shape_complexity=shape_complexity,
            edge_density=edge_density,
            texture_pattern=texture_pattern
        )

    def compare_patterns(self, pattern1: np.ndarray, pattern2: np.ndarray) -> float:
        """Compare texture patterns using histogram comparison"""
        if pattern1.size == 0 or pattern2.size == 0:
            return 0.0
            
        hist1 = cv2.calcHist([pattern1], [0], None, [16], [0,16])
        hist2 = cv2.calcHist([pattern2], [0], None, [16], [0,16])
        
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    def should_group(self, region1: DifferenceRegion, region2: DifferenceRegion, 
                    touch_distance: float) -> bool:
        # Calculate spatial relationship
        dx = region2.center[0] - region1.center[0]
        dy = region2.center[1] - region1.center[1]
        distance = math.sqrt(dx*dx + dy*dy)
        angle = math.degrees(math.atan2(dy, dx)) % 360
        
        # Calculate relative metrics
        avg_area = (region1.area + region2.area) / 2
        relative_distance = distance / math.sqrt(avg_area)
        size_ratio = max(region1.area, region2.area) / min(region1.area, region2.area)
        
        # Pattern similarity
        pattern_similarity = self.compare_patterns(region1.texture_pattern, region2.texture_pattern)
        
        # Decision criteria
        is_close = distance <= touch_distance
        is_similar_size = size_ratio < 3.0
        has_similar_intensity = abs(region1.intensity - region2.intensity) < 30
        has_similar_complexity = abs(region1.shape_complexity - region2.shape_complexity) < 1.0
        has_similar_edges = abs(region1.edge_density - region2.edge_density) < 0.3
        has_similar_pattern = pattern_similarity > 0.7
        
        # Angle-based alignment check
        is_aligned = (angle < 45 or abs(angle - 90) < 45 or 
                     abs(angle - 180) < 45 or abs(angle - 270) < 45)
        
        # Combined decision
        if is_close and is_similar_size and has_similar_intensity:
            if has_similar_complexity and has_similar_edges:
                return True
                
        if relative_distance < 2.0 and is_aligned:
            if has_similar_pattern and has_similar_complexity:
                return True
                
        return False

    def group_differences(self, img1: np.ndarray, img2: np.ndarray, 
                        contours: List[np.ndarray], touch_distance: float) -> List[List[np.ndarray]]:
        # Analyze all regions
        self.regions = [self.analyze_region(img1, img2, cnt) for cnt in contours]
        self.groups = []
        
        # Sort regions by size (largest first)
        self.regions.sort(key=lambda r: r.area, reverse=True)
        
        # Group regions
        for i, region in enumerate(self.regions):
            if region.grouped:
                continue
                
            current_group = [region]
            region.grouped = True
            
            group_changed = True
            iteration_count = 0
            max_iterations = 100  # Prevent infinite loops
            
            while group_changed and iteration_count < max_iterations:
                iteration_count += 1
                group_changed = False
                
                for other_region in self.regions:
                    if other_region.grouped:
                        continue
                        
                    # Check against all regions in current group
                    should_add = any(
                        self.should_group(grouped_region, other_region, touch_distance)
                        for grouped_region in current_group
                    )
                    
                    if should_add:
                        current_group.append(other_region)
                        other_region.grouped = True
                        group_changed = True
                        break
            
            self.groups.append([r.contour for r in current_group])
        
        return self.groups

def process_images(img1, img2, settings):
    """Process two images and return the results"""
    
    # Convert base64 images to numpy arrays
    def base64_to_numpy(base64_str):
        base64_str = base64_str.split(',')[1]
        img_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img1 = base64_to_numpy(img1)
    img2 = base64_to_numpy(img2)

    # Resize if necessary
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Compute absolute difference with enhancement
    diff = cv2.absdiff(gray1, gray2)
    blurred_diff = cv2.GaussianBlur(diff, (3, 3), 0)
    _, thresh = cv2.threshold(blurred_diff, settings['threshold'], 255, cv2.THRESH_BINARY)

    # Dilate
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=settings['dilationIter'])

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by minimum area
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= settings['minArea']]

    # Use smart difference detector
    detector = SmartDifferenceDetector()
    groups = detector.group_differences(img1, img2, filtered_contours, settings['touchDistance'])

    # Create answer image
    answer_img = img2.copy()
    circles_created = 0

    # Convert hex colors to BGR
    def hex_to_bgr(hex_color):
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (b, g, r)

    circle_color = hex_to_bgr(settings['circleColor'])
    overlay_color = hex_to_bgr(settings['overlayColor'])
    separator_color = hex_to_bgr(settings['separatorColor'])

    # Apply overlay
    if settings['overlayOpacity'] > 0:
        mask = dilated
        alpha = settings['overlayOpacity'] / 255
        overlay = np.zeros_like(answer_img)
        overlay[:] = overlay_color
        cv2.addWeighted(overlay, alpha, answer_img, 1 - alpha, 0, answer_img, mask=mask)

    # Draw circles and numbers
    for idx, group in enumerate(groups, 1):
        all_points = np.concatenate([cnt.reshape(-1, 2) for cnt in group])
        (x, y), radius = cv2.minEnclosingCircle(all_points)
        center = (int(x), int(y))
        radius = int(radius)

        # Draw circle
        cv2.circle(answer_img, center, radius, circle_color, settings['circleThickness'])
        circles_created += 1

        # Draw number
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        text = str(idx)
        
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Draw white background circle
        bg_radius = max(text_width, text_height) // 2 + 5
        cv2.circle(answer_img, center, bg_radius, (255, 255, 255), -1)
        circles_created += 1
        
        # Draw text
        text_x = int(x - text_width/2)
        text_y = int(y + text_height/2)
        cv2.putText(answer_img, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

    # Create combined image with separator line
    spacing = settings['imageSpacing']
    separator_thickness = settings['separatorThickness']
    total_spacing = spacing * 2 + separator_thickness

    combined_img = np.full((max(img1.shape[0], img2.shape[0]),
                           img1.shape[1] + total_spacing + img2.shape[1], 3),
                          255, dtype=np.uint8)
    
    # Draw first image
    combined_img[0:img1.shape[0], 0:img1.shape[1]] = img1
    
    # Draw separator line
    separator_start = img1.shape[1] + spacing
    separator_end = separator_start + separator_thickness
    combined_img[:, separator_start:separator_end] = separator_color
    
    # Draw second image
    combined_img[0:img2.shape[0], separator_end + spacing:] = img2

    # Convert images to base64
    def numpy_to_base64(img):
        _, buffer = cv2.imencode('.jpg', img)
        return 'data:image/jpeg;base64,' + base64.b64encode(buffer).decode('utf-8')

    return {
        'combined_image': numpy_to_base64(combined_img),
        'answer_image': numpy_to_base64(answer_img),
        'diff_count': len(groups),
        'circles_created': circles_created
    }

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Find the Difference Web Tool</title>
    <style>
        :root {
            --primary-color: #2196F3;
            --secondary-color: #1976D2;
            --background-color: #f5f5f5;
            --border-color: #ddd;
            --text-color: #333;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
            line-height: 1.6;
            background-color: var(--background-color);
            color: var(--text-color);
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }

        h1 {
            text-align: center;
            margin-bottom: 2rem;
            color: var(--primary-color);
        }

        .upload-section {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            margin-bottom: 2rem;
        }

        .image-upload {
            text-align: center;
        }

        .drop-zone {
            width: 100%;
            height: 200px;
            padding: 25px;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            font-size: 20px;
            font-weight: 500;
            cursor: pointer;
            color: #777
                        color: #777;
            border: 2px dashed var(--primary-color);
            border-radius: 10px;
            background-color: white;
            transition: all 0.3s ease;
        }

        .drop-zone:hover {
            background-color: rgba(33, 150, 243, 0.05);
        }

        .drop-zone.drop-zone--over {
            border-style: solid;
            background-color: rgba(33, 150, 243, 0.1);
        }

        .preview-image {
            max-width: 100%;
            max-height: 300px;
            margin-top: 1rem;
            border-radius: 5px;
            display: none;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }

        .settings-section {
            background-color: white;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
        }

        .settings-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
        }

        .setting-item {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
        }

        .setting-item label {
            font-weight: 500;
            color: var(--secondary-color);
        }

        .process-button, .regenerate-button {
            display: block;
            width: 100%;
            padding: 1rem;
            font-size: 1.2rem;
            font-weight: 600;
            color: white;
            background-color: var(--primary-color);
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: all 0.3s ease;
            margin: 20px 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }

        .process-button:hover, .regenerate-button:hover {
            background-color: var(--secondary-color);
            transform: translateY(-1px);
        }

        .process-button:disabled {
            background-color: #ccc;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        .results {
            display: none;
            margin-top: 2rem;
            padding: 2rem;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }

        .result-images {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            margin-bottom: 2rem;
        }

        .result-image {
            text-align: center;
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
        }

        .result-image img {
            max-width: 100%;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }

        .stats {
            text-align: center;
            margin: 1rem 0;
            font-size: 1.1rem;
            padding: 1rem;
            background: #f8f9fa;
            border-radius: 8px;
        }

        .download-button {
            display: inline-block;
            padding: 0.8rem 1.5rem;
            background-color: var(--primary-color);
            color: white;
            text-decoration: none;
            border-radius: 5px;
            margin-top: 10px;
            cursor: pointer;
            transition: all 0.3s ease;
            border: none;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }

        .download-button:hover {
            background-color: var(--secondary-color);
            transform: translateY(-1px);
        }

        .loading {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(255,255,255,0.9);
            justify-content: center;
            align-items: center;
            z-index: 1000;
        }

        .loading-spinner {
            width: 50px;
            height: 50px;
            border: 5px solid #f3f3f3;
            border-top: 5px solid var(--primary-color);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        @media (max-width: 768px) {
            .upload-section, .result-images {
                grid-template-columns: 1fr;
            }
            
            .container {
                padding: 1rem;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Find the Difference Web Tool</h1>
        
        <div class="upload-section">
            <div class="image-upload">
                <h3>Original Image</h3>
                <div class="drop-zone" id="originalDrop">
                    <span class="drop-zone__prompt">Drop file here or click to upload</span>
                    <input type="file" name="original" accept="image/*">
                </div>
                <img id="originalPreview" class="preview-image" src="" alt="Original preview">
            </div>

            <div class="image-upload">
                <h3>Modified Image</h3>
                <div class="drop-zone" id="modifiedDrop">
                    <span class="drop-zone__prompt">Drop file here or click to upload</span>
                    <input type="file" name="modified" accept="image/*">
                </div>
                <img id="modifiedPreview" class="preview-image" src="" alt="Modified preview">
            </div>
        </div>

        <div class="settings-section">
            <h3>Settings</h3>
            <div class="settings-grid">
                <div class="setting-item">
                    <label for="threshold">Threshold: <span id="thresholdValue">30</span></label>
                    <input type="range" id="threshold" min="1" max="255" value="30">
                </div>

                <div class="setting-item">
                    <label for="minArea">Minimum Area: <span id="minAreaValue">40</span></label>
                    <input type="range" id="minArea" min="1" max="200" value="40">
                </div>

                <div class="setting-item">
                    <label for="dilationIter">Dilation Iterations: <span id="dilationIterValue">2</span></label>
                    <input type="range" id="dilationIter" min="0" max="10" value="2">
                </div>

                <div class="setting-item">
                    <label for="circleThickness">Circle Thickness: <span id="circleThicknessValue">2</span></label>
                    <input type="range" id="circleThickness" min="1" max="10" value="2">
                </div>

                <div class="setting-item">
                    <label for="circleColor">Circle Color:</label>
                    <input type="color" id="circleColor" value="#FF0000">
                </div>

                <div class="setting-item">
                    <label for="overlayColor">Overlay Color:</label>
                    <input type="color" id="overlayColor" value="#FF0000">
                </div>

                <div class="setting-item">
                    <label for="overlayOpacity">Overlay Opacity: <span id="overlayOpacityValue">100</span></label>
                    <input type="range" id="overlayOpacity" min="0" max="255" value="100">
                </div>

                <div class="setting-item">
                    <label for="touchDistance">Touch Distance: <span id="touchDistanceValue">10</span></label>
                    <input type="range" id="touchDistance" min="1" max="50" value="10">
                </div>

                <div class="setting-item">
                    <label for="imageSpacing">Image Spacing: <span id="imageSpacingValue">10</span>px</label>
                    <input type="range" id="imageSpacing" min="0" max="100" value="10">
                </div>

                <div class="setting-item">
                    <label for="separatorColor">Separator Line Color:</label>
                    <input type="color" id="separatorColor" value="#87CEEB">
                </div>

                <div class="setting-item">
                    <label for="separatorThickness">Separator Thickness: <span id="separatorThicknessValue">2</span>px</label>
                    <input type="range" id="separatorThickness" min="1" max="10" value="2">
                </div>
            </div>
        </div>

        <button id="processButton" class="process-button" disabled>Process Images</button>

        <div id="results" class="results">
            <div class="result-images">
                <div class="result-image">
                    <h3>Combined Result</h3>
                    <img id="combinedResult" src="" alt="Combined result">
                    <br>
                    <button class="download-button" onclick="downloadImage('combinedResult', 'combined.jpg')">Download Combined</button>
                </div>
                <div class="result-image">
                    <h3>Answer Result</h3>
                    <img id="answerResult" src="" alt="Answer result">
                    <br>
                    <button class="download-button" onclick="downloadImage('answerResult', 'answer.jpg')">Download Answer</button>
                </div>
            </div>
            <div class="stats">
                <p>Differences found: <span id="diffCount">0</span></p>
                <p>Circles created: <span id="circlesCount">0</span></p>
            </div>
            <button class="regenerate-button" onclick="regenerateImages()">Regenerate with Current Settings</button>
            <button id="downloadZip" class="download-button" onclick="downloadZip()">Download All as ZIP</button>
        </div>
    </div>

    <div class="loading">
        <div class="loading-spinner"></div>
    </div>

    <script>
        // Update range input values
        function updateRangeValue(inputId, valueId, suffix = '') {
            const input = document.getElementById(inputId);
            const value = document.getElementById(valueId);
            value.textContent = input.value + suffix;
            input.addEventListener('input', () => value.textContent = input.value + suffix);
        }

        // Initialize range input values
        updateRangeValue('threshold', 'thresholdValue');
        updateRangeValue('minArea', 'minAreaValue');
        updateRangeValue('dilationIter', 'dilationIterValue');
        updateRangeValue('circleThickness', 'circleThicknessValue');
        updateRangeValue('overlayOpacity', 'overlayOpacityValue');
        updateRangeValue('touchDistance', 'touchDistanceValue');
        updateRangeValue('imageSpacing', 'imageSpacingValue', 'px');
        updateRangeValue('separatorThickness', 'separatorThicknessValue', 'px');

        // Show/hide loading spinner
        function toggleLoading(show) {
            document.querySelector('.loading').style.display = show ? 'flex' : 'none';
        }

        // Handle file drops
        document.querySelectorAll('.drop-zone').forEach(dropZone => {
            const input = dropZone.querySelector('input');
            const preview = document.getElementById(dropZone.id.replace('Drop', 'Preview'));

            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                dropZone.addEventListener(eventName, preventDefaults, false);
            });

            function preventDefaults(e) {
                e.preventDefault();
                e.stopPropagation();
            }

            ['dragenter', 'dragover'].forEach(eventName => {
                dropZone.addEventListener(eventName, highlight, false);
            });

            ['dragleave', 'drop'].forEach(eventName => {
                dropZone.addEventListener(eventName, unhighlight, false);
            });

            function highlight(e) {
                dropZone.classList.add('drop-zone--over');
            }

            function unhighlight(e) {
                dropZone.classList.remove('drop-zone--over');
            }

            dropZone.addEventListener('drop', handleDrop, false);
            input.addEventListener('change', handleChange, false);

            function handleDrop(e) {
                const dt = e.dataTransfer;
                const file = dt.files[0];
                handleFile(file);
            }

            function handleChange(e) {
                const file = e.target.files[0];
                handleFile(file);
            }

            function handleFile(file) {
                if (file && file.type.startsWith('image/')) {
                    const reader = new FileReader();
                    reader.readAsDataURL(file);
                    reader.onload = function(e) {
                        preview.src = e.target.result;
                        preview.style.display = 'block';
                        checkProcessButton();
                    }
                }
            }
        });

        // Check if both images are loaded
        function checkProcessButton() {
            const originalPreview = document.getElementById('originalPreview');
            const modifiedPreview = document.getElementById('modifiedPreview');
            const processButton = document.getElementById('processButton');
            
            processButton.disabled = !(originalPreview.src && modifiedPreview.src);
        }

        // Get current settings
        function getSettings() {
            return {
                threshold: parseInt(document.getElementById('threshold').value),
                minArea: parseInt(document.getElementById('minArea').value),
                dilationIter: parseInt(document.getElementById('dilationIter').value),
                circleThickness: parseInt(document.getElementById('circleThickness').value),
                circleColor: document.getElementById('circleColor').value,
                overlayColor: document.getElementById('overlayColor').value,
                overlayOpacity: parseInt(document.getElementById('overlayOpacity').value),
                touchDistance: parseInt(document.getElementById('touchDistance').value),
                imageSpacing: parseInt(document.getElementById('imageSpacing').value),
                separatorColor: document.getElementById('separatorColor').value,
                separatorThickness: parseInt(document.getElementById('separatorThickness').value)
            };
        }

        // Process images
        async function processImages() {
            toggleLoading(true);

            const settings = getSettings();
            const formData = new FormData();
            formData.append('original', document.getElementById('originalPreview').src);
            formData.append('modified', document.getElementById('modifiedPreview').src);
            formData.append('settings', JSON.stringify(settings));

            try {
                const response = await fetch('/process', {
                    method: 'POST',
                    body: formData
                });

                if (response.ok) {
                    const result = await response.json();
                    
                    document.getElementById('combinedResult').src = result.combined_image;
                                        document.getElementById('answerResult').src = result.answer_image;
                    document.getElementById('diffCount').textContent = result.diff_count;
                    document.getElementById('circlesCount').textContent = result.circles_created;
                    document.getElementById('results').style.display = 'block';
                } else {
                    alert('Error processing images');
                }
            } catch (error) {
                console.error('Error:', error);
                alert('Error processing images');
            } finally {
                toggleLoading(false);
            }
        }

        // Regenerate images with current settings
        function regenerateImages() {
            processImages();
        }

        document.getElementById('processButton').addEventListener('click', processImages);

        // Download functions
        function downloadImage(imageId, filename) {
            const image = document.getElementById(imageId);
            const link = document.createElement('a');
            link.download = filename;
            link.href = image.src;
            link.click();
        }

        async function downloadZip() {
            toggleLoading(true);
            try {
                const response = await fetch('/download-zip', {
                    method: 'POST',
                    body: JSON.stringify({
                        combined_image: document.getElementById('combinedResult').src,
                        answer_image: document.getElementById('answerResult').src
                    }),
                    headers: {
                        'Content-Type': 'application/json'
                    }
                });

                if (response.ok) {
                    const blob = await response.blob();
                    const url = window.URL.createObjectURL(blob);
                    const link = document.createElement('a');
                    link.href = url;
                    link.download = 'results.zip';
                    link.click();
                    window.URL.revokeObjectURL(url);
                } else {
                    alert('Error creating ZIP file');
                }
            } catch (error) {
                console.error('Error:', error);
                alert('Error creating ZIP file');
            } finally {
                toggleLoading(false);
            }
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/process', methods=['POST'])
def process():
    try:
        original = request.form['original']
        modified = request.form['modified']
        settings = json.loads(request.form['settings'])
        
        result = process_images(original, modified, settings)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download-zip', methods=['POST'])
def download_zip():
    try:
        data = request.json
        
        # Create a ZIP file in memory
        memory_file = BytesIO()
        with zipfile.ZipFile(memory_file, 'w') as zf:
            # Create a text file with information
            message = (
"Thank you for using our Find the Difference puzzles!\n\n"

"Visit https://findthediff.ddnsfree.com for more challenges and fun puzzles.\n\n"

"If you are the owner of any image on this site and do not wish it to appear, please contact us at findthediff-main@iname.com.\n\n"

"Have a great day and keep exploring!"
            )
            zf.writestr('message.txt', message)
            
            # Add the images
            for image_name, image_data in [
                ('combined.jpg', data['combined_image']),
                ('answer.jpg', data['answer_image'])
            ]:
                # Convert base64 to bytes
                image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                zf.writestr(image_name, image_bytes)
        
        # Prepare the ZIP file for download
        memory_file.seek(0)
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name='results.zip'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=85, debug=True)