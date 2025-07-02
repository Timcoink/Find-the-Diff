import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict
import math

@dataclass
class DifferenceRegion:
    contour: np.ndarray
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    area: float
    intensity: float
    shape_complexity: float
    neighbors: List[int] = None
    grouped: bool = False

class SmartDifferenceDetector:
    def __init__(self):
        self.regions = []
        self.groups = []
        
    def analyze_region(self, img1, img2, contour):
        x, y, w, h = cv2.boundingRect(contour)
        center = (x + w//2, y + h//2)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        shape_complexity = perimeter * perimeter / (4 * np.pi * area) if area > 0 else float('inf')
        
        roi1 = img1[y:y+h, x:x+w]
        roi2 = img2[y:y+h, x:x+w]
        diff_intensity = np.mean(cv2.absdiff(roi1, roi2))
        
        return DifferenceRegion(
            contour=contour,
            bbox=(x, y, w, h),
            center=center,
            area=area,
            intensity=diff_intensity,
            shape_complexity=shape_complexity
        )

    def should_group(self, region1, region2, touch_distance):
        dx = region2.center[0] - region1.center[0]
        dy = region2.center[1] - region1.center[1]
        distance = math.sqrt(dx*dx + dy*dy)
        
        avg_area = (region1.area + region2.area) / 2
        relative_distance = distance / math.sqrt(avg_area)
        size_ratio = max(region1.area, region2.area) / min(region1.area, region2.area)
        angle = math.degrees(math.atan2(dy, dx)) % 360
        
        is_close = distance <= touch_distance
        is_similar_size = size_ratio < 3.0
        has_similar_intensity = abs(region1.intensity - region2.intensity) < 30
        has_similar_complexity = abs(region1.shape_complexity - region2.shape_complexity) < 1.0
        is_aligned = angle < 45 or abs(angle - 90) < 45 or abs(angle - 180) < 45
        
        return (is_close and is_similar_size and has_similar_intensity) or \
               (relative_distance < 2.0 and is_aligned and has_similar_complexity)

    def group_differences(self, img1, img2, contours, touch_distance):
        self.regions = [self.analyze_region(img1, img2, cnt) for cnt in contours]
        self.groups = []
        
        self.regions.sort(key=lambda r: r.area, reverse=True)
        
        for region in self.regions:
            if region.grouped:
                continue
                
            current_group = [region]
            region.grouped = True
            
            group_changed = True
            while group_changed:
                group_changed = False
                for other_region in self.regions:
                    if other_region.grouped:
                        continue
                        
                    if any(self.should_group(grouped_region, other_region, touch_distance)
                           for grouped_region in current_group):
                        current_group.append(other_region)
                        other_region.grouped = True
                        group_changed = True
                        break
            
            self.groups.append([r.contour for r in current_group])
        
        return self.groups

def process_images(img1_base64, img2_base64, settings):
    def base64_to_numpy(base64_str):
        base64_str = base64_str.split(',')[1]
        img_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    def hex_to_bgr(hex_color):
        hex_color = hex_color.lstrip('#')
        r, g, b = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
        return (b, g, r)

    # Convert images
    img1 = base64_to_numpy(img1_base64)
    img2 = base64_to_numpy(img2_base64)
    
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Process differences
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    blurred = cv2.GaussianBlur(diff, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, settings['threshold'], 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=settings['dilationIter'])

    # Find and filter contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= settings['minArea']]

    # Group differences
    detector = SmartDifferenceDetector()
    groups = detector.group_differences(gray1, gray2, filtered_contours, settings['touchDistance'])

    # Create result images
    answer_img = img2.copy()
    circles_created = 0

    # Draw differences
    circle_color = hex_to_bgr(settings['circleColor'])
    overlay_color = hex_to_bgr(settings['overlayColor'])

    if settings['overlayOpacity'] > 0:
        alpha = settings['overlayOpacity'] / 255
        overlay = np.zeros_like(answer_img)
        overlay[:] = overlay_color
        cv2.addWeighted(overlay, alpha, answer_img, 1 - alpha, 0, answer_img, mask=dilated)

    # Draw circles and numbers
    for idx, group in enumerate(groups, 1):
        all_points = np.concatenate([cnt.reshape(-1, 2) for cnt in group])
        (x, y), radius = cv2.minEnclosingCircle(all_points)
        center = (int(x), int(y))
        radius = int(radius)

        cv2.circle(answer_img, center, radius, circle_color, settings['circleThickness'])
        circles_created += 1

        # Draw number
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = str(idx)
        (text_width, text_height), _ = cv2.getTextSize(text, font, 0.8, 2)
        
        bg_radius = max(text_width, text_height) // 2 + 5
        cv2.circle(answer_img, center, bg_radius, (255, 255, 255), -1)
        circles_created += 1
        
        text_x = int(x - text_width/2)
        text_y = int(y + text_height/2)
        cv2.putText(answer_img, text, (text_x, text_y), font, 0.8, (0, 0, 0), 2)

    # Create combined image
    spacing = settings['imageSpacing']
    sep_thickness = settings['separatorThickness']
    total_spacing = spacing * 2 + sep_thickness

    combined_img = np.full((max(img1.shape[0], img2.shape[0]),
                           img1.shape[1] + total_spacing + img2.shape[1], 3),
                          255, dtype=np.uint8)
    
    combined_img[0:img1.shape[0], 0:img1.shape[1]] = img1
    combined_img[:, img1.shape[1]+spacing:img1.shape[1]+spacing+sep_thickness] = hex_to_bgr(settings['separatorColor'])
    combined_img[0:img2.shape[0], -img2.shape[1]:] = img2

    # Convert to base64
    def numpy_to_base64(img):
        _, buffer = cv2.imencode('.jpg', img)
        return 'data:image/jpeg;base64,' + base64.b64encode(buffer).decode('utf-8')

    return {
        'combined_image': numpy_to_base64(combined_img),
        'answer_image': numpy_to_base64(answer_img),
        'diff_count': len(groups),
        'circles_created': circles_created
    }