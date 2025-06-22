#!/usr/bin/env python3
import sys
import os
import argparse
import zipfile
import subprocess
import importlib

# Global flag: if set (by the -w flag), skip dependency instructions.
SKIP_DEPENDENCY = False
if '-w' in sys.argv:
    SKIP_DEPENDENCY = True
    sys.argv.remove('-w')

def check_dependency_manual(module_name, pip_module, manual_cmd):
    """
    Try to import a module by name. If the import fails and the user hasn't opted
    to skip auto-installation, prompt them if they'd like to see the manual-install
    command. If they answer 'y', print the command and exit; if 'n', continue with
    a warning.
    """
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        if SKIP_DEPENDENCY:
            print(f"Warning: {module_name} is not installed. Skipping dependency instructions as per user request. (You may encounter errors later.)")
            return False
        else:
            prompt = (f"The '{pip_module}' library (module '{module_name}') is either not installed or missing required libraries.\n"
                      f"For example, you might see a 'libGL.so.1' error. Would you like to see the command to install it manually? (y/n): ")
            choice = input(prompt).strip().lower()
            if choice in ['y', 'yes']:
                print("\nPlease run the following command in your terminal to install the necessary dependency:")
                print(manual_cmd)
                sys.exit(1)
            else:
                print(f"Continuing without installing {pip_module}. You may experience errors later.")
                return False

# Check dependencies one by one.
if not check_dependency_manual("cv2", "opencv-python", "sudo apt-get update && sudo apt-get install libgl1"):  # adjust as needed
    pass
if not check_dependency_manual("numpy", "numpy", "pip install numpy"):
    pass

# Now attempt to import our dependencies.
try:
    import cv2
except ImportError:
    print("Error: cv2 is still not importable. Exiting.")
    sys.exit(1)
try:
    import numpy as np
except ImportError:
    print("Error: numpy is still not importable. Exiting.")
    sys.exit(1)

def process_images(original_path, modified_path, threshold_value, min_area, dilation_iter,
                   circle_thickness, circle_color, overlay_color=None, overlay_opacity=0):
    """
    Load the original and modified images, create an answer image by overlaying a translucent
    difference mask and drawing circles around detected differences, and generate a side-by-side
    combined image.
    
    Parameters:
      overlay_color: a BGR tuple that will be used to mark pixel differences.
      overlay_opacity: an integer from 0 to 255 controlling the overlay's opacity (0 means off).
    """
    # Load images
    img1 = cv2.imread(original_path)
    img2 = cv2.imread(modified_path)
    if img1 is None or img2 is None:
        raise ValueError("One or both image files could not be loaded.")
    
    # Resize modified image if necessary
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Compute the absolute difference and threshold it
    diff = cv2.absdiff(gray1, gray2)
    _, thresh_img = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Dilate to accentuate the difference mask
    kernel = np.ones((5, 5), np.uint8)
    thresh_img = cv2.dilate(thresh_img, kernel, iterations=dilation_iter)
    
    # Retrieve contours from the threshold mask
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Start with a copy of the modified image as the answer image.
    answer_img = img2.copy()
    
    # If an overlay is desired, mark the differing areas with the overlay color.
    if overlay_color is not None and overlay_opacity > 0:
        alpha = overlay_opacity / 255.0  # normalize opacity
        # Create a boolean mask where differences are present.
        mask = thresh_img == 255
        # Convert answer_img to float32 for blending.
        temp = answer_img.astype(np.float32)
        # For all pixels in the mask, blend the original pixel with the overlay color.
        # The overlay_color is expected to be a tuple in BGR order.
        overlay_arr = np.array(overlay_color, dtype=np.float32)
        temp[mask] = (1 - alpha) * temp[mask] + alpha * overlay_arr
        answer_img = np.clip(temp, 0, 255).astype(np.uint8)
    
    # Draw circles around each detected contour (on top of the overlay).
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(answer_img, center, radius, circle_color, circle_thickness)
    
    # Create a side-by-side combined image with spacing.
    spacing = 10
    height = max(img1.shape[0], img2.shape[0])
    width = img1.shape[1] + img2.shape[1] + spacing
    combined_img = np.full((height, width, 3), 255, dtype=np.uint8)
    combined_img[0:img1.shape[0], 0:img1.shape[1]] = img1
    combined_img[0:img2.shape[0], img1.shape[1]+spacing: img1.shape[1]+spacing+img2.shape[1]] = img2

    return combined_img, answer_img

def create_zip(zip_filename, combined_filename, answer_filename, message_filename):
    """
    Bundle the combined image, answer image, and a message text file into a ZIP archive.
    """
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        zipf.write(combined_filename, arcname=os.path.basename(combined_filename))
        zipf.write(answer_filename, arcname=os.path.basename(answer_filename))
        zipf.write(message_filename, arcname=os.path.basename(message_filename))

def main():
    parser = argparse.ArgumentParser(
        description="Generate difference images by producing a side-by-side combined image and an answer image with both an overlay (red pixels) and circles marking differences. Package these along with a message file into a ZIP archive."
    )
    parser.add_argument('--original', required=True, help="Path to the original image (e.g., og.jpg)")
    parser.add_argument('--modified', required=True, help="Path to the modified image (e.g., diff.jpg)")
    parser.add_argument('--zipname', required=True, help="Output ZIP file name (e.g., puzzle1.zip)")
    parser.add_argument('--threshold', type=int, default=30, help="Threshold for detecting differences (default: 30)")
    parser.add_argument('--min_area', type=int, default=40, help="Minimum contour area to consider (default: 40)")
    parser.add_argument('--dilation_iter', type=int, default=2, help="Number of dilation iterations (default: 2)")
    parser.add_argument('--circle_thickness', type=int, default=2, help="Thickness of the drawn circles (default: 2)")
    parser.add_argument('--circle_color', type=str, default="FF0000", help="Hex color for the circles (default: FF0000 for red)")
    parser.add_argument('--overlay_color', type=str, default="FF0000", help="Hex color for the difference overlay (default: FF0000 for red)")
    parser.add_argument('--overlay_opacity', type=int, default=100, help="Opacity (0-255) for the difference overlay (default: 100)")
    args = parser.parse_args()
    
    # Convert hex color strings to BGR tuples (since OpenCV uses BGR).
    def hex_to_bgr(hex_color):
        hex_color = hex_color.lstrip('#')
        if len(hex_color) != 6:
            raise ValueError("Invalid hex color format. Use 6 hexadecimal characters.")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (b, g, r)
    
    circle_color_bgr = hex_to_bgr(args.circle_color)
    overlay_color_bgr = hex_to_bgr(args.overlay_color)
    
    try:
        combined_img, answer_img = process_images(
            args.original, args.modified,
            args.threshold, args.min_area, args.dilation_iter,
            args.circle_thickness, circle_color_bgr,
            overlay_color=overlay_color_bgr, overlay_opacity=args.overlay_opacity
        )
    except Exception as e:
        print(f"Error processing images: {e}")
        sys.exit(1)
    
    combined_filename = "combined.jpg"
    answer_filename = "answer.jpg"
    message_filename = "message.txt"
    
    cv2.imwrite(combined_filename, combined_img)
    cv2.imwrite(answer_filename, answer_img)
    
    message_text = (
        "Thank you for using our Find the Difference puzzles!\n\n"
        "Visit https://findthediff.ddnsfree.com for more challenges and fun puzzles.\n\n"
        "If you are the owner of any image on this site and do not wish it to appear, "
        "please contact us at findthediff-main@iname.com.\n\n"
        "Have a great day and keep exploring!"
    )
    with open(message_filename, 'w', encoding='utf-8') as f:
        f.write(message_text)
    
    create_zip(args.zipname, combined_filename, answer_filename, message_filename)
    print(f"ZIP file '{args.zipname}' created successfully!")

if __name__ == "__main__":
    main()
