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
    to skip auto-installation, prompt the user if they'd like to see the command
    to install the missing dependency manually.

    If the user answers 'y', print the command and exit.
    If the user answers 'n' (or opts to skip via -w), warn and continue.
    Returns True if the module is importable, or False if not.
    """
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        if SKIP_DEPENDENCY:
            print(f"Warning: {module_name} is not installed. Skipping dependency instructions as per user request. (You may encounter errors later.)")
            return False
        else:
            prompt = (f"The '{pip_module}' library (module '{module_name}') is either not installed or is missing some required system libraries.\n"
                      f"For example, you might see an error like 'libGL.so.1: cannot open shared object file.'\n"
                      f"Would you like to see the command to install it manually? (y/n): ")
            choice = input(prompt).strip().lower()
            if choice in ['y', 'yes']:
                print("\nPlease run the following command in your terminal to install the necessary dependency:")
                print(manual_cmd)
                sys.exit(1)
            else:
                print(f"Continuing without installing {pip_module}. You may experience errors later.")
                return False


# Check dependencies one by one.
#
# For opencv-python, a common missing system dependency is libGL.so.1.
# On Debian/Ubuntu systems you can usually fix this by executing:
#   sudo apt-get install libgl1-mesa-glx
#
if not check_dependency_manual("cv2", "opencv-python", "sudo apt-get update && sudo apt-get install libgl1-mesa-glx"):
    pass  # Continue regardless; note that later code using cv2 may fail.

# For numpy, a simple pip install should suffice.
if not check_dependency_manual("numpy", "numpy", "pip install numpy"):
    pass  # Continue regardless.

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


def process_images(original_path, modified_path, threshold_value, min_area, dilation_iter, circle_thickness, circle_color):
    """
    Load the original and modified images, create an "answer" image by detecting differences and drawing circles,
    and generate a side-by-side combined image.
    """
    # Load the images
    img1 = cv2.imread(original_path)
    img2 = cv2.imread(modified_path)
    if img1 is None or img2 is None:
        raise ValueError("One or both image files could not be loaded.")
    
    # If dimensions differ, resize the modified image to match the original.
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Convert images to grayscale.
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Compute the absolute difference and threshold it.
    diff = cv2.absdiff(gray1, gray2)
    _, thresh_img = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Dilate the threshold image to accentuate differences.
    kernel = np.ones((5, 5), np.uint8)
    thresh_img = cv2.dilate(thresh_img, kernel, iterations=dilation_iter)
    
    # Find contours of the difference regions.
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Copy the modified image to draw circles.
    answer_img = img2.copy()
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
    combined_img = np.full((height, width, 3), 255, dtype=np.uint8)  # white background.
    combined_img[0:img1.shape[0], 0:img1.shape[1]] = img1
    combined_img[0:img2.shape[0], img1.shape[1]+spacing: img1.shape[1]+spacing+img2.shape[1]] = img2

    return combined_img, answer_img


def create_zip(zip_filename, combined_filename, answer_filename, message_filename):
    """
    Bundle the combined image, answer image, and text message file into a ZIP archive.
    """
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        zipf.write(combined_filename, arcname=os.path.basename(combined_filename))
        zipf.write(answer_filename, arcname=os.path.basename(answer_filename))
        zipf.write(message_filename, arcname=os.path.basename(message_filename))


def main():
    parser = argparse.ArgumentParser(
        description="Generate difference images: produce a side-by-side combined image and an answer image with circles, then bundle these with a message file into a ZIP archive."
    )
    parser.add_argument('--original', required=True, help="Path to the original image (e.g., og.jpg)")
    parser.add_argument('--modified', required=True, help="Path to the modified image (e.g., diff.jpg)")
    parser.add_argument('--zipname', required=True, help="Name for the output ZIP file (e.g., puzzle1.zip)")
    parser.add_argument('--threshold', type=int, default=30, help="Threshold for detecting differences (default: 30)")
    parser.add_argument('--min_area', type=int, default=40, help="Minimum contour area to consider (default: 40)")
    parser.add_argument('--dilation_iter', type=int, default=2, help="Number of dilation iterations (default: 2)")
    parser.add_argument('--circle_thickness', type=int, default=2, help="Thickness of the drawn circles (default: 2)")
    parser.add_argument('--circle_color', type=str, default="FF0000", help="Hex color for the circle (default: FF0000 for red)")
    args = parser.parse_args()
    
    # Convert the hex color (e.g., "FF0000") to a BGR tuple (OpenCV uses BGR).
    hex_color = args.circle_color.lstrip('#')
    if len(hex_color) != 6:
        raise ValueError("Invalid hex color format. It should be 6 hexadecimal characters.")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    circle_color_bgr = (b, g, r)
    
    try:
        combined_img, answer_img = process_images(
            args.original, args.modified,
            args.threshold, args.min_area, args.dilation_iter,
            args.circle_thickness, circle_color_bgr
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
