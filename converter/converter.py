#!/usr/bin/env python3
import sys
import os
import argparse

# Dependency check: Ensure Pillow is installed.
try:
    from PIL import Image
except ImportError:
    choice = input(
        "The 'Pillow' library is not installed. Would you like to install it automatically? (y/n): "
    ).strip().lower()
    if choice in ['y', 'yes', '']:
        try:
            import subprocess

            subprocess.check_call([sys.executable, "-m", "pip", "install", "pillow"])
            print("Pillow installed successfully. Please re-run the script.")
            sys.exit(0)
        except Exception as e:
            print(f"Automatic installation failed: {e}")
            print("Please install Pillow manually by running: pip install pillow")
            sys.exit(1)
    else:
        print("Please install Pillow manually by running: pip install pillow")
        sys.exit(1)


def convert_to_jpg(file_path):
    """
    Open an image file, convert it to JPEG (RGB mode), and save it.
    Returns the new file's path on success or None on failure.
    """
    try:
        with Image.open(file_path) as im:
            # Convert image to RGB (this removes alpha transparency)
            rgb_im = im.convert("RGB")
            new_file = os.path.splitext(file_path)[0] + ".jpg"
            rgb_im.save(new_file, quality=95)
            return new_file
    except Exception as e:
        print(f"Error converting {file_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert image files (e.g. PNG, BMP, GIF, TIFF) to JPEG. "
            "By default, the original file is removed after conversion, "
            "unless the -remove_off flag is provided."
        )
    )
    parser.add_argument(
        "-remove_off",
        action="store_true",
        help="Keep original file(s) after conversion (do not remove them).",
    )
    args = parser.parse_args()

    # Get the directory where the script is located.
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Supported input file extensions (excluding JPEG/JPG).
    supported_ext = (".png", ".bmp", ".gif", ".tif", ".tiff")

    converted_count = 0

    # Process each file in this directory.
    for filename in os.listdir(script_dir):
        lower_name = filename.lower()
        if lower_name.endswith(supported_ext):
            file_path = os.path.join(script_dir, filename)
            print(f"Converting '{filename}'...")
            new_file = convert_to_jpg(file_path)
            if new_file:
                print(f" -> Created '{os.path.basename(new_file)}'.")
                converted_count += 1
                if not args.remove_off:
                    try:
                        os.remove(file_path)
                        print(f" -> Removed original file '{filename}'.")
                    except Exception as e:
                        print(f" -> Could not remove '{filename}' due to: {e}")
            else:
                print(f" -> Skipped '{filename}' due to an error.")

    print(f"\nConversion complete. Total files converted: {converted_count}")


if __name__ == "__main__":
    main()
