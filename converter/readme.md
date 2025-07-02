# Find the Differences Image Tool

This is a web-based tool that helps find and highlight differences between two images. It uses computer vision techniques to detect changes between images and marks them with numbered circles.

## Features

- Drag and drop image upload
- Adjustable detection settings
- Overlay highlighting of differences
- Numbered circles showing differences
- Combined view with both images
- Download options for results
- Mobile responsive design

## Installation

1. Install Python 3.7+ if you haven't already
2. Clone this repository or download the files
3. Install the required dependencies:

```bash
pip install flask==2.0.1 numpy==1.21.1 opencv-python==4.5.3.56
```

Or use the requirements.txt:

```bash
pip install -r requirements.txt
```

## Project Structure

```
find-differences/
├── requirements.txt
├── app.py 
├── templates/
│   └── index.html
└── README.md
```

## Usage

1. Run the application:
```bash
python app.py
```

2. Open a web browser and go to:
```
http://localhost:85
```

3. Upload two images:
   - Original image on the left
   - Modified image on the right
   - Either drag & drop or click to select files

4. Adjust settings as needed:
   - Threshold: Sensitivity of difference detection
   - Minimum Area: Smallest difference to detect
   - Dilation Iterations: Expand detected areas
   - Circle Thickness: Width of marking circles
   - Colors & Opacity: Visual appearance settings
   - Touch Distance: How close differences need to be to group
   - Spacing: Visual layout settings

5. Click "Process Images" to analyze

6. View and download results:
   - Combined view shows both images side by side
   - Answer view shows differences marked with numbered circles
   - Download individual images or ZIP with both results

## Settings Explained

- **Threshold** (1-255): How different pixels need to be to count as a difference
- **Minimum Area** (1-200): Filters out differences smaller than this size
- **Dilation Iterations** (0-10): Expands detected areas to help connect nearby differences
- **Circle Thickness** (1-10): Width of the circles marking differences
- **Touch Distance** (1-50): How close differences need to be to be grouped together
- **Image Spacing** (0-100): Space between images in combined view
- **Separator Thickness** (1-10): Width of line between images
- **Colors**: Customize the appearance of markings and overlays

## Troubleshooting

If images fail to process:
- Check that both images are valid image files
- Try images of similar dimensions
- Reduce image sizes if they're very large
- Check browser console for specific errors
- Make sure all dependencies are installed correctly

## Dependencies

- Flask: Web framework
- NumPy: Numerical processing
- OpenCV (cv2): Computer vision and image processing

## Browser Compatibility

Tested and working on:
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Performance Notes

- Large images may take longer to process
- Processing happens on the server side
- Maximum image size limit is 100MB