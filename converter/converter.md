### How to Use

1. **Place the Script:**  
   Save the file as `converter.py` in the folder where your images are stored (e.g. `/png/`).

2. **Default Behavior:**  
   By default, run the script with:
   ```
   python converter.py
   ```
   It will locate any images matching the supported extensions (like `pn.png`), convert them to JPEG (`pn.jpg`), and then remove the original file.

3. **Keep Originals:**  
   To keep the original images, run:
   ```
   python converter.py -remove_off
   ```

4. **Dependency Check:**  
   If Pillow isn’t installed, you’ll be prompted on whether you’d like to have it installed automatically. Answer “y” to install it automatically or “n” to do it yourself.

This should provide a complete solution that makes file conversions and dependency management user-friendly. Enjoy!