import os
from PIL import Image

# Directory where your PPM files are stored
input_dir = 'D:\YOLOX Dataset\images/val'  # Change to the folder containing the PPM images
output_dir = 'D:\YOLOX Dataset\images\jpg'  # Change to the folder where you want to save PNG images

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Convert images from 00000.ppm to 00599.ppm
for i in range(600):
    ppm_filename = f"{i:05d}.ppm"  # Format the filename (e.g., 00000.ppm, 00001.ppm, ...)
    ppm_filepath = os.path.join(input_dir, ppm_filename)
    
    if os.path.exists(ppm_filepath):
        # Open the PPM image
        with Image.open(ppm_filepath) as img:
            # Define output filename (e.g., 00000.png)
            output_filename = f"{i:05d}.png"
            output_filepath = os.path.join(output_dir, output_filename)
            
            # Save the image in PNG format
            img.save(output_filepath)
            print(f"Converted {ppm_filename} to {output_filename}")
    else:
        print(f"File {ppm_filename} not found!")

print("Conversion complete!")
