import os
from PIL import Image

# Directory where your png files are stored
input_dir = 'D:/YOLOX Dataset/images/val'  # Change to the folder containing the png images
output_dir = 'D:/YOLOX Dataset/images/jpg'  # Change to the folder where you want to save PNG images

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Convert images from 00000.png to 00599.png
for i in range(600):
    png_filename = f"{i:05d}.png"  # Format the filename (e.g., 00000.png, 00001.png, ...)
    png_filepath = os.path.join(input_dir, png_filename)
    
    if os.path.exists(png_filepath):
        # Open the png image
        with Image.open(png_filepath) as img:
            # Define output filename (e.g., 00000.png)
            output_filename = f"{i:05d}.png"
            output_filepath = os.path.join(output_dir, output_filename)
            
            # Save the image in PNG format
            img.save(output_filepath)
            print(f"Converted {png_filename} to {output_filename}")
    else:
        print(f"File {png_filename} not found!")

print("Conversion complete!")
