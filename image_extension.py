from PIL import Image
import os

def convert_png_to_jpg(png_path, jpg_path):
    # Open the PNG image
    png_image = Image.open(png_path)

    # Convert PNG to JPEG
    png_image.convert("RGB").save(jpg_path, "JPEG")

    # Close the image
    png_image.close()

def batch_convert_png_to_jpg(png_folder, jpg_folder):
    # Create the JPEG folder if it doesn't exist
    if not os.path.exists(jpg_folder):
        os.makedirs(jpg_folder)

    # List all files in the PNG folder
    png_files = os.listdir(png_folder)

    # Filter out only the PNG files
    png_files = [file for file in png_files if file.lower().endswith('.png')]

    # Convert each PNG file to JPEG
    for png_file in png_files:
        png_path = os.path.join(png_folder, png_file)
        jpg_path = os.path.join(jpg_folder, os.path.splitext(png_file)[0] + '.jpg')
        convert_png_to_jpg(png_path, jpg_path)

# Example usage
png_folder = "Frames/frames2"
jpg_folder = "Frames/frames2"
batch_convert_png_to_jpg(png_folder, jpg_folder)
