import os
import cv2
import numpy as np
from pathlib import Path

def rectify_single_image(input_path, output_path, color_mode=True):
    """Rectify single catadioptric image"""
    if color_mode:
        cata_image = cv2.imread(input_path)
        cata_image = cv2.rotate(cata_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        cata_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        cata_image = cv2.rotate(cata_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    r1, r2, w, h = 242, 511, 1024, 1024
    angle_res = 1800
    
    if color_mode:
        pixels = np.zeros((r2-r1+1, angle_res, 3), dtype=np.uint8)
    else:
        pixels = np.zeros((r2-r1+1, angle_res), dtype=np.uint8)
    
    theta = np.linspace(-np.pi, np.pi, angle_res, endpoint=False)
    
    for radius in range(r1, r2+1):
        x_centered = radius * np.cos(theta)
        y_centered = radius * np.sin(theta)
        x = x_centered + w/2
        y = y_centered + h/2
        
        for i in range(len(y)):
            x_coord = int(x[i])
            y_coord = int(y[i])
            if color_mode:
                pixels[radius-r1][i] = cata_image[y_coord][x_coord]
            else:
                pixels[radius-r1][i] = cata_image[y_coord][x_coord]
    
    cv2.imwrite(output_path, pixels)

def rectify_folder(input_folder, output_folder, color_mode=True, start_id=None, end_id=None):
    """Rectify images in input folder and save to output folder
    
    Args:
        input_folder: Input folder containing images
        output_folder: Output folder for rectified images
        color_mode: Whether to process in color mode
        start_id: Starting image ID (inclusive), None for all
        end_id: Ending image ID (inclusive), None for all
    """
    os.makedirs(output_folder, exist_ok=True)
    
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if start_id is not None or end_id is not None:
        image_files = sorted(image_files)
        if start_id is not None:
            image_files = [f for f in image_files if int(f.split('_')[-1].split('.')[0]) >= start_id]
        if end_id is not None:
            image_files = [f for f in image_files if int(f.split('_')[-1].split('.')[0]) <= end_id]
    
    print(f"Rectifying {len(image_files)} images in {input_folder} to {output_folder}")
    for img_file in image_files:
        input_path = os.path.join(input_folder, img_file)
        output_path = os.path.join(output_folder, f"rectified_{img_file}")
        rectify_single_image(input_path, output_path, color_mode)
        print(f"Rectified: {img_file} -> rectified_{img_file}")

if __name__ == "__main__":
    input_folder = "../path_following/panorama_cassini"
    output_folder = "../path_following/rectified_panorama_cassini"

    # rectify_folder(input_folder, output_folder, color_mode=False)  # Set to True for color mode
    rectify_folder(input_folder, output_folder, start_id=1, end_id=2701)
