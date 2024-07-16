import cv2
import os
import numpy as np
from src.preprocessing import generate_magnitude_and_angle_maps, angle_magnitude_to_hsv
from src.visualizations import draw_arrows_on_image, draw_circles_on_image


def preprocess_image(image, temp_folder): 
    angles, magnitudes = generate_magnitude_and_angle_maps(image, 3)
    if not (os.path.isdir(temp_folder)):
        return angles, magnitudes
    
    arrow_plot = draw_arrows_on_image(image, angles, magnitudes)
    cv2.imwrite(os.path.join(temp_folder, "arrow_plot.png"), arrow_plot)
    hsv_image = cv2.cvtColor(angle_magnitude_to_hsv(angles, magnitudes), cv2.COLOR_HSV2BGR)
    cv2.imwrite(os.path.join(temp_folder, "hsv_plot.png"), hsv_image)
    return angles, magnitudes 

def create_initial_points(image, temp_folder, num_points):
    height, width = image.shape
    points = (np.random.rand(num_points, 2) * np.array([height, width])).astype(np.uint16)
    if not (os.path.isdir(temp_folder)):
        return points
    
    random_points = draw_circles_on_image(image, points)
    cv2.imwrite(os.path.join(temp_folder, "initial_points.png"), random_points)
    return points
    
