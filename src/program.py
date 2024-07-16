import cv2
import os
from src.preprocessing import generate_magnitude_and_angle_maps, angle_magnitude_to_hsv
from src.visualizations import draw_arrows_on_image


def preprocess_image(image_path, temp_folder): 
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    angles, magnitudes = generate_magnitude_and_angle_maps(image, 3)
    if not (os.path.isdir(temp_folder)):
        return angles, magnitudes
    
    arrow_plot = draw_arrows_on_image(image, angles, magnitudes)
    cv2.imwrite(os.path.join(temp_folder, "arrow_plot.png"), arrow_plot)
    hsv_image = cv2.cvtColor(angle_magnitude_to_hsv(angles, magnitudes), cv2.COLOR_HSV2BGR)
    cv2.imwrite(os.path.join(temp_folder, "hsv_plot.png"), hsv_image)
    return angles, magnitudes 
