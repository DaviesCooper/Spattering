import numpy as np
import cv2
from gradients import generate_magnitude_and_angle_maps, angle_magnitude_to_hsv, draw_arrows_on_image

image_path = '/home/zigzalgo/git/Spattering/the_thinker.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
angles, magnitudes = generate_magnitude_and_angle_maps(image, 3)
arrow_plot = draw_arrows_on_image(image, angles, magnitudes)
cv2.imwrite("/home/zigzalgo/git/Spattering/arrow.png", arrow_plot)
hsv_image = cv2.cvtColor(angle_magnitude_to_hsv(angles, magnitudes), cv2.COLOR_HSV2BGR)
cv2.imwrite("/home/zigzalgo/git/Spattering/hsv.png", hsv_image)
