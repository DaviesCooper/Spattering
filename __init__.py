print("Initializing Spattering")
from src import program
import cv2

image_path = '/home/zigzalgo/git/Spattering/the_thinker.png'
temp_folder = '/home/zigzalgo/git/Spattering/output'

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
program.preprocess_image(image, temp_folder)
vor, points = program.create_initial_points(image, temp_folder, 1000)
