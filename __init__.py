print("Initializing Spattering")
from src import program
import cv2

image_path = '/home/zigzalgo/git/Spattering/the_thinker.png'
temp_folder = '/home/zigzalgo/git/Spattering/output'
iteration_folder = '/home/zigzalgo/git/Spattering/output/iters'

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
angles, magnitudes = program.preprocess_image(image, temp_folder)
vor, points = program.create_initial_points(image, temp_folder, 1000)
new_points = program.relax_points(image, angles, magnitudes, temp_folder, vor, points, 1)
for i in range(100):
    new_points = program.relax_points(image, angles, magnitudes, iteration_folder, vor, new_points, i + 1)