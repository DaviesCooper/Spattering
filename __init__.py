print("Initializing Spattering")
from src import program
from src.utils import images_to_video, delete_files_in_directory
import numpy as np
import cv2

image_path = '/home/zigzalgo/git/Spattering/the_thinker.png'
temp_folder = '/home/zigzalgo/git/Spattering/output'
iteration_folder = '/home/zigzalgo/git/Spattering/output/iters'
num_points = 5000

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
angles, magnitudes = program.preprocess_image(image, temp_folder)
points = (np.random.rand(num_points, 2) * np.array([image.shape[0] - 1, image.shape[1]-1])).astype(np.uint16)  
for i in range(200):
    points = program.relax_points(image, angles, magnitudes, iteration_folder, points, i + 1)

images_to_video(iteration_folder, "/home/zigzalgo/git/Spattering/output/relaxing.mp4")
delete_files_in_directory(iteration_folder)
