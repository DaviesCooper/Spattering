print("Initializing Spattering")
from src import program

image_path = '/home/zigzalgo/git/Spattering/the_thinker.png'
temp_folder = '/home/zigzalgo/git/Spattering/output'
program.preprocess_image(image_path, temp_folder)