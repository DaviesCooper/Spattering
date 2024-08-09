from src import PreprocessingStippleGenerator, DebugOptions
import cv2

image_path = '/home/zigzalgo/git/Spattering/the_thinker.png'
debugOptions = DebugOptions("/home/zigzalgo/git/Spattering/debug", True, True, True)
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
x = PreprocessingStippleGenerator(image, 6000,  25, 1/16, 1, 50, debugOptions)
x.stipple()

