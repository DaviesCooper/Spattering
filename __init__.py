from src import PreprocessingStippleGenerator, DebugOptions, StandardStippleGenerator
import cv2

image_path = '/home/zigzalgo/git/Spattering/the_thinker.png'
debugOptions = DebugOptions("/home/zigzalgo/git/Spattering/standarddebug", True, True, True)
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
numPoints= 8000,
dpi = 25,
rad = 1/16
iterations = 5
x = StandardStippleGenerator(image, numPoints,  dpi, rad, iterations, debugOptions)
x.stipple()
x.exportToSVG("/home/zigzalgo/git/Spattering/the_thinker.svg")
