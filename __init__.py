from src import PreprocessingStippleGenerator, DebugOptions, StandardStippleGenerator, StippleGeneratorV2
import cv2

image_path = '/home/zigzalgo/git/Spattering/wedding.png'
debugOptions = DebugOptions("/home/zigzalgo/git/Spattering/debug", True, True, True)
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
numPoints= 40000
dpi = 28.8888
rad = 1/32
minRad = 1/32
iterations = 10
x = StippleGeneratorV2(image, numPoints, dpi, rad, rad, iterations, debugOptions)
points = list(x.stipple())
x.exportToPNG(points, "/home/zigzalgo/git/Spattering/weddingPlot.png")
x.exportToSVG(points, "/home/zigzalgo/git/Spattering/wedding.svg")
x.exportToSVG(points, "/home/zigzalgo/git/Spattering/wedding2.svg", True)
