import cv2
import os
import numpy as np
from scipy.spatial import Voronoi
from src.utils import polygon_centroid, left_pad
from src.preprocessing import generate_magnitude_and_angle_maps, angle_magnitude_to_hsv
from src.visualizations import draw_arrows_on_image, draw_circles_on_image, draw_voronoi_on_image


def preprocess_image(image, temp_folder): 
    angles, magnitudes = generate_magnitude_and_angle_maps(image, 5)
    if not (os.path.isdir(temp_folder)):
        return angles, magnitudes
    arrow_plot = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    arrow_plot = draw_arrows_on_image(arrow_plot, angles, magnitudes, (0,0,255))
    cv2.imwrite(os.path.join(temp_folder, "arrow_plot.png"), arrow_plot)
    hsv_image = cv2.cvtColor(angle_magnitude_to_hsv(angles, magnitudes), cv2.COLOR_HSV2BGR)
    cv2.imwrite(os.path.join(temp_folder, "hsv_plot.png"), hsv_image)
    return angles, magnitudes 

def create_initial_points(image, temp_folder, num_points):
    height, width = image.shape
    points = (np.random.rand(num_points, 2) * np.array([height, width])).astype(np.uint16)  
    vor = Voronoi(points)
    if not (os.path.isdir(temp_folder)):
        return vor, points
    random_points = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    random_points = draw_circles_on_image(random_points, points, (255,0,0))
    voronoi_diagram = draw_voronoi_on_image(random_points, vor, (0,255,0))
    cv2.imwrite(os.path.join(temp_folder, "voronoi_diagram.png"), voronoi_diagram)

    return vor, points

def relax_points(image, angles, magnitudes, temp_folder, voronoi, points, iteration):
    pre_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    pre_image = draw_circles_on_image(pre_image, points, (255, 0, 0))

    for point_index in range(len(points)):
        y, x = points[point_index]
        region_index = voronoi.point_region[point_index] 
        region = voronoi.regions[region_index]
        if not region or -1 in region:
            continue
        
        vertices = [voronoi.vertices[i] for i in region]

        centroid = polygon_centroid(vertices)

        newx = x.astype(np.int32) + int(magnitudes[y, x] * np.cos(np.radians(angles[y, x])))
        newy = y.astype(np.int32) + int(magnitudes[y, x] * np.sin(np.radians(angles[y, x])))

        newx += centroid[1]
        newy += centroid[0]

        newx /= 2
        newy /= 2
        
        points[point_index] = np.array([np.clip(newy, 0, image.shape[0] - 1), np.clip(newx, 0, image.shape[1] - 1)])
    
    relaxed_image = draw_circles_on_image(pre_image, points, (0,255,0))
    
    fileName = left_pad(f"{iteration}", 6, "0")
    cv2.imwrite(os.path.join(temp_folder, f"relaxed{fileName}.png"), relaxed_image)
    
    return points

        
        