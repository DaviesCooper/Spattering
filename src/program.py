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

def relax_points(image, angles, magnitudes, temp_folder, points, iteration):
    voronoi = Voronoi(points)
    for point_index in range(len(points)):
        y, x = points[point_index]
        if image[y, x] == 255 or image[y,x] == 0:
            continue
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

        newx+=x
        newy+=y

        newx /= 3
        newy /= 3
        
        points[point_index] = np.array([np.clip(newy, 0, image.shape[0] - 1), np.clip(newx, 0, image.shape[1] - 1)])
    
    if not (os.path.isdir(temp_folder)):
        return points
    
    pre_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    points_image = draw_circles_on_image(pre_image, points, (255,0,0))
    voronoi_diagram = draw_voronoi_on_image(points_image, voronoi, (0,255,0))
    fileName = left_pad(f"{iteration}", 6, "0")
    cv2.imwrite(os.path.join(temp_folder, f"relaxed{fileName}.png"), voronoi_diagram)
    
    return points

        
        