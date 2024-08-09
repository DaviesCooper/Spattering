import numpy as np
import os
import cv2
import shutil
from scipy.spatial import Voronoi

def polygon_area(vertices: np.ndarray):
    """
    Calculates the area of a polygon

    Args:
    - vertices (np.ndarray): An array of (y, x) coordinates representing the vertices of a polygon

    Returns:
    float: The area of the polygon
    """
    retVal = 0
    for vert_index in range(len(vertices)):
        next_vert_index = (vert_index + 1) % len(vertices)
        retVal += vertices[vert_index][1] * vertices[next_vert_index][0]
        retVal -= vertices[next_vert_index][1] * vertices[vert_index][0]
    return retVal / 2.0

def polygon_centroid(vertices):
    """
    Calculates the centroid of a polygon

    Args:
    - vertices (np.ndarray): An array of (y, x) coordinates representing the vertices of a polygon

    Returns:
    - np.ndarray: An array containing two elements representing the centroid in the form [y, x]
    """
    n = len(vertices)
    signed_area = polygon_area(vertices)
    cx = 0.0
    cy = 0.0
    for vert_index in range(len(vertices)):
        next_vert_index = (vert_index + 1) % len(vertices)
        factor = (vertices[vert_index][1] * vertices[next_vert_index][0]) - vertices[next_vert_index][1] * vertices[vert_index][0]
        cx += (vertices[vert_index][1] + vertices[next_vert_index][1]) * factor
        cy += (vertices[vert_index][0] + vertices[next_vert_index][0]) * factor
    cx /= (6.0 * signed_area)
    cy /= (6.0 * signed_area)
    return np.array([cy, cx])

def left_pad(string: str, length: int, char: str):
    """
    Left pads the given string with the specified character to reach the desired length.

    Args:
    - string (str): The original string to be padded.
    - length (int): The desired length of the padded string.
    - char (str): The character to use for padding.

    Returns:
    - str: The padded string.
    """
    toPad = length - len(string)
    if(toPad > 0):
        return str(char)*toPad + string
    return string

def images_to_video(image_folder: str, video_name: str):
    """
    Convert a sequence of images in a folder into a video.

    Args:
    - image_folder (str): Path to the folder containing images.
    - video_name (str): Output video file name (e.g., 'output.mp4').

    Raises:
    - ValueError: If the specified image folder does not exist or is empty.
    - RuntimeError: If an error occurs during video writing.

    Returns:
    - None
    """
    images = os.listdir(image_folder)
    images.sort()
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, 10, (width, height)) 

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

def ensure_empty_directory(directory_path):
    """
    Ensures that a directory exists and is empty.

    If the directory already exists, all files and subdirectories within it are deleted.
    If the directory does not exist, it is created.

    Args:
    - directory_path (str): The path of the directory to ensure.

    Raises:
    - OSError: If any error occurs while deleting the directory contents or creating the directory.
    """
    # Check if the path exists
    if os.path.exists(directory_path):
        # If the path exists, delete all contents
        shutil.rmtree(directory_path) 
    # then create the directory             
    os.makedirs(directory_path)
    
def draw_arrows_on_image(image: cv2.Mat, angles: np.ndarray, magnitudes: np.ndarray, color: tuple[int, int, int], step:int=20):
    """
    Draw arrows on an image based on angle and magnitude data. Used to represent directional vectors on an image

    Args:
    - image (np.ndarray): The original grayscale image (2D).
    - angles (np.ndarray): A 2D array of angles (in degrees).
    - magnitudes (np.ndarray): A 2D array of magnitudes.
    - step (int): The step size for iterating over pixels to draw arrows (default is 20).

    Returns:
    - np.ndarray: The original image with arrows drawn on it indicating the flow direction and magnitude.
    """

    # copy over original image and convert to bgr
    retVal = image.copy()

    # iterate pixels by step count
    for y in range(0, retVal.shape[0], step):
        for x in range(0, retVal.shape[1], step):
            
            # so long as the magnitude is positive
            if magnitudes[y, x] > 0:          
                # calculate the displacement of applying the angle and magnitude
                dx = int(magnitudes[y, x] * np.cos(np.radians(angles[y, x])))
                dy = int(magnitudes[y, x] * np.sin(np.radians(angles[y, x])))

                # draw the arrow                
                cv2.arrowedLine(retVal, (x, y), (x + dx, y + dy), color, 1, tipLength=0.5)

    return retVal

def draw_circles_on_image(image: cv2.Mat, points: np.ndarray, color: tuple[int, int, int], radius:int = 2):
    """
    Draw points on an image with a desired radius

    Args:
    - image (np.ndarray): The original grayscale image (2D).
    - points (np.ndarray): A 1D array of 2D points.
    - color (tiple[int, int, int]): The color to draw the circles with

    Returns:
    - np.ndarray: The original image with circles drawn on it indicating the points.
    """

    # copy over original image
    retVal = image.copy()
    # iterate points
    for p in points:
        # draw the point                
        cv2.circle(retVal, (p[1], p[0]), radius, color, -1)

    return retVal

def draw_voronoi_on_image(image: cv2.Mat, voronoi: Voronoi, color: tuple[int, int, int]):
    """
    Draw the voronoi tesselation of some points on an image

    Args:
    - image (np.ndarray): The original grayscale image (2D).
    - voronoi (scipy.spatial.Voronoi) The voronoi tesselation of the points
    - color (tiple[int, int, int]): The color to draw the circles with

    Returns:
    - np.ndarray: The original image with the boundaries of the tesselation drawn on it.
    """
    retVal = image.copy()
    # aggregate the lines for a single opencv call
    lines = []

    # skip the last region as it encompasses the entire image
    for region_idx in range(len(voronoi.regions) - 1):
        # extract the region
        region = voronoi.regions[region_idx]

        # skip empty/invalid regions
        if not region or -1 in region:  # skip empty or invalid regions
            continue
        
        # Have to inver the vertex coordinates because opencv is (cols, rows) while scipy is (row, cols)
        vertices = [[voronoi.vertices[i][1], voronoi.vertices[i][0]] for i in region]

        # opencv expects this shape as outlined in their documentation
        vertices = np.array(vertices, np.int32).reshape((-1, 1, 2))

        # append to our list
        lines.append(vertices)

    # Draw all Voronoi cells in one go
    cv2.polylines(retVal, lines, True, color, 1)

    return retVal            
