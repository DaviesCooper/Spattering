import numpy as np
import cv2

def draw_arrows_on_image(image, angles, magnitudes, color, step=20):
    """
    Draw arrows on an image based on angle and magnitude data. Used to represent directional vectors on an image

    Parameters:
    image (np.ndarray): The original grayscale image (2D).
    angles (np.ndarray): A 2D array of angles (in degrees).
    magnitudes (np.ndarray): A 2D array of magnitudes.
    step (int): The step size for iterating over pixels to draw arrows (default is 20).

    Returns:
    np.ndarray: The original image with arrows drawn on it indicating the flow direction and magnitude.
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


def draw_circles_on_image(image, points, color):
    """
    Draw points on an image with a desired radius

    Parameters:
    image (np.ndarray): The original grayscale image (2D).
    points (np.ndarray): A 1D array of 2D points.
    radius (int): The radius, in px, of each point

    Returns:
    np.ndarray: The original image with circles drawn on it indicating the points.
    """

    # copy over original image
    retVal = image.copy()
    # iterate points
    for p in points:
        # draw the point                
        cv2.circle(retVal, (p[1], p[0]), 2, color, -1)

    return retVal

def draw_voronoi_on_image(image, voronoi, color):
    """
    Draw the voronoi tesselation of some points on an image

    Parameters:
    image (np.ndarray): The original grayscale image (2D).
    Voronoi (scipy.spatial.Voronoi) The voronoi tesselation of the points

    Returns:
    np.ndarray: The original image with the boundaries of the tesselation drawn on it.
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
