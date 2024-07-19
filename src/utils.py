import numpy as np

# Function to calculate the signed area of the polygon
def polygon_area(vertices):
    """
    Calculates the area of a polygon

    Parameters:
    vertices (np.ndarray): An array of (y, x) coordinates representing the vertices of a polygon

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

    Parameters:
    vertices (np.ndarray): An array of (y, x) coordinates representing the vertices of a polygon

    Returns:
    np.ndarray: An array containing two elements representing the centroid in the form [y, x]
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