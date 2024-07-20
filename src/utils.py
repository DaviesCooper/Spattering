import numpy as np
import os
import cv2

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


def left_pad(string, length, char):
    toPad = length - len(string)
    if(toPad > 0):
        return str(char)*toPad + string
    return string


def images_to_video(image_folder, video_name):
    images = os.listdir(image_folder)
    images.sort()
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v', 'avc1', 'h264', or 'x264'
    video = cv2.VideoWriter(video_name, fourcc, 10, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

def delete_files_in_directory(directory):
    # Get list of all files in the directory
    file_list = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    # Iterate over the file list and delete each file
    for f in file_list:
        file_path = os.path.join(directory, f)
        try:
            os.remove(file_path)
            print(f"Deleted {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")