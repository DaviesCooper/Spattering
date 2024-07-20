import numpy as np
import cv2

def angle_magnitude_to_hsv(angles, magnitudes):
    """
    Convert angle and magnitude information to an HSV image.

    Parameters:
    angles (np.ndarray): A 2D array of angles (in degrees) where each value represents the hue.
    magnitudes (np.ndarray): A 2D array of magnitudes corresponding to the brightness value in the HSV image.

    Returns:
    np.ndarray: A 3D array representing the HSV image, where H is scaled to [0, 179],
                S is set to 255, and V is normalized based on the magnitudes.
    """

    # create blank image to write to
    hsv_image = np.zeros((angles.shape[0], angles.shape[1], 3), dtype=np.uint8)

    # convert hues to opencv range (0-179)
    hues = (angles / 360) * 179

    # set the hues of the image
    hsv_image[..., 0] = hues.astype(np.uint8)

    # set the saturation to always be 255
    hsv_image[..., 1] = 255

    # normalize the magnitudes to the greatest magnitude found in the image
    hsv_image[..., 2] = (magnitudes / (magnitudes.max() + 1e-10) * 255).astype(np.uint8)
    
    return hsv_image

def displacement_to_angle_and_magnitude(y, x):
    """
    Calculate the angle and magnitude from displacement coordinates.

    Parameters:
    y (float): The vertical displacement.
    x (float): The horizontal displacement.

    Returns:
    tuple: A tuple containing:
        - angle (float): The angle in degrees, normalized to [0, 360).
        - magnitude (float): The magnitude calculated as the Euclidean distance.
    """

    # angle is the arctan2 of the rise and run
    angle = np.arctan2(y, x)
    # make sure the angle is between 0 - 360
    angle = angle % (2 * np.pi)  * (180 / np.pi)
    # calculate the magnitude
    magnitude = np.sqrt(x ** 2 + y **2)
    return (angle, magnitude)
    
def generate_magnitude_and_angle_maps(image, window_size = 5):
    """
    Generate angle and magnitude maps from a binary image.

    Parameters:
    image (np.ndarray): A 2D binary image (e.g., a heightmap) where white pixels indicate points of interest.
    window_size (int): The size of the kernel used to find the darkest pixel around each pixel (default is 5).

    Returns:
    tuple: A tuple containing:
        - angles (np.ndarray): A 2D array of angles (in degrees) for each pixel.
        - magnitudes (np.ndarray): A 2D array of magnitudes corresponding to the flow direction for each pixel.
    """

    # need the width and height for iterator
    height, width = image.shape

    # need the center so all white pixels can point in that direction
    center_x, center_y = width / 2, height / 2

    # Gaussian blur the original image
    smoothed_heightmap = cv2.GaussianBlur(image, ksize=(3,3), sigmaX=1)
    
    angles = np.zeros((image.shape[0], image.shape[1]), np.float64)
    magnitudes = np.zeros((image.shape[0], image.shape[1]), np.float64)

    # iterate over each pixel
    for pixel_X in range(smoothed_heightmap.shape[1]):
        for pixel_Y in range(smoothed_heightmap.shape[0]):

            # if the pixel is fully white, point towards the center
            if smoothed_heightmap[pixel_Y, pixel_X] == 255:
                # get the angle and magnitude to the center of the image, and set the angle and momentum of the flow chart
                angle, magnitude = displacement_to_angle_and_magnitude(pixel_Y - center_y, pixel_X - center_x)
                angles[pixel_Y, pixel_X] = angle
                magnitudes[pixel_Y, pixel_X] = 10

            # if the pixel is fully black, point nowhere
            elif smoothed_heightmap[pixel_Y, pixel_X] == 0:
                angles[pixel_Y, pixel_X] = 0
                magnitudes[pixel_Y, pixel_X] = 0

            # otherwise
            else:
                # find the kernel window in the image
                roi = smoothed_heightmap[pixel_Y - window_size:pixel_Y + window_size +1, pixel_X - window_size:pixel_X + window_size + 1]
                # find the location of the darkest pixel
                _, _, min_loc, _ = cv2.minMaxLoc(roi)
                # convert the location back to absolute index
                x, y = min_loc - np.array([window_size,window_size])
                # get the angle and magnitude to the darkest pixel of the kernel window, and set the angle and momentum of the flow chart
                angle, magnitude = displacement_to_angle_and_magnitude(y, x)
                angles[pixel_Y, pixel_X] = angle
                magnitudes[pixel_Y, pixel_X] = magnitude

    return (angles, magnitudes)
