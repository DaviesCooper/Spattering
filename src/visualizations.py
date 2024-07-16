import numpy as np
import cv2

def draw_arrows_on_image(image, angles, magnitudes, step=20):
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

    # grab height and width from original to compute step
    height, width = image.shape[:2]

    # copy over original image and convert to bgr
    retVal = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # iterate pixels by step count
    for y in range(0, height, step):
        for x in range(0, width, step):
            
            # so long as the magnitude is positive
            if magnitudes[y, x] > 0:          
                # calculate the displacement of applying the angle and magnitude
                dx = int(magnitudes[y, x] * np.cos(np.radians(angles[y, x])))
                dy = int(magnitudes[y, x] * np.sin(np.radians(angles[y, x])))

                # draw the arrow                
                cv2.arrowedLine(retVal, (x, y), (x + dx, y + dy), (0,0,255), 1, tipLength=0.5)

    return retVal