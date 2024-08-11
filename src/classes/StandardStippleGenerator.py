from src.classes import DebugOptions, AbstractStippleGenerator
from src.utils import draw_circles_on_image, left_pad
from scipy.spatial import cKDTree
import numpy as np
import os
import cv2


class StandardStippleGenerator(AbstractStippleGenerator):
    """Class for generating weighted voronoi stippling of an image using preprocessing.

    A "flow chart" is created as a preprocessing step which is used to weight the
    voronoi stippling rather than 

    Attributes:
        - image (cv2.Mat): The image data used for stipple generation.
        - numPoints (int): The number of points to try to populate in the image. 
        - dpUnit (int): Dots per unit. This is not really inches, it is in whatever units are assumed by the user.
                    Used so that scale is arbitrary for the svg.
        - pointUnitRadius (int): Point unit radius. It is in whatever units are assumed by the user.
        - debug (DebugOptions): Debug options for debugging purposes.
    """

#region Constructors
    def __init__(self, image: cv2.Mat, numPoints: int, dpUnit: int, pointUnitRadius: int, relaxationIterations: int, debugOptions: DebugOptions):
        """
        Initializes an instance of AbstractStippleGenerator.

        Args:
            - image (cv2.Mat): The image data used for stipple generation.
            - dpu (int): Dots per (arbitrary) unit. Used to have arbitrary precision in the final svg.
            - ppu (int): Point radius in (arbitrary) units.
            - debugOptions (DebugOptions): Debug options for debugging purposes.
        """
        self.iterations = relaxationIterations
        super().__init__(image, numPoints, dpUnit, pointUnitRadius, debugOptions)
#endregion

#region Public Methods
    def stipple(self):
        points = self._generatePoints()
        relaxed_points = self._relax_points(points)
        self.result = self._postprocess(relaxed_points)
#endregion

#region Private Methods

    def _generatePoints(self):
        """Private Method

        Generates random points covering the image

        Returns:
            - points (list): List of generated points.
        """

        # Generate random points uniformly distributed on a black background.
        points = self._genRandomPointsOnBlackUniformly()
        
        # Create a blank white image with the same shape as the original image.
        blank = np.ones(self.image.shape, np.uint8) * 255
        
        # Draw the generated points on the white image.
        points_on_white = draw_circles_on_image(blank, points, (0,0,0), self.rad)
        
        # Visualize the image with points drawn on the white background.
        self.visualizeImage(points_on_white, "initial_points.png")
        
        # Draw the generated points on the original image.
        points_image = draw_circles_on_image(self.image, points, (0, 0, 0), self.rad)
        
        # Visualize the original image with points overlaid.
        self.visualizeImage(points_image, "initial_points_overlayed.png")
        
        # Log the number of generated points.
        self.debugString(f"Generated {len(points)}.")

        return points

    def _relax_points(self, points:np.array):
        retVal = points.copy()

        for iteration in range(self.iterations):

            if (np.round(iteration / self.iterations) * 100) % 10 <= 1:
                self.debugString(f"{np.round(iteration / self.iterations * 100)}%...")

            tree= cKDTree(retVal)
            weights = np.zeros(len(retVal))
            centroids = np.zeros((len(retVal), 2))

            for y in range(self.image.shape[0]):
                for x in range(self.image.shape[1]):
                    weight = 1 - (self.image[y,x] / 255)
                    dist, index = tree.query(np.array([y, x]))
                    weights[index] += weight
                    centroids[index][0] += y * weight
                    centroids[index][1] += x * weight

            for index in range(len(weights)):
                if(weights[index]) == 0:
                    centroids[index] = retVal[index]
                else:
                    centroids[index] /= weights[index]
            
            retVal =  np.floor(retVal + .25 * (centroids - retVal)).astype(int)
        
            # Visualize the Voronoi diagram and updated points for the current iteration.
            blank = np.ones(self.image.shape, np.uint8) * 255
            points_image = draw_circles_on_image(blank, retVal, (0, 0, 0), self.rad)
            fileName = left_pad(f"{iteration}", 6, "0")
            self.visualizeImage(points_image, os.path.join(self.iterationDirectoryName, f"relaxed{fileName}.png"))
        
        # Final visualization of the relaxed points on white and original backgrounds.
        blank = np.ones(self.image.shape, np.uint8) * 255
        points_on_white = draw_circles_on_image(blank, retVal, (0, 0, 0), self.rad)
        self.visualizeImage(points_on_white, "relaxed_points.png")
        points_image = draw_circles_on_image(self.image, retVal, (0, 0, 0), self.rad)
        self.visualizeImage(points_image, "relaxed_points_overlayed.png")
        self.visualizeVideo()
        
        self.debugString("Relaxation complete.")
        
        return retVal

    def _postprocess(self, relaxed_points:np.array):
        """Private Method

        Post-processes the relaxed points by filtering out points that fall on white areas of the image, and visualizes the results.

        The function performs the following steps:
        1. Filters out points that are located on white areas (value 255) of the image.
        2. Visualizes the filtered points on a white background and overlays them on the original image.
        3. Logs the start and completion of the post-processing.

        Args:
            - relaxed_points (np.ndarray): Array of points that have been relaxed.

        Returns:
            - retVal (np.ndarray): Array of filtered points, excluding those on white areas.
        """

        self.debugString("Postprocessing...")
        # Initialize an empty list to store points that are not on white areas.
        retVal = []
        
        # Filter out points located on white areas of the image.
        for point in relaxed_points:
            if self.image[point[0], point[1]] == 255:
                continue
            retVal.append(point)

        # Convert the list of filtered points to a NumPy array.
        retVal = np.array(retVal)
        
        # Create a blank white image with the same shape as the original image.
        blank = np.ones(self.image.shape, np.uint8) * 255
        
        # Draw the filtered points on the white image.
        points_on_white = draw_circles_on_image(blank, retVal, (0, 0, 0), self.rad)
        self.visualizeImage(points_on_white, "post_processed_points.png")
        
        # Draw the filtered points on the original image.
        points_image = draw_circles_on_image(self.image, retVal, (0, 0, 0), self.rad)
        self.visualizeImage(points_image, "post_processed_points_overlayed.png")
        
        # Log the completion of the post-processing.
        self.debugString("Postprocessing complete.")

        return retVal

#endregion
