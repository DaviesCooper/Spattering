from src.classes import DebugOptions, AbstractStippleGenerator
from src.utils import draw_arrows_on_image, draw_circles_on_image, draw_voronoi_on_image, polygon_centroid, left_pad
from scipy.spatial import Voronoi
import numpy as np
import os
import cv2


class PreprocessingStippleGenerator(AbstractStippleGenerator):
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
    def __init__(self, image: cv2.Mat, numPoints: int, dpUnit: int, pointUnitRadius: int, preprocessWindowSize: int, relaxationIterations: int, debugOptions: DebugOptions):
        """
        Initializes an instance of AbstractStippleGenerator.

        Args:
            - image (cv2.Mat): The image data used for stipple generation.
            - dpu (int): Dots per (arbitrary) unit. Used to have arbitrary precision in the final svg.
            - ppu (int): Point radius in (arbitrary) units.
            - debugOptions (DebugOptions): Debug options for debugging purposes.
        """
        self.windowSize = preprocessWindowSize
        self.iterations = relaxationIterations
        super().__init__(image, numPoints, dpUnit, pointUnitRadius, debugOptions)
        self.debugString(F"Window Size: {self.windowSize}.")
#endregion

#region Public Methods
    def stipple(self):
        angles, magnitudes = self._preprocess()
        points = self._generatePoints()
        relaxed_points = self._relax_points(angles, magnitudes, points)
        self.result = self._postprocess(relaxed_points)
        
    def exportToSVG(self, outputPath: str):
        pass

    def exportToPNG(self, outputPath: str):
        pass
#endregion

#region Private Methods

    def _preprocess(self):
        """Private Method
        
        Performs the preprocessing by generating the "flow chart"
         Returns:
        - angles (np.ndarray): Array of angles corresponding to each pixel.
        - magnitudes (np.ndarray): Array of magnitudes corresponding to each pixel.
        """
        self.debugString("Preprocessing...")
        
        # Generate magnitude and angle maps from the image.
        angles, magnitudes = self._generate_magnitude_and_angle_maps()
        
        # Create a color version of the original image to draw arrows on.
        arrow_plot = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        
        # Draw arrows on the color image representing angles and magnitudes.
        arrow_plot = draw_arrows_on_image(arrow_plot, angles, magnitudes, (0, 0, 255))
        
        # Visualize the arrow plot.
        self.visualizeImage(arrow_plot, "arrow_plot.png")
        
        # Convert the angle and magnitude maps to HSV color space and visualize.
        hsv_image = cv2.cvtColor(self._angle_magnitude_to_hsv(angles, magnitudes), cv2.COLOR_HSV2BGR)
        self.visualizeImage(hsv_image, "hsv_plot.png")
        
        # Log the completion of preprocessing.
        self.debugString("Preprocessing complete.")
        
        return angles, magnitudes

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

    def _relax_points(self, angles:np.array, magnitudes:np.array, points:np.array):
        """Private Method

        Relaxes the given points by iteratively adjusting their positions based on Voronoi diagram centroids and magnitudes/angles.

        Args:
            - angles (np.ndarray): Array of angles associated with each point.
            - magnitudes (np.ndarray): Array of magnitudes associated with each point.
            - points (np.ndarray): Array of initial points to be relaxed.

        Returns:
            - retVal (np.ndarray): Array of refined points after relaxation.
        """

        retVal = points
        self.debugString("Relaxing...")
        for i in range(self.iterations):
            # Log progress at every 10% of iterations.
            if (np.round(i / self.iterations) * 100) % 10 <= 1:
                self.debugString(f"{np.round(i / self.iterations * 100)}%...")
            
            # Compute the Voronoi diagram for the current points.
            voronoi = Voronoi(retVal)
            
            # Update each point's position based on the centroid of its Voronoi region.
            for point_index in range(len(retVal)):
                y, x = retVal[point_index]
                if(self.image[y,x] == 255):
                    continue
                region_index = voronoi.point_region[point_index]
                region = voronoi.regions[region_index]
                
                # Skip if the region is invalid.
                if not region or -1 in region:
                    continue
                
                # Calculate the centroid of the Voronoi region.
                vertices = [voronoi.vertices[i] for i in region]
                centroid = polygon_centroid(vertices)
                
                newx = centroid[1]
                newy = centroid[0]
                newx += x
                newy += y
                divisor = 2

                # Adjust the point position using magnitudes and angles if the point is not on a specific value.
                if self.image[y, x] != 0:
                    newx += x.astype(np.int32) + int(magnitudes[y, x] * np.cos(np.radians(angles[y, x])))
                    newy += y.astype(np.int32) + int(magnitudes[y, x] * np.sin(np.radians(angles[y, x])))
                    divisor = 3

                # Average the new position and clip to image boundaries.
                newx /= divisor
                newy /= divisor
                retVal[point_index] = np.array([np.clip(newy, 0, self.image.shape[0] - 1), np.clip(newx, 0, self.image.shape[1] - 1)])
            
            # Visualize the Voronoi diagram and updated points for the current iteration.
            pre_image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
            points_image = draw_circles_on_image(pre_image, retVal, (255, 0, 0))
            voronoi_diagram = draw_voronoi_on_image(points_image, voronoi, (0, 255, 0))
            fileName = left_pad(f"{i}", 6, "0")
            self.visualizeImage(voronoi_diagram, os.path.join(self.iterationDirectoryName, f"relaxed{fileName}.png"))
        
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

#region Helper Methods

    def _angle_magnitude_to_hsv(self, angles, magnitudes):
        """Private Method
        
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

    def _displacement_to_angle_and_magnitude(self, y, x):
        """Private Method

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
        
    def _generate_magnitude_and_angle_maps(self):
        """Private Method

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
        height, width = self.image.shape

        # need the center so all white pixels can point in that direction
        center_x, center_y = width / 2, height / 2

        # Gaussian blur the original image
        smoothed_heightmap = cv2.GaussianBlur(self.image, ksize=(3,3), sigmaX=1)
        
        angles = np.zeros((self.image.shape[0], self.image.shape[1]), np.float64)
        magnitudes = np.zeros((self.image.shape[0], self.image.shape[1]), np.float64)

        # iterate over each pixel
        for pixel_X in range(smoothed_heightmap.shape[1]):
            for pixel_Y in range(smoothed_heightmap.shape[0]):

                # if the pixel is fully white, point towards the center
                if smoothed_heightmap[pixel_Y, pixel_X] == 255:
                    # get the angle and magnitude to the center of the image, and set the angle and momentum of the flow chart
                    angle, magnitude = self._displacement_to_angle_and_magnitude(pixel_Y - center_y, pixel_X - center_x)
                    angles[pixel_Y, pixel_X] = angle
                    magnitudes[pixel_Y, pixel_X] = 10

                # if the pixel is fully black, point nowhere
                elif smoothed_heightmap[pixel_Y, pixel_X] == 0:
                    angles[pixel_Y, pixel_X] = 0
                    magnitudes[pixel_Y, pixel_X] = 0

                # otherwise
                else:
                    # find the kernel window in the image
                    roi = smoothed_heightmap[pixel_Y - self.windowSize:pixel_Y + self.windowSize +1, pixel_X - self.windowSize:pixel_X + self.windowSize + 1]
                    # find the location of the darkest pixel
                    _, _, min_loc, _ = cv2.minMaxLoc(roi)
                    # convert the location back to absolute index
                    x, y = min_loc - np.array([self.windowSize,self.windowSize])
                    # get the angle and magnitude to the darkest pixel of the kernel window, and set the angle and momentum of the flow chart
                    angle, magnitude = self._displacement_to_angle_and_magnitude(y, x)
                    angles[pixel_Y, pixel_X] = angle
                    magnitudes[pixel_Y, pixel_X] = magnitude

        return (angles, magnitudes)

#endregion
