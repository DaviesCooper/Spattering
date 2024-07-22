from src.classes import DebugOptions, AbstractStippleGenerator
import numpy as np
import cv2

"""Class for generating weighted voronoi stippling of an image using preprocessing.

    A "flow chart" is created as a preprocessing step which is used to weight the
    voronoi stippling rather than 

    Attributes:
        image (cv2.Mat): The image data used for stipple generation.
        dpi (int): Dots per inch (dpi) unit. This is not really inches, it is in whatever units are assumed by the user.
                    Used so that scale is arbitrary for the svg.
        rad (int): Point unit radius. It is in whatever units are assumed by the user.
        debug (DebugOptions): Debug options for debugging purposes.
    """
class PreprocessingStippleGenerator(AbstractStippleGenerator):

    def __init__(self, image: cv2.Mat, dpUnit: int, pointUnitRadius: int, debugOptions: DebugOptions):
        super().__init__(self, image, dpUnit, pointUnitRadius, debugOptions)

    def stipple(self):
        pass

    def exportToSVG(self, outputPath: str):
        pass

    def exportToPNG(self, outputPath: str):
        pass

    def _preprocess(self):
        """ Private Method
        
        Performs the preprocessing by generating the "flow chart"
        """
        angles, magnitudes = self._generate_magnitude_and_angle_maps(self.image, 5)
        
        arrow_plot = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        arrow_plot = draw_arrows_on_image(arrow_plot, angles, magnitudes, (0,0,255))
        cv2.imwrite(os.path.join(temp_folder, "arrow_plot.png"), arrow_plot)
        hsv_image = cv2.cvtColor(angle_magnitude_to_hsv(angles, magnitudes), cv2.COLOR_HSV2BGR)
        cv2.imwrite(os.path.join(temp_folder, "hsv_plot.png"), hsv_image)
        
        return angles, magnitudes 

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
        
    def _generate_magnitude_and_angle_maps(self, image, window_size = 5):
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
                    roi = smoothed_heightmap[pixel_Y - window_size:pixel_Y + window_size +1, pixel_X - window_size:pixel_X + window_size + 1]
                    # find the location of the darkest pixel
                    _, _, min_loc, _ = cv2.minMaxLoc(roi)
                    # convert the location back to absolute index
                    x, y = min_loc - np.array([window_size,window_size])
                    # get the angle and magnitude to the darkest pixel of the kernel window, and set the angle and momentum of the flow chart
                    angle, magnitude = self._displacement_to_angle_and_magnitude(y, x)
                    angles[pixel_Y, pixel_X] = angle
                    magnitudes[pixel_Y, pixel_X] = magnitude

        return (angles, magnitudes)
    
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

