from abc import ABC, abstractmethod
import cv2
import os
import datetime
import numpy as np
import xml.etree.cElementTree as ET
from src.utils import ensure_empty_directory, images_to_video
from src.classes import DebugOptions

class AbstractStippleGenerator(ABC):
    """
    Abstract base class for stipple generators.

    Attributes:
        - image (cv2.Mat): The image data used for stipple generation.
        - numPoints (int): The number of points to attempt to stipple with.
        - iters (int): The number of iterations to optimize stippling with.
        - dpUnit (unit): Dots per unit. This is not really inches, it is in whatever units are assumed by the user.
                    Used so that scale is arbitrary for the svg.
        - pointUnitRadius (int): Point unit radius. It is in whatever units are assumed by the user.
        - debugOptions (DebugOptions): Debug options for debugging purposes.
    """

#region CONSTANTS
    debugFileName = "Debug.txt"
    iterationDirectoryName = "iterations"
#endregion

#region Abstract Methods

    @abstractmethod
    def __init__(self, image: cv2.Mat, numPoints: int, dpi: int, pointUnitRadius: int, debugOptions: DebugOptions = None):
        """
        Initializes an instance of AbstractStippleGenerator.

        Args:
            - image (cv2.Mat): The image data used for stipple generation.
            - dpu (int): Dots per (arbitrary) unit. Used to have arbitrary precision in the final svg.
            - ppu (int): Point radius in (arbitrary) units.
            - debugOptions (DebugOptions): Debug options for debugging purposes.
        """

        # The image to use for stippling
        self.image = image

        # The number of points to attempt to stipple with
        self.numPoints = numPoints

        # The dots per (arbitrary) unit. Used to have arbitrary precision in the final svg.
        self.dpi = dpi

        # The point radius in (arbitrary) units.
        self.pointUnitRadius = pointUnitRadius

        # The pixel radius of each point.
        self.rad = int(np.round(self.pointUnitRadius * self.dpi))

        # The debug options for debugging purposes.
        self.debugOptions = debugOptions

        # Make sure the debug directory is exists, and is empty
        if(self.debugOptions):
            ensure_empty_directory(self.debugOptions.debugDir)
            os.makedirs(os.path.join(self.debugOptions.debugDir, self.iterationDirectoryName))
            if(self.debugOptions.txtDebug):
                with open(os.path.join(self.debugOptions.debugDir, self.debugFileName), 'w') as debug_file:
                    debug_file.write(f"{datetime.datetime.now()}: Created\n")


        self.debugString(self.__str__())
        super().__init__()

    @abstractmethod
    def stipple(self):
        """
        Abstract method to generate the weighted Voronoi stipple pattern of the image given to the constructor.
        """
        pass

    def exportToSVG(self, outputPath: str):
        """
        Abstract method to export generated stipple patterns to SVG format.

        Args:
            - points (np.array): The points to export to svg
            - outputPath (str): The path where the SVG file will be saved.
        """
        radius = self.pointUnitRadius * self.dpi
        xml_declaration = '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
        postSort = sorted(self.result, key=lambda point: point[0]**2 + point[1]**2)

        root = ET.Element("svg", 
                          width=f"{self.image.shape[1]}", 
                          height=f"{self.image.shape[0]}", 
                          viewbox=f"{0} {0} {self.image.shape[1]} {self.image.shape[0]}",
                          version="1.1",
                          id="svg1")
        layer = ET.SubElement(root, "g", id="layer1")
        for idx, p in enumerate(postSort):
            ET.SubElement(layer, "circle", 
                          style='vector-effect:non-scaling-stroke;fill:none;fill-opacity:1;stroke:#000000;stroke-width:0.264583;stroke-dasharray:none;stroke-opacity:1;-inkscape-stroke:hairline',
                          id=f"Point{idx}",
                          cx=f"{p[1]}",
                          cy=f"{p[0]}",
                          r=f"{radius}"
                          )
        
        with open(outputPath, 'wb') as f:
            f.write(xml_declaration.encode('utf-8'))  # Write the XML declaration
            tree = ET.ElementTree(root)
            tree.write(f, encoding='utf-8', xml_declaration=False)
        
        

#endregion

#region Public Methods

    def debugString(self, string: str):
        """
        Logs debug information to console and/or text file based on debug options.

        Args:
            - string (str): Debug message to log.
        """
        if self.debugOptions:
            if self.debugOptions.consoleDebug:
                print(string)
            if self.debugOptions.txtDebug:
                with open(os.path.join(self.debugOptions.debugDir, self.debugFileName), 'a') as debug_file:
                    debug_file.write(f"{datetime.datetime.now()}: {string}\n")

    def visualizeImage(self, image: cv2.Mat, name: str):
        """
        Visualizes an image to a specified path if visualization debug option is enabled.

        Args:
            - image (cv2.Mat): Image data to visualize.
            - path (str): Path where the visualization image will be saved.
        """
        if self.debugOptions:
            if self.debugOptions.visualizeDebug:
                cv2.imwrite(os.path.join(self.debugOptions.debugDir, name), image)

    def visualizeVideo(self):
        if self.debugOptions:
            if self.debugOptions.visualizeDebug:
                images_to_video(os.path.join(self.debugOptions.debugDir, self.iterationDirectoryName), os.path.join(self.debugOptions.debugDir, "relaxing.mp4"))

#endregion

#region Private Methods

    def _genRandomPointsOnBlackUniformly(self):
        """Private method
        
        Generates random points using numPoints in a uniform distribution across the image only where the pixel intensity is less than 255.
        Non-deterministic runtime since it will retry creating a point if one is created on an white pixel until all numPoints have been created.
        """

        coordinates = []
        while len(coordinates) < self.numPoints:
            x = np.random.randint(0, self.image.shape[1])
            y = np.random.randint(0, self.image.shape[0])
            if self.image[y,x] != 255:
                coordinates.append(np.array([y, x]))
        return np.array(coordinates)

    def _genRandomPointsUniformly(self):
        """Private method

        Generates random points using numPoints in a uniform distribution across the image.
        """

        return (np.random.rand(self.numPoints, 2) * np.array([self.image.shape[0] - 1, self.image.shape[1]-1])).astype(np.uint16)  

#endregion

#region Override Methods

    def __str__(self):
        return (
            f"Stipple Generator\n\n"
            f"Image Shape: {self.image.shape}\n"
            f"Number of Points: {self.numPoints}\n"
            f"Dots per unit: {self.dpi}\n"
            f"Point radius in units: {self.pointUnitRadius}\n"
            f"Point radius in pixels: {self.rad}\n"
            f"Debug Options: {self.debugOptions.__str__()}"
        )

#endregion
