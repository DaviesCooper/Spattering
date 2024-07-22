from abc import ABC, abstractmethod
import cv2
import os
import datetime
import numpy as np
from src.utils import ensure_empty_directory
from classes import DebugOptions

class AbstractStippleGenerator(ABC):
    """
    Abstract base class for stipple generators.

    Attributes:
        - image (cv2.Mat): The image data used for stipple generation.
        - numPoints (int): The number of points to attempt to stipple with.
        - iters (int): The number of iterations to optimize stippling with.
        - dpu (int): Dots per (arbitrary) unit. It is in whatever units are assumed by the user.
                    Used so that scale is arbitrary for the svg.
        - pru (int): Point radius in (arbitrary) units.
        - rad (int): The pixel radius of the points as determined by the dpu and the pru. (pru * dpu)
        - debugOptions (DebugOptions): Debug options for debugging purposes.
    """

    debugFileName = "Debug.txt"

    @abstractmethod
    def __init__(self, image: cv2.Mat, numPoints: int, iters:int, dpu: int, pru: int, debugOptions: DebugOptions = None):
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
        self.dpu = dpu

        # The point radius in (arbitrary) units.
        self.pru = pru

        # The number of iterations to perform during stippling
        self.iters = iters

        # The pixel radius of each point.
        self.rad = np.floor(self.pru * self.dpu)

        # The debug options for debugging purposes.
        self.debugOptions = debugOptions

        # Make sure the debug directory is exists, and is empty
        if(self.debugOptions):
            ensure_empty_directory(self.debugOptions.debugDir)
            if(self.debugOptions.txtDebug):
                with open(os.path.join(self.debugOptions.debugDir, self.debugFileName), 'w') as debug_file:
                    debug_file.write(f"{datetime.datetime.now()}: Created\n")


        self.debugString(self.__str__())
        super().__init__()

    def __str__(self):
        return (
            f"Stipple Generator\n\n"
            f"Image Shape: {self.image.shape}\n"
            f"Number of Points: {self.numPoints}\n"
            f"Iterations: {self.iters}\n"
            f"Dots per unit: {self.dpu}\n"
            f"Point radius in units: {self.pru}\n"
            f"Point pixel radius: {self.rad}\n"
            f"Debug Options: {self.debugOptions.__str__()}"
        )

    @abstractmethod
    def stipple(self):
        """
        Abstract method to generate the weighted Voronoi stipple pattern of the image given to the constructor.
        """
        pass

    @abstractmethod
    def exportToSVG(self, outputPath: str):
        """
        Abstract method to export generated stipple patterns to SVG format.

        Args:
            - outputPath (str): The path where the SVG file will be saved.
        """
        pass

    @abstractmethod
    def exportToPNG(self, outputPath: str):
        """
        Abstract method to export generated stipple patterns to PNG format.

        Args:
            - outputPath (str): The path where the PNG file will be saved.
        """
        pass

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

    def visualizeImage(self, image: cv2.Mat, path: str):
        """
        Visualizes an image to a specified path if visualization debug option is enabled.

        Args:
            - image (cv2.Mat): Image data to visualize.
            - path (str): Path where the visualization image will be saved.
        """
        if self.debugOptions:
            if self.debugOptions.visualizeDebug:
                cv2.imwrite(path, image)
