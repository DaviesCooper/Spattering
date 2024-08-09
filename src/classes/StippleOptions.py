class StippleOptions:
    """
    Configuration options for stippling.


    - image (cv2.Mat): The image data used for stipple generation.
        - numPoints (int): The number of points to attempt to stipple with.
        - iters (int): The number of iterations to optimize stippling with.
        - dpUnit (unit): Dots per unit. This is not really inches, it is in whatever units are assumed by the user.
            Used so that scale is arbitrary for the svg.
        - pointUnitRadius (int): Point unit radius. It is in whatever units are assumed by the user.
        - debugOptions (DebugOptions): Debug options for debugging purposes.
    """

    def __init__(self,):
        # The image to use for stippling
        self.image = image

        # The number of points to attempt to stipple with
        self.numPoints = numPoints

        # The dots per (arbitrary) unit. Used to have arbitrary precision in the final svg.
        self.dpi = dpi

        # The point radius in (arbitrary) units.
        self.pointUnitRadius = pointUnitRadius

        # The number of iterations to perform during stippling
        self.iters = iters

        # The pixel radius of each point.
        self.rad = int(np.round(self.pointUnitRadius * self.dpi))
    
    def __str__(self):
        return (
            f"Directory: {self.debugDir}, "
            f"Console: {self.consoleDebug}, "
            f"File: {self.txtDebug}, "
            f"Visualize: {self.visualizeDebug}.")