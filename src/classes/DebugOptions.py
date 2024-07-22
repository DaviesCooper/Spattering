class DebugOptions:
    """
    Configuration options for debugging.
txt
    Attributes:
        debugDir (str): The directory path to output debug visuals to.
        consoleDebug (bool): Whether or not to print debug statements to the console.
        txtDebug (bool): Whether or not to write debug statements to a file.
        visualizeDebug (bool): Whether or not to draw intermediate visuals for debugging purposes.
    """

    def __init__(self, debugDir: str, consoleDebug = False, txtDebug = False, visualizeDebug = False):
        """
        Initializes an instance of DebugOptions.

        Args:
            debugDir (str): The directory path to output debug visuals to.
            consoleDebug (bool, optional): Whether or not to print debug statements to the console. Default is False.
            txtDebug (bool, optional): Whether or not to write debug statements to a file. Default is False.
            visualizeDebug (bool, optional): Whether or not to draw intermediate visuals for debugging purposes. Default is False.
        """
        
        # Whether or not to print debug statements to the console
        self.consoleDebug = consoleDebug

        # Whether or not to write debug statements to a file
        self.txtDebug = txtDebug

        # Whether or not to draw the intermediate visuals to see what is happening at every step
        self.visualizeDebug = visualizeDebug

        # The directory to output the debug visuals to
        self.debugDir = debugDir
    
    def __str__(self):
        return (
            f"Directory: {self.debugDir}, "
            f"Console: {self.consoleDebug}, "
            f"File: {self.txtDebug}, "
            f"Visualize: {self.visualizeDebug}.")