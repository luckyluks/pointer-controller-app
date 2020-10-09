import sys
import time
import cv2

import logging as log
import numpy as np

from src.model_classes.model import Model


class GazeEstimationModel(Model):
    """
    Class for the Gaze Estimation Model.
    """

    def preprocess_output(self, results, image, draw_output, **kwargs):
        """
        Process the output and draw estimation results if applicable.
        """
        # Check the model output blob shape
        if not ("gaze_vector" in results.keys()) and (len(results[list(results.keys())[0]][0]) != 3) :
            log.error(f"Incorrect output dimensions in \"{self.__class__.__name__}\": {results.shape}")
            sys.exit(1)  

        # Unpack output
        gaze_vector = dict(zip(["x", "y", "z"], np.vstack(results["gaze_vector"]).ravel()))

        # Draw output, if applicable
        if draw_output:
            left_eye_point = (kwargs["landmarks"]["left_eye_x"], kwargs["landmarks"]["left_eye_y"])
            right_eye_point = (kwargs["landmarks"]["right_eye_x"], kwargs["landmarks"]["right_eye_y"])
            cv2.arrowedLine(
                image,
                (left_eye_point[0], left_eye_point[1]),
                (
                    left_eye_point[0] + int(gaze_vector["x"] * 100),
                    left_eye_point[1] - int(gaze_vector["y"] * 100),
                ),
                
                color=(0, 0, 255),
                thickness=2,
                tipLength=0.2,
            )
            cv2.arrowedLine(
                image,
                (right_eye_point[0], right_eye_point[1]),
                (
                    right_eye_point[0] + int(gaze_vector["x"] * 100),
                    right_eye_point[1] - int(gaze_vector["y"] * 100),
                ),
                
                color=(0, 0, 255),
                thickness=2,
                tipLength=0.2,
            )
        
        return image, gaze_vector


    def check_input(self, image, **kwargs):
        """
        Check data input.
        """
        if not isinstance(image, np.ndarray):
            raise IOError("Image input is in the wrong format. Expected \"np.ndarray\"!")

        if not all(any(x == y for y in kwargs.keys()) for x in ["landmarks", "pose"]):
            raise NotImplementedError(f"Incorrect input for \"{self.__class__.__name__}\"!")