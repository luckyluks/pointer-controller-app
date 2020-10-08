import sys
import time
import cv2
import math

import numpy as np
import logging as log

from src.model_classes.model import Model

class PoseEstimationModel(Model):
    """
    Class for the Pose Estimation Model.
    """

    def preprocess_output(self, results, image, draw_output):
        """
        Process the output and draw estimation results if applicable.
        """
        # Check the model output blob shape
        if (len(results) != 3) :
            log.error(f"Incorrect output dimensions in \"{self.__class__.__name__}\": {results.shape}")
            sys.exit(1)  

        # Process network output
        pose_angles = {
            "yaw" : results["angle_y_fc"].item() ,
            "pitch" : results["angle_p_fc"].item() ,
            "roll" : results["angle_r_fc"].item()
        }

        # Draw output, if applicable
        if draw_output:
            yaw, pitch, roll = pose_angles.values()

            yaw = (yaw * np.pi / 180)
            pitch = pitch * np.pi / 180
            roll = roll * np.pi / 180

            height, width = image.shape[:2]
            tdx = width / 2
            tdy = height / 2
            size = 1000

            # X-Axis pointing to right. drawn in red
            x1 = size * (math.cos(yaw) * math.cos(roll)) + tdx
            y1 = (
                size
                * (
                    math.cos(pitch) * math.sin(roll)
                    + math.cos(roll) * math.sin(pitch) * math.sin(yaw)
                )
                + tdy
            )

            # Y-Axis | drawn in green
            #        v
            x2 = size * (-math.cos(yaw) * math.sin(roll)) + tdx
            y2 = -(
                size
                * (
                    math.cos(pitch) * math.cos(roll)
                    - math.sin(pitch) * math.sin(yaw) * math.sin(roll)
                )
                + tdy
            )

            # Z-Axis (out of the screen) drawn in blue
            x3 = size * (math.sin(yaw)) + tdx
            y3 = size * (-math.cos(yaw) * math.sin(pitch)) + tdy

            cv2.line(image, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 3)
            cv2.line(image, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 3)
            cv2.line(image, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2)


        return image, pose_angles

    def check_input(self, image, **kwargs):
        """
        Check data input.
        """
        if not isinstance(image, np.ndarray):
            raise IOError("Image input is in the wrong format. Expected \"np.ndarray\"!")