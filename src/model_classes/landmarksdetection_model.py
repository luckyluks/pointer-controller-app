import sys
import time
import cv2

import logging as log
import numpy as np

from src.model_classes.model import Model


class LandmarksDetectionModel(Model):
    """
    Class for the Facial Landmarks Detection Model.
    """

    def preprocess_output(self, results, image, draw_output):
        """
        Process the output and draw estimation results if applicable.
        """
        # Check the model output blob shape
        if not ("95" in results.keys()) and (len(results[list(results.keys())[0]][0]) != 10) :
            log.error(f"Incorrect output dimensions in \"{self.__class__.__name__}\": {results.shape}")
            sys.exit(1)

        # Look up frame shapes
        width = image.shape[1]
        height = image.shape[0]

        # Process network output
        clean_output = results["95"][0].flatten()
        face_landmarks = {
            "left_eye_x" : int(clean_output[0] * width),
            "left_eye_y" : int(clean_output[1] * height),
            "right_eye_x" : int(clean_output[2] * width),
            "right_eye_y" : int(clean_output[3] * height),
            # "nose_x" : int(clean_output[4] * width),
            # "nose_y" : int(clean_output[5] * height),
            # "left_lip_x" : int(clean_output[6] * width),
            # "left_lip_y" : int(clean_output[7] * height),
            # "rigth_lip_x" : int(clean_output[8] * width),
            # "rigth_lip_y" : int(clean_output[9] * height)
        }

        # TODO: globalize?
        eye_radius = 15

        # Create quadratic window around the eye landmark (left)
        left_eye_xmin = face_landmarks["left_eye_x"] - eye_radius
        left_eye_xmax = face_landmarks["left_eye_x"] + eye_radius
        left_eye_ymin = face_landmarks["left_eye_y"] - eye_radius
        left_eye_ymax = face_landmarks["left_eye_y"] + eye_radius

        # Create quadratic window around the eye landmark (left)
        right_eye_xmin = face_landmarks["right_eye_x"] - eye_radius
        right_eye_xmax = face_landmarks["right_eye_x"] + eye_radius
        right_eye_ymin = face_landmarks["right_eye_y"] - eye_radius
        right_eye_ymax = face_landmarks["right_eye_y"] + eye_radius

        # Add cropped eye images to the landmarks dict, to handle output compact
        face_landmarks["left_eye_image"]= image[
                left_eye_ymin:left_eye_ymax, left_eye_xmin:left_eye_xmax
        ]
        face_landmarks["right_eye_image"]= image[
                right_eye_ymin:right_eye_ymax, right_eye_xmin:right_eye_xmax
        ]

        if face_landmarks["right_eye_image"].shape != (eye_radius*2, eye_radius*2, 3):
            log.warn(f"right eye reduced shape: {face_landmarks['right_eye_image'].shape}")

        if face_landmarks["left_eye_image"].shape != (eye_radius*2, eye_radius*2, 3):
            log.warn(f"left eye reduced shape: {face_landmarks['left_eye_image'].shape}")

        if face_landmarks["left_eye_image"].shape == (eye_radius*2, 0, 3):
            cv2.imwrite("face.png", image)

        # Draw output, if applicable
        if draw_output:
            # Draw circels around the eyes
            cv2.circle(image, (face_landmarks["left_eye_x"], face_landmarks["left_eye_y"]), radius=20, color=(0, 0, 255), thickness=2)
            cv2.circle(image, (face_landmarks["right_eye_x"], face_landmarks["right_eye_y"]), radius=20, color=(0, 0, 255), thickness=2)

        return image, face_landmarks


    def check_input(self, image, **kwargs):
        """
        Check data input.
        """
        if not isinstance(image, np.ndarray):
            raise IOError("Image input is in the wrong format. Expected \"np.ndarray\"!")
        