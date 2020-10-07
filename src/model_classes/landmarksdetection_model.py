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

    def predict(self, image, request_id=0, draw_output=True):
        """
        Estimate the facial Landmarks for the given image.
        """
        # Check valid input
        if not isinstance(image, np.ndarray):
            raise IOError("Image input is in the wrong format. Expected \"np.ndarray\"!")

        # Pre-process the image
        p_image = self.preprocess_input(image)

        # Do the inference
        predict_start_time = time.time()
        self.network.start_async(request_id=request_id, 
            inputs={self.input_blob: p_image}
        )
        inference_status = self.network.requests[request_id].wait(-1)
        predict_end_time = (time.time() - predict_start_time) * 1000

        # Process network output
        if inference_status == 0:

            # Parse network output
            pred_result = {}
            for output_name in self.model.outputs.keys():
                pred_result[output_name] = self.network.requests[request_id].outputs[output_name]
            
            # Process output
            landmark_coords, out_image = self.preprocess_output(pred_result, image, draw_output)
            return out_image, landmark_coords, predict_end_time

    def preprocess_input(self, image):
        """
        Preprocess the image (use before feeding it to the network).
        """
        # Get the input shape
        _, _, height, width = self.input_shape
        p_frame = cv2.resize(image, (width, height))
        # Change data layout from HWC to CHW
        p_frame = p_frame.transpose((2, 0, 1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        return p_frame

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

        return face_landmarks, image
