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

    def predict(self, image, request_id=0, draw_output=True, **kwargs):
        """
        Estimate the gaze for the given image.
        """
        # Check valid kwargs input
        if not all(any(x == y for y in kwargs.keys()) for x in ["landmarks", "pose"]):
            raise NotImplementedError(f"Incorrect input for \"{self.__class__.__name__}\"!")

        # Pre-process the input (images and angles)
        p_image_left_eye, p_image_right_eye, head_pose_angles = self.preprocess_input(image, **kwargs)


        # Do the inference
        predict_start_time = time.time()
        self.network.start_async(
            request_id=request_id,
            inputs={
                "left_eye_image": p_image_left_eye,
                "right_eye_image": p_image_right_eye,
                "head_pose_angles": head_pose_angles,
            },
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
            landmark_coords, out_image = self.preprocess_output(pred_result, kwargs["face"], draw_output, **kwargs)
            return out_image, landmark_coords, predict_end_time

    def preprocess_input(self, image, **kwargs):
        
        # Pre-process left eye image
        p_image_left_eye = self.preprocess_image(kwargs["landmarks"]["left_eye_image"], 
            self.model.inputs["left_eye_image"].shape[2], 
            self.model.inputs["left_eye_image"].shape[3]
        )

        # Pre-process right eye image
        p_image_right_eye = self.preprocess_image(kwargs["landmarks"]["right_eye_image"], 
            self.model.inputs["right_eye_image"].shape[2], 
            self.model.inputs["right_eye_image"].shape[3]
        )

        # Pre-process head pose angles
        head_pose_angles = np.array(list(kwargs["pose"].values()))
        
        return p_image_left_eye, p_image_right_eye, head_pose_angles 

    def preprocess_image(self, image, width, height):
        """
        Preprocess single image.
        """
        # Get the input shape
        p_frame = cv2.resize(image, (width, height))
        # Change data layout from HWC to CHW
        p_frame = p_frame.transpose((2, 0, 1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        return p_frame

    def preprocess_output(self, results, image, draw_output, **kwargs):
        """
        Process the output and draw estimation results if applicable.
        """
        # Check the model output blob shape
        if not ("gaze_vector" in results.keys()) and (len(results[list(results.keys())[0]][0]) != 3) :
            log.error(f"Incorrect output dimensions in \"{self.__class__.__name__}\": {results.shape}")
            sys.exit(1)  

        # Unpack output
        gaze_vector = results["gaze_vector"].flatten()

        # Draw output, if applicable
        if draw_output:
            left_eye_point = (kwargs["landmarks"]["left_eye_x"], kwargs["landmarks"]["left_eye_y"])
            right_eye_point = (kwargs["landmarks"]["right_eye_x"], kwargs["landmarks"]["right_eye_y"])
            cv2.arrowedLine(
                image,
                (left_eye_point[0], left_eye_point[1]),
                (
                    left_eye_point[0] + int(gaze_vector[0] * 100),
                    left_eye_point[1] - int(gaze_vector[1] * 100),
                ),
                
                color=(0, 0, 255),
                thickness=2,
                tipLength=0.2,
            )
            cv2.arrowedLine(
                image,
                (right_eye_point[0], right_eye_point[1]),
                (
                    right_eye_point[0] + int(gaze_vector[0] * 100),
                    right_eye_point[1] - int(gaze_vector[1] * 100),
                ),
                
                color=(0, 0, 255),
                thickness=2,
                tipLength=0.2,
            )
        
        return gaze_vector, image
