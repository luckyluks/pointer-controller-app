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

    def predict(self, image, request_id=0, draw_output=True):
        """
        Estimate the facial pose for the given image.
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
            pose_angles, out_image = self.preprocess_output(pred_result, image, draw_output)
            return out_image, pose_angles, predict_end_time

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
            pass

        return pose_angles, image
