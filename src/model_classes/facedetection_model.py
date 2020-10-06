import sys
import time
import cv2

import logging as log
import numpy as np

from src.model_classes.model import Model


class FaceDetectionModel(Model):
    """
    Class for the Face Detection Model.
    """

    def predict(self, image, request_id=0, draw_boxes=True):
        """
        Predict the bounding box for the given image
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
            outputs = self.network.requests[request_id].outputs[self.output_blob]
            out_image, coords = self.preprocess_output(outputs, image, draw_box)
            return out_image, coords, predict_end_time

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

    def preprocess_output(self, results, image, draw_boxes):
        """
        Process the output and draw detection results if applicable.
        """
        # Check the model output blob shape
        output_dims = results.shape
        if (len(output_dims) != 4)  or (output_dims[3] != 7):
            log.error("Incorrect output dimensions: {}".format(output_dims))
            sys.exit(1)  

        # Look up frame shapes
        width = image.shape[1]
        height = image.shape[0]

        # Parse detection results
        coords = []
        for box in results[0][0]:  # 1x1xNx7
            conf = box[2]
            if conf >= self.probability_threshold:
                xmin = int(box[3] * width)
                ymin = int(box[4] * height)
                xmax = int(box[5] * width)
                ymax = int(box[6] * height)
                coords.append((xmin, ymin, xmax, ymax))

                # Draw detection box, if applicable
                if draw_boxes:
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
        return image, coords
