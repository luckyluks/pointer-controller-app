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

    def preprocess_output(self, results, image, draw_boxes):
        """
        Process the output and draw detection results if applicable.
        """
        # Check the model output blob shape
        if not ("detection_out" in results.keys()):
            log.error(f"Incorrect model output dictonary in \"{self.__class__.__name__}\": {results.keys()}")
            sys.exit(1)
        output_dims = results["detection_out"].shape
        if (len(output_dims) != 4)  or (output_dims[3] != 7):
            log.error(f"Incorrect output dimensions in \"{self.__class__.__name__}\": {output_dims}")
            sys.exit(1)

        # Look up frame shapes
        width = image.shape[1]
        height = image.shape[0]

        # Parse detection results
        coords = []
        for box in results["detection_out"][0][0]:  # 1x1xNx7
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


    def check_input(self, image, **kwargs):
        """
        Check data input.
        """
        if not isinstance(image, np.ndarray):
            raise IOError("Image input is in the wrong format. Expected \"np.ndarray\"!")