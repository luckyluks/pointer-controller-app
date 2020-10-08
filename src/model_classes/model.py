import os
import sys
import time
import cv2

import logging as log
import numpy as np

from abc import ABC, abstractmethod
from openvino.inference_engine import IENetwork, IECore


class Model(ABC):
    """
    This is the base model class, which can be used to inherit basic model utilities.
    The class is based on the Abstract Base Class (ABC), which allows to use abstract (placeholder) methods.
    """
    def __init__(self, model_name, device="CPU", extensions=None, probability_threshold=0.5):
        """
        Set the model instance variables.
        """
        self.core = None
        self.network = None
        self.device = device
        self.extensions = extensions
        self.input_blob = None
        self.input_shape = None
        self.output_blob = None
        self.output_shape = None
        self.model = None
        self.model_xml = model_name
        self.model_bin = os.path.splitext(self.model_xml)[0] + ".bin"
        self.probability_threshold = probability_threshold

    def load_model(self):
        """
        This method is loading the model to the device and with extensions specified by the user.
        """
        # Initialize the core
        self.core = IECore()

        # Load extensions, if applicable
        if (self.extensions) and ("CPU" in self.device):
            self.core.add_extension(self.extensions, self.device)

        # Check the IR model
        self.check_model()

        # Load the model
        start_time = time.time()
        self.network = self.core.load_network(network = self.model, device_name=self.device, num_requests=1)

        # Get input/output layers and shapes
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))
        self.input_shape = self.network.inputs[self.input_blob].shape
        self.output_shape = self.network.outputs[self.output_blob].shape

        # Print debug
        log.debug(f"Model \"{self.__class__.__name__}\": "
                  f"sucessfully loaded! (in {1000*(time.time()-start_time):.1f}ms)")


    def check_model(self):
        """
        Read in the IR format of the model and check for supported and unsupported layers in the network.
        """
        # Read in the network
        try:
            self.model = IENetwork(model=self.model_xml, weights=self.model_bin)
        except Exception as loading_error:
            log.error(f"Unable to load the IR model: {loading_error}")
            exit()

        # Check the layers
        supported_layers = self.core.query_network(network=self.model, device_name=self.device)
        unsupported_layers = [layer for layer in self.model.layers.keys() if layer not in supported_layers]
        if len(unsupported_layers) != 0:
            log.warn(f"Unsupported layers found: {unsupported_layers}")
            log.warn("Check whether extensions are available to add to IECore!")
            log.error(f"Unsupported layers found in: {self.__class__.__name__}")
            exit()


    def preprocess_input(self, image, width=None, height=None, **kwargs):

        model_input_dict = dict()

        # determine if input is only image or more
        if all(any(x == y for y in kwargs.keys()) for x in ["landmarks", "pose"]):

            # Pre-process left eye image
            model_input_dict["left_eye_image"] = self.preprocess_image_input(kwargs["landmarks"]["left_eye_image"], 
                self.model.inputs["left_eye_image"].shape[2], 
                self.model.inputs["left_eye_image"].shape[3]
            )

            # Pre-process right eye image
            model_input_dict["right_eye_image"] = self.preprocess_image_input(kwargs["landmarks"]["right_eye_image"], 
                self.model.inputs["right_eye_image"].shape[2], 
                self.model.inputs["right_eye_image"].shape[3]
            )

            # Pre-process head pose
            model_input_dict["head_pose_angles"] = np.array(list(kwargs["pose"].values()))
            return model_input_dict

        else:
            # Pre-process single image
            model_input_dict[self.input_blob] = self.preprocess_image_input(image, width=None, height=None)
            return model_input_dict


    def preprocess_image_input(self, image, width=None, height=None):
        """
        Preprocess image input (use before feeding it to the network).
        """
        # Get the input shape, if input is None
        if (width is None) or (height is None):
            _, _, height, width = self.input_shape
        
        # Scale Image and Change data layout
        p_frame = cv2.resize(image, (width, height))
        p_frame = p_frame.transpose((2, 0, 1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        return p_frame


    def predict(self, image, request_id=0, draw_output=False, **kwargs):
        """
        Predict the bounding box for the given image
        """
        # Check valid input
        self.check_input(image, **kwargs)

        # Pre-process the image
        input_dict = self.preprocess_input(image, **kwargs)

        # Do the inference
        predict_start_time = time.time()
        
        self.network.start_async(request_id=request_id, 
            inputs=input_dict
        )
        inference_status = self.network.requests[request_id].wait(-1)
        inference_time_ms = (time.time() - predict_start_time) * 1000

        # Process network output
        if inference_status == 0:

            # Parse network output
            results = {}
            for output_name in self.model.outputs.keys():
                results[output_name] = self.network.requests[request_id].outputs[output_name]
            
            # Process output
            out_image, ouput = self.preprocess_output(results, image, draw_output, **kwargs)
            return out_image, ouput, inference_time_ms


    @abstractmethod
    def preprocess_output(self, results, image, draw_output=False, **kwargs):
        """
        Process model output. Additionally, draw output to the given image, if draw_output is true.
        """
        raise NotImplementedError("This method is not generally implemented!")


    @abstractmethod
    def check_input(self, image, **kwargs):
        """
        Check model input, depending on the model,
        """
        raise NotImplementedError("This method is not generally implemented!")
