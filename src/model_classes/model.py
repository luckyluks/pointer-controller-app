import os
import sys

import logging as log
from openvino.inference_engine import IENetwork, IECore

class Model():
    """
    This is the base model class, which can be used to inherit basic model utilities.
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
        self.network = self.core.load_network(network = self.model, device_name=self.device, num_requests=1)

        # Get input/output layers and shapes
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))
        self.input_shape = self.network.inputs[self.input_blob].shape
        self.output_shape = self.network.outputs[self.output_blob].shape

        # Print debug
        log.debug(f"Model \"{self.__class__.__name__}\": sucessfully loaded!")

    # def predict(self, image):
    #     """
    #     TODO: You will need to complete this method.
    #     This method is meant for running predictions on the input image.
    #     """
    #     raise NotImplementedError

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
        
    # def preprocess_input(self, image):
    #     """
    #     Before feeding the data into the model for inference,
    #     you might have to preprocess it. This function is where you can do that.
    #     """
    #     raise NotImplementedError

    # def preprocess_output(self, outputs):
    #     """
    #     Before feeding the output of this model to the next model,
    #     you might have to preprocess the output. This function is where you can do that.
    #     """
    #     raise NotImplementedError
