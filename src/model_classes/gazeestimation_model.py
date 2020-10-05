from src.model_classes.model import Model

class GazeEstimationModel(Model):
    """
    Class for the Gaze Estimation Model.
    """
    # def __init__(self, model_name, device="CPU", extensions=None):
    #     """
    #     TODO: Use this to set your instance variables.
    #     """
    #     raise NotImplementedError

    def predict(self, image):
        """
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        """
        raise NotImplementedError

    def preprocess_input(self, image):
        """
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        """
        raise NotImplementedError

    def preprocess_output(self, outputs):
        """
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        """
        raise NotImplementedError
