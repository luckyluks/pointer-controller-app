import cv2

import logging as log
from argparse import ArgumentParser
from src.input_feeder import InputFeeder

from src.model_classes.facedetection_model import FaceDetectionModel
from src.model_classes.poseestimation_model import PoseEstimationModel
from src.model_classes.landmarksdetection_model import LandmarksDetectionModel
from src.model_classes.gazeestimation_model import GazeEstimationModel

# from src.mouse_controller import MouseController

def infer_on_stream(args):
    """
    
    """

    # # Load the mouse controller
    # mouse_controller = MouseController(
    #     precision=args.mouse_precision, speed=args.mouse_speed
    # )

    # Load the models
    model_face_detection = FaceDetectionModel(args.model_face_detection)
    model_face_detection.load_model()
    model_pose_estimation = PoseEstimationModel(args.model_pose_estimation)
    model_pose_estimation.load_model()
    model_landmarks_detection = LandmarksDetectionModel(args.model_landmarks_detection)
    model_landmarks_detection.load_model()
    model_gaze_estimation = GazeEstimationModel(args.model_gaze_estimation)
    model_gaze_estimation.load_model()

    # Open input and output stream
    feed = InputFeeder(input_file=args.input)
    fps, input_frame_format = feed.load_data()
    print(fps, input_frame_format)
    output_stream = open_output(args.output, fps, input_frame_format)

    # # DEBUG
    # counter=0

    for batch in feed.next_batch():

        # Break the loop, when playback ends
        if batch is None:
            log.info("Input stream ended!")
            break

        # Write on disk 
        if output_stream:
            output_stream.write(batch)

        # DEBUG
        # print(counter)
        # counter+=1


    feed.close()

    # Release Output stream if the writer selected
    if output_stream:
        output_stream.release()
        


def main():
    """
    Load the network and parse the output.
    :return: None
    """
    # Set log level to log some info in console
    log.basicConfig(level=log.DEBUG)
    
    # Grab command line args
    args = build_argparser().parse_args()
    
    # Perform inference on the input stream
    infer_on_stream(args)
    
    # Log end
    log.info("[SUCCESS] Application finished!")


def open_output(path, fps, frame_size):
    """
    Open the output stream with the given settings.
    """
    output_stream = None
    if path != "":
        if not path.endswith('.mp4'):
            log.warning("Output file extension is not 'mp4'. " \
                        "Some issues with output can occur!")
        try:
            output_stream = cv2.VideoWriter(path, cv2.VideoWriter.fourcc(*'mp4v'), fps, frame_size)
        except Exception as identifier:
            log.warning(f"Failed to open output stream with setting: " \
                         f"path={path}, fps={fps}, format={frame_size} ...")
            log.warning(identifier)
        
        log.info(f"Output stream is open: \"{path}\"")
    return output_stream
    

def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-mfd", "--model_face_detection", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-mpe", "--model_pose_estimation", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-mle", "--model_landmarks_detection", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-mge", "--model_gaze_estimation", required=True, type=str,
                        help="Path to an xml file with a trained model.")

    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image file OR video file OR camera "
                             "stream. For stream use \"CAM\" as input!")
    parser.add_argument('-o', '--output', default="",
                         help="(optional) Path to save the output video to")

    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                             "(0.5 by default)")
    parser.add_argument("-md", "--maximum_detections", type=int, default=3,
                        help="Maximum count of detections in the frame "
                             " before a warning appears")
    parser.add_argument("-mt", "--maximum_time", type=int, default=10,
                        help="Maximum time a detected person is in the frame"
                             " before a warning appears (in seconds)")
    parser.add_argument("-mr", "--maximum_requests", type=int, default=4,
                        help="Maximum numer of requests that can be handled"
                             " by the network at once (integer)")
    
    return parser

if __name__ == "__main__":
    main()