import os
import cv2
import time

import logging as log
from tqdm import tqdm
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

    # Open input stream
    input_stream = InputFeeder(input_file=args.input)
    fps, input_frame_format = input_stream.load_data()
    
    # Open output stream (is None if image)
    output_stream = open_output(args.output, fps, input_frame_format)

    # Create vars to collect inference time
    inference_time_dict = {
        "face" : [],
        "pose" : [],
        "landsmarks" : [],
        "gaze" : [],
        "total" : []
    }
        
    # Create progress bar
    bar_format="{l_bar}{bar}|{n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]"
    pbar = tqdm(total=input_stream.get_framecount(), bar_format=bar_format)

    # Catch unexpected behaviour at the inference
    try:
        # Fetch next frame batch
        for batch in input_stream.next_batch():

            # Break the loop, when playback ends
            if batch is None:
                # Exit progress bar
                pbar.close()
                log.info("Input stream ended!")
                break

            # Start inference time
            time_start_face = time.time()

            # Do face detection
            out_image, coords, face_detection_time = model_face_detection.predict(batch, draw_boxes=args.draw_prediction)

            # Do not go further if no face is detected
            if coords:
                for face_bbox in coords:

                    # Crop the frame to the face region
                    (xmin, ymin, xmax, ymax) = face_bbox
                    face = batch[ymin:ymax, xmin:xmax]
                    
                    # Ensure sufficient cropped size 
                    if any([True if length < 20 else False for length in face.shape[:2]]):
                        log.warn(f"Cropped frame size not sufficient (under 20px): {face.shape[:2]}")
                        continue

                    # # TODO DO landmarks detection
                    # facial_landmarks_pred_time, eyes_coords = facial_landmarks.predict(
                    # face, show_bbox=args.show_bbox
                    # )

                    # # TODO Do pose estimation
                    # hp_est_pred_time, head_pose_angles = head_pose_estimation.predict(
                    #     face, show_bbox=args.show_bbox
                    # )

                    # # TODO Do gaze estimation
                    # gaze_pred_time, gaze_vector = gaze_estimation.predict(
                    #     frame,
                    #     show_bbox=args.show_bbox,
                    #     face=face,
                    #     eyes_coords=eyes_coords,
                    #     head_pose_angles=head_pose_angles,
                    # )

                    # if args.debug:
                    #     head_pose_estimation.show_text(frame, head_pose_angles)
                    #     gaze_estimation.show_text(frame, gaze_vector)

                    # if args.enable_mouse:
                    #     mouse_controller.move(gaze_vector["x"], gaze_vector["y"])




            # Sum up inference time
            inference_time_dict["face"].append(face_detection_time)
            total_inference_time = face_detection_time
            inference_time_dict["total"].append(total_inference_time)

            # Write on disk 
            if args.output:
                if fps == 1:
                    cv2.imwrite(os.path.splitext(args.input)[0] + "_out.png", face)
                else:
                    output_stream.write(out_image)

            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({'inference time': f"{total_inference_time:.2f}ms"})

    except Exception as e:
        log.error(f"Could not run Inference ({type(e).__name__}): {e}")

    # Close Input stream
    input_stream.close()

    # Release Output stream if the writer selected TODO: replace
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


def open_output(file_path, fps, frame_size):
    """
    Open the output stream with the given settings.
    """
    output_stream = None
    if (file_path != "") and (fps != 1):
        if not file_path.endswith('.mp4'):
            log.warning("Output file extension is not 'mp4'. " \
                        "Some issues with output can occur!")
        try:
            output_stream = cv2.VideoWriter(file_path, cv2.VideoWriter.fourcc(*'mp4v'), fps, frame_size)
        except Exception as identifier:
            log.warning(f"Failed to open output stream with setting: " \
                         f"path={file_path}, fps={fps}, format={frame_size} ...")
            log.warning(identifier)
        
        log.info(f"Output stream is open: \"{file_path}\"")
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
    parser.add_argument('-o', '--output', required=False, default="",
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

    parser.add_argument("--draw_prediction", type=bool, default=True,
                        help="(optional) Draw the prediction outputs.")
    # parser.add_argument("-mt", "--maximum_time", type=int, default=10,
    #                     help="Maximum time a detected person is in the frame"
    #                          " before a warning appears (in seconds)")
    # parser.add_argument("-mr", "--maximum_requests", type=int, default=4,
    #                     help="Maximum numer of requests that can be handled"
    #                          " by the network at once (integer)")
    
    return parser

if __name__ == "__main__":
    main()