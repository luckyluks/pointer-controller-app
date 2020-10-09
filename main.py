import os
import cv2
import time

import numpy as np
import logging as log
from tqdm import tqdm
from argparse import ArgumentParser

from src.input_feeder import InputFeeder
from src.model_classes.facedetection_model import FaceDetectionModel
from src.model_classes.poseestimation_model import PoseEstimationModel
from src.model_classes.landmarksdetection_model import LandmarksDetectionModel
from src.model_classes.gazeestimation_model import GazeEstimationModel
# from src.mouse_controller import MouseController


def run_on_stream(args):
    """
    Run inference on stream (cam or video or image) and controll the mouse.
    """

    # # Load the mouse controller
    # mouse_controller = MouseController(
    #     precision=args.mouse_precision, speed=args.mouse_speed
    # )

    # Detect lower log level
    low_log_level = log.getLogger().level in [log.INFO, log.DEBUG]

    log.info(f"{'='*10} Setup: mouse enabled: {args.enable_mouse}, " \
             f"draw predicition results: {args.draw_prediction} {'='*10}")

    # Load the models
    precisions = parse_precisions(args.precision)
    model_face_detection = FaceDetectionModel(args.model_face_detection, precision=precisions["face"])
    model_face_detection.load_model()
    model_pose_estimation = PoseEstimationModel(args.model_pose_estimation, precision=precisions["pose"])
    model_pose_estimation.load_model()
    model_landmarks_detection = LandmarksDetectionModel(args.model_landmarks_detection, precision=precisions["landmarks"])
    model_landmarks_detection.load_model()
    model_gaze_estimation = GazeEstimationModel(args.model_gaze_estimation, precision=precisions["gaze"])
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
    if low_log_level:
        bar_format="{l_bar}{bar}|{n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]"
        pbar = tqdm(total=input_stream.get_framecount(), bar_format=bar_format)

    # Start inference time
    time_start = time.time()

    # Catch unexpected behaviour at the inference
    try:
        # Fetch next frame batch
        for batch in input_stream.next_batch():

            # Break the loop, when playback ends
            if batch is None:
                # Exit progress bar, if present
                if low_log_level:
                    pbar.close()
                
                # Log output
                log.info("Input stream ended!")
                log.info(f"Total processing time elapsed: {time.time()-time_start :.2f}s")
                log.info(f"Average inference time: {np.mean(inference_time_dict['total']):.1f}ms")
                break

            # Do face detection
            out_image, coords, face_detection_time = model_face_detection.predict(batch, draw_output=args.draw_prediction)

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

                    # TODO DO landmarks detection
                    _, landmarks_info, landmarks_detection_time = model_landmarks_detection.predict(
                    face, draw_output=args.draw_prediction
                    )

                    # cv2.imwrite("test_right.png", landmarks_info["right_eye_image"])
                    # cv2.imwrite("test_left.png", landmarks_info["left_eye_image"])

                    # # TODO Do pose estimation
                    _, pose_angels, pose_estimation_time = model_pose_estimation.predict(
                        face, draw_output=args.draw_prediction
                    )

                    # # TODO Do gaze estimation
                    _, gaze_vector, gaze_estimation_time = model_gaze_estimation.predict(
                        face,
                        draw_output=args.draw_prediction,
                        landmarks=landmarks_info,
                        pose=pose_angels
                    )

                    if args.enable_mouse:
                        print(gaze_vector)
                        # mouse_controller.move(gaze_vector["x"], gaze_vector["y"])

            # Sum up inference time
            inference_time_dict["face"].append(face_detection_time)
            inference_time_dict["pose"].append(pose_estimation_time)
            inference_time_dict["landsmarks"].append(landmarks_detection_time)
            inference_time_dict["gaze"].append(gaze_estimation_time)
            
            total_inference_time = face_detection_time \
                                   + pose_estimation_time \
                                   + landmarks_detection_time \
                                   + gaze_estimation_time
            inference_time_dict["total"].append(total_inference_time)

            # Write on disk (if fps is 1, then the output is image!)
            if args.output:
                if fps == 1:
                    # Check output path and create "_out" filename, if needed
                    if args.output.endswith((".jpg", ".bmp", ".png")):
                        cv2.imwrite(args.output, out_image)
                    else:
                        cv2.imwrite(os.path.splitext(args.input)[0] + "_out.png", out_image)
                else:
                    output_stream.write(out_image)

            # Update progress bar
            if low_log_level:
                pbar.update(1)
                pbar.set_postfix({'inference time': f"{total_inference_time:.1f}ms"})

    except Exception as e:
        log.error(f"Could not run Inference ({type(e).__name__}): {e}")
        return False

    finally:
        # Close Input stream
        input_stream.close()

        # Release Output stream if the writer selected
        if output_stream:
            output_stream.release()

        # Close progress bar if present
        if low_log_level:
            pbar.close()
    
    return True


def main():
    """
    Run the application.
    """
    # Grab command line args
    args = build_argparser().parse_args()

    # Set log level to log some info in console
    log.basicConfig(level=args.log_level)
    
    # Perform inference on the input stream
    success = run_on_stream(args)
    
    # Log end, if successful
    if success:
        log.info(f"{'='*10} [SUCCESS] Application finished! {'='*10}")


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


def parse_precisions(precisions_input):
    """
    Tries to parse the given precisions input:
    either one-for-all, like FP32 or FP16 (FP32 by default),
    or per-model seprated with \"&\" symbol, like FP32&FP16&FP16&FP16.
    """
    if precisions_input.count("/") == 3:
        precisions_tokens = precisions_input.split("/")
        return {
            "face": precisions_tokens[0],
            "pose": precisions_tokens[1],
            "landmarks": precisions_tokens[2],
            "gaze": precisions_tokens[3]
        }

    else:
        return {
            "face": "FP32",
            "pose": "FP32",
            "landmarks": "FP32",
            "gaze": "FP32",
        }




def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image file OR video file OR camera "
                             "stream. For stream use \"CAM\" as input!")
    parser.add_argument('-o', '--output', required=False, default="",
                         help="(optional) Path to save the output video to")

    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="(optional) MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="(optional) Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-p", "--precision", type=str, default="FP32",
                        help="(optional) Specify the model inference precision: "
                             "either one-for-all, like FP32 or FP16 (FP32 by default), "
                             "or per-model seprated with \"/\" symbol, like FP32&FP16&FP16&FP16."
                             "WARNING: Only works with default model paths from download_models.sh")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="(optional) Set probability threshold for detections filtering"
                             "(0.5 by default)")
    
    parser.add_argument("-dp", "--draw_prediction", default=False, action='store_true',
                        help="(optional) Draw the prediction outputs.")
    parser.add_argument("-em", "--enable_mouse", default=False, action='store_true',
                        help="(optional) Enable mouse movement.")

    parser.add_argument("-mfd", "--model_face_detection", type=str,
                        default="models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml",
                        help="(optional) Set path to an xml file with a trained face detection model. "
                             "Default is the FP32 face-detection-adas-binary-0001")
    parser.add_argument("-mpe", "--model_pose_estimation", type=str,
                        default="models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml",
                        help="(optional) Set path to an xml file with a trained pose estimation model. "
                             "Default is the FP32 head-pose-estimation-adas-0001")
    parser.add_argument("-mle", "--model_landmarks_detection", type=str,
                        default="models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml",
                        help="(optional) Set path to an xml file with a trained landmarks detection model. "
                             "Default is the FP32 landmarks-regression-retail-0009")
    parser.add_argument("-mge", "--model_gaze_estimation", type=str,
                        default="models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml",
                        help="(optional) Set path to an xml file with a trained gaze estimation model. "
                             "Default is the FP32 gaze-estimation-adas-0002.xml")

    parser.add_argument('-db', '--debug', action="store_const",
                        dest="log_level", const=log.DEBUG, default=log.WARNING,
                        help="(optional) Sets loging level to DEBUG, "
                        "instead of WARNING (for developers).")
    parser.add_argument('-v', '--verbose', action="store_const",
                        dest="log_level", const=log.INFO,
                        help="(optional) Sets loging level to INFO, "
                        "instead of WARNING (for users).")
    
    return parser


if __name__ == "__main__":
    main()