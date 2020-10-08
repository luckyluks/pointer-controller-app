# Computer Pointer Controller

This app allows to control the computer mouse pointer to move according to the gaze of a person (either based on a video or camera stream).
Therefore, this project is build with the inference engine of the Intel OpenVINO™ toolkit [asd](assssd) for vision based deep learning models.
This project demonstrates the ability of creating a data pipeline that can handle multiple data sources and inference with multiple models in sequence.

| Details            |              |
|-----------------------|---------------|
| Language: |  Python 3.6.X |
| OpenVINO ToolKit: | 2020.1.023 |
| Hardware Used: | Intel(R) Core(TM) i5-6300U |
| Device (OpenVINO) Used: | CPU |

## How it works
This project builds on the inference engine of the [Intel OpenVINO™ toolkit] (https://docs.openvinotoolkit.org/).

As input for the app, the user can specify a data source (camera stream, video file or image).
OpenCV is used for handling this user data in a data pipline.

![data_pipeline](/bin/data_pipeline.png)

The gaze estimation model (in this project: [gaze-estimation-adas-0002](https://docs.openvinotoolkit.org/latest/omz_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)) requires **three** inputs (head pose, left eye image, right eye image) and is therefore supported by **three** other OpenVINO models in this project.
1. **Face detection model**: to detect the face of a person in the data and allows to crop the frame to a face frame.  
Used model in this project: [
face-detection-adas-binary-0001](https://docs.openvinotoolkit.org/latest/omz_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
2. **Pose estimation model**: to estimate the head pose (defined by yaw, pitch and roll).  
Used model in this project: [
head-pose-estimation-adas-0001](https://docs.openvinotoolkit.org/latest/omz_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
2. **Landmarks detection model**: to detect facial landmarks (eyes, mouth, nose)  
to allow the crop of the face frame to an eye frame per side.  
Used model in this project: [landmarks-regression-retail-0009](https://docs.openvinotoolkit.org/latest/omz_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)

## Project Set Up and Installation
*TODO:* Explain the setup procedures to run your project. For instance, this can include your project directory structure, the models you need to download and where to place them etc. Also include details about how to install the dependencies your project requires.

## Demo
*TODO:* Explain how to run a basic demo of your model.

## Documentation
*TODO:* Include any documentation that users might need to better understand your project code. For instance, this is a good place to explain the command line arguments that your project supports.

## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.
