echo "==================== Creating output directory: models/"
mkdir models/

echo "==================== Models can be found in: $(pwd)/models/"



echo "==================== Starting OpenVINO model download"
# model downloader path variable
md_path="/opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py"

# download
if [ "$1" = "--all_available_precisions" ] ; then
    echo "==================== Starting download of all available precisions"
    python3 $md_path --name face-detection-adas-binary-0001 --output_dir models/
    python3 $md_path --name head-pose-estimation-adas-0001 --output_dir models/
    python3 $md_path --name landmarks-regression-retail-0009 --output_dir models/
    python3 $md_path --name gaze-estimation-adas-0002 --output_dir models/
else
    echo "==================== Starting download off maximum precisions"
    python3 $md_path --name face-detection-adas-binary-0001 --output_dir models/ --precisions FP32-INT1
    python3 $md_path --name head-pose-estimation-adas-0001 --output_dir models/ --precisions FP32
    python3 $md_path --name landmarks-regression-retail-0009 --output_dir models/ --precisions FP32
    python3 $md_path --name gaze-estimation-adas-0002 --output_dir models/ --precisions FP32
fi
echo "==================== [SUCCESS] Download finished"