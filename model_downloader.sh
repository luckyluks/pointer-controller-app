echo "Starting OpenVINO model download"

cd /opt/intel/openvino/deployment_tools/tools/model_downloader/

python3 downloader.py --name face-detection-adas-binary-0001 --output_dir /home/lukas/pointer-controller-app/models/
python3 downloader.py --name head-pose-estimation-adas-0001 --output_dir /home/lukas/pointer-controller-app/models/
python3 downloader.py --name landmarks-regression-retail-0009 --output_dir /home/lukas/pointer-controller-app/models/
python3 downloader.py --name gaze-estimation-adas-0002 --output_dir /home/lukas/pointer-controller-app/models/

echo "[SUCCESS] download finished"