
SDV (Self-Driving Vehicle) Project — Quanser QCar2
Complete Project Context & Master Prompt

🚗 PROJECT OVERVIEW
We are building a market-level autonomous self-driving vehicle (SDV) using the Quanser QCar2 platform. The goal is to develop a full autonomous driving stack — from raw sensor perception to vehicle control — comparable to industry-level systems like Tesla Autopilot or Waymo, scaled to the QCar2 research platform.

🖥️ SYSTEM ARCHITECTURE
Access Setup

User Machine: Windows HP ProBook laptop
Connection: Remote Desktop Protocol (RDP) → QCar2 Ubuntu Desktop
QCar2 IP: 192.168.41.88
Username: nvidia

QCar2 Hardware

Platform: Quanser QCar2 (1/10th scale research vehicle)
SBC: NVIDIA Jetson AGX Orin Developer Kit
OS: Ubuntu 20.04.6 LTS (Focal Fossa)
RAM: 30GB
Storage: 185GB free
CPU: 8-core ARM Cortex-A78AE (ARMv8)
GPU: NVIDIA Ampere (Orin integrated), CUDA 11.4
GPU Devices: nvhost-gpu, nvdla0, nvdla1


✅ VERIFIED WORKING SENSORS
SensorTopicStatusDetailsRealSense D435 RGB/camera/color_image✅ LIVE1280x720 BGR8 @ 30fpsRealSense D435 Depth/camera/depth_image✅ LIVE1280x720 16UC1RPLiDAR/scan✅ LIVE360° range 0-10mIMU/qcar2_imu✅ LIVE9-DOF, gravity 9.87 m/s²EKF Odometry/odometry/filtered✅ LIVEFused pose estimateMap/map✅ LIVEOccupancy gridNav2 StackFull stack✅ RUNNINGPath planning active

⚠️ CSI CAMERA STATUS
The QCar2 has 4 CSI cameras (IMX219, 820x616 @ 80fps):

Right = Camera ID 0
Rear = Camera ID 1
Front = Camera ID 2
Left = Camera ID 3

Current Issue: CSI cameras only work when an HDMI monitor is physically connected to the QCar2. Without HDMI, the NVIDIA Argus daemon cannot initialize EGL display context, causing all CSI camera reads to return blank frames.
Root Cause: The Quanser libquanser_media.so uses libnvargus_socketclient.so which requires a valid EGL display. Remote Desktop (RDP/XRDP) does not provide a real GPU-backed display.
Workaround: Keep HDMI monitor plugged into QCar2 permanently. The monitor does not need to be watched — just physically connected.
Fix Required: Contact Quanser support (support@quanser.com) and request the custom JetPack image for QCar2 that creates proper /dev/camera/ device nodes.

🔧 SOFTWARE ENVIRONMENT
ROS2

Version: ROS2 Humble Hawksbill
Workspace: /home/nvidia/Desktop/SDV_workspace/
Source Command:

bashsource /opt/ros/humble/setup.bash
source /home/nvidia/Documents/Quanser/5_research/sdcs/qcar2/ros2/install/setup.bash
source /home/nvidia/ros2/install/setup.bash
export PYTHONPATH="/home/nvidia/Documents/Quanser/0_libraries/python:$PYTHONPATH"
Python & ML Stack

Python: 3.8.10
PyTorch: 2.1.0a0+41361538.nv23.06 (Jetson custom build, CUDA enabled)
Torchvision: 0.16.0 (built from source)
OpenCV: 4.12.0
NumPy: 1.23.0
YOLOv8 (Ultralytics): 8.2.101
CUDA: 11.4 at /usr/local/cuda-11.4

Quanser Packages

qcar2_autonomy — lane_detector, path_follower, traffic_system_detector, yolo_detector
qcar2_interfaces — custom message types
qcar2_nodes — csi, lidar, rgbd, qcar2_hardware, nav2_qcar2_converter

Key Paths
SDV Workspace:    /home/nvidia/Desktop/SDV_workspace/
Quanser Library:  /home/nvidia/Documents/Quanser/0_libraries/python/
YOLOv8 Model:     /home/nvidia/Desktop/SDV_workspace/models/yolov8n.pt
PAL Library:      /home/nvidia/Documents/Quanser/0_libraries/python/pal/
Default Map:      /home/nvidia/maps/qcar_map.yaml
EKF Config:       /home/nvidia/Documents/ekf.yaml

🚨 CRITICAL SYSTEM CONSTRAINTS
⚠️ DO NOT BREAK THESE — EVER
1. PyTorch CUDA Build
The Jetson has a custom PyTorch wheel that enables GPU/CUDA support. Standard pip install torch will overwrite it with a CPU-only version and break the entire ML stack.

Rule: ALWAYS use pip install --no-deps when installing anything
If broken: Reinstall from ~/Downloads/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
Verify CUDA: python3 -c "import torch; print(torch.cuda.is_available())"

2. Torchvision Source Build
Torchvision was built from source at v0.16.0 to match the custom PyTorch.

Rule: Never reinstall torchvision via pip without --no-deps
If broken: Rebuild from ~/Downloads/torchvision/

3. ROS2 Quanser Packages
The Quanser ROS2 packages at /home/nvidia/Documents/Quanser/5_research/sdcs/qcar2/ros2/ are pre-built and calibrated.

Rule: Never rebuild or modify these packages
Rule: Never run colcon build in the Quanser directory

4. System Libraries

Rule: Never run apt upgrade or apt dist-upgrade — can break Jetson kernel/drivers
Rule: Never update CUDA, cuDNN, or TensorRT via apt
Rule: Never modify /boot/extlinux/extlinux.conf without a backup

5. pip Installations

Rule: Always use pip3 install --no-deps package_name to avoid dependency conflicts
Rule: Prefer installing packages to the user space: pip3 install --user package
Rule: Never use pip3 install --upgrade without checking what it will affect


🔒 SAFE INSTALLATION RULES
When installing ANY new package, library, or tool:
bash# SAFE: install without touching dependencies
pip3 install --no-deps <package>

# SAFE: install to user space only
pip3 install --user --no-deps <package>

# SAFE: check what will be installed BEFORE installing
pip3 install --dry-run <package>

# NEVER DO THIS:
pip3 install torch          # will break CUDA PyTorch
pip3 install --upgrade pip  # may break package resolution
sudo apt upgrade             # may break Jetson kernel
Any new ROS2 packages should be:

Created inside /home/nvidia/Desktop/SDV_workspace/src/
Built with colcon build --packages-select <package_name>
Never touching the Quanser workspace


🗺️ FULL AUTONOMOUS DRIVING ARCHITECTURE PLAN
┌─────────────────────────────────────────────────────────┐
│                    SENSOR LAYER                          │
│  RealSense RGB │ RealSense Depth │ LiDAR │ IMU │ CSI    │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                 PHASE 1: PERCEPTION          ← NOW       │
│  YOLOv8 Detection │ Lane Detection │ Depth Fusion        │
│  LiDAR Processing │ Obstacle Detection │ Sign Recognition│
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│           PHASE 2: LOCALIZATION & SLAM                   │
│  EKF Fusion │ Visual Odometry │ Map Building             │
│  Loop Closure │ Global Localization                      │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│              PHASE 3: PATH PLANNING                      │
│  Global Planner │ Local Planner │ Obstacle Avoidance     │
│  Traffic Rules │ Behavior Planning                       │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│           PHASE 4: VEHICLE CONTROL (MPC)                 │
│  Model Predictive Control │ Steering │ Throttle │ Brake  │
│  PID Controllers │ Safety Limits                         │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│               PHASE 5: ADAS FEATURES                     │
│  Emergency Braking │ Lane Keep Assist │ Adaptive Cruise  │
│  Traffic Sign Compliance │ Pedestrian Detection          │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│              PHASE 6: FULL AUTONOMY                      │
│  End-to-End Autonomous Navigation                        │
│  Multi-scenario handling │ Edge case management          │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│           PHASE 7: OPTIMIZATION                          │
│  TensorRT Inference │ INT8 Quantization                  │
│  Multi-stream pipeline │ <5ms latency target             │
└─────────────────────────────────────────────────────────┘

📦 CURRENT PROGRESS
Completed ✅

Full environment setup (.bashrc, CUDA, ROS2, Quanser paths)
PyTorch CUDA verified working on Orin GPU
YOLOv8 imported and running on GPU
All sensors verified live (RealSense, LiDAR, IMU, Nav2)
SDV workspace created at ~/Desktop/SDV_workspace/
Base perception pipeline written (sdv_perception_pipeline.py)
Device tree overlay applied for CSI cameras
QUARC target manager identified and running

In Progress 🔄

Phase 1: Perception pipeline (YOLOv8 + lane detection + depth fusion)
CSI camera fix (waiting for Quanser support / HDMI workaround)

Pending ⏳

Phase 2: SLAM & Localization
Phase 3: Path Planning
Phase 4: MPC Vehicle Control
Phase 5: ADAS Features
Phase 6: Full Autonomy
Phase 7: TensorRT Optimization


🎯 CURRENT PHASE GOALS (Phase 1 — Perception)

Object Detection: YOLOv8n running on GPU detecting people, cars, signs, obstacles
Depth Fusion: Each detection gets a real 3D position in meters using RealSense depth
Lane Detection: Canny + Hough pipeline detecting left/right lanes and center offset
LiDAR Obstacle Map: Convert LiDAR scan to obstacle grid in front 180° arc
Critical Obstacle Alert: Trigger emergency flag if obstacle < 1.5m ahead
ROS2 Integration: Publish all outputs to /sdv/perception/fused at 30fps
