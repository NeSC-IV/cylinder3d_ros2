# Cylinder3D-ROS2

Deploy [Cylinder3D](https://github.com/xinge008/Cylinder3D) on ROS2

## Prerequisites
### conda env: environment.yml

Tested on Ubuntu 20.04 LTS. Recommend pip install over conda install.
- Python 3.8
- Pytorch (test with Pytorch==2.0.0 and cuda 11.7)
- yaml
- Cython
- torch-scatter
- nuScenes-devkit (optional for nuScenes)
- spconv (test with spconv==2.3.5)
- ROS2 Humble

## Install
```
# mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
git clone https://github.com/NeSC-IV/cylinder3d_ros2.git
cd ..
colcon build --symlink-install --packages-select cylinder3d_ros2
```

## Run
```
# python env path in launch file need to be modified
'/home/$USER$/.conda/envs/$ENV_NAME$/bin/python'

# ROS2 RUN
ros2 launch cylinder3d_ros2 cylinder3d_ros2.launch.py
# PYTHON
python test_semantic.py

### wait for the model to be loaded, when message
[semantic_listener]: Listener is started, listening to /velodyne_points
[semantic_publisher]: Publisher has been started, publishing topic: /sem_points
[semantic_network]: Init ready!
 are printed, the model is ready to use
```
