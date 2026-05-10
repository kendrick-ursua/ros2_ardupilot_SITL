#!/bin/bash
xhost +local:docker
docker rm -f ardupilot_sitl 2>/dev/null || true
docker run -it \
  --user ros \
  --name ardupilot_sitl \
  --network=host \
  --gpus all \
  --ipc=host \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --env=DISPLAY \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e GZ_VERSION=harmonic \
  --entrypoint /bin/bash \
  ros2_ardupilot_sitl-ardupilot_ros:latest \
  -c "source /opt/ros/humble/setup.bash && \
      cd ~/ros2_ws && \
      source install/setup.bash && \
      ros2 launch ardupilot_gz_bringup iris_runway.launch.py"