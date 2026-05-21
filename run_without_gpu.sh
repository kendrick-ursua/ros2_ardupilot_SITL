#!/bin/bash
xhost +local:docker
docker rm -f ardupilot_sitl 2>/dev/null || true
docker run -it \
  --user ros \
  --name ardupilot_sitl \
  --network=host \
  --ipc=host \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --env=DISPLAY \
  -e GZ_VERSION=harmonic \
  --entrypoint /bin/bash \
  ros2_ardupilot_sitl-ardupilot_ros:latest \
  -c "source /opt/ros/jazzy/setup.bash && \
      cd ~/ros2_ws && \
      source install/setup.bash && \
      ros2 launch ardupilot_gz_bringup iris_runway.launch.py"