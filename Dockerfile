#Created By Jagadeesh-P
#03-08-2024 | Migrated to Jazzy - 2024

FROM osrf/ros:jazzy-desktop

# Install necessary programs
RUN apt-get update \
    && apt-get install -y \
    nano \
    vim \
    git \
    curl \
    lsb-release \
    gnupg \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
ARG USERNAME=ros
ARG USER_UID=1001
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
  && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
  && mkdir /home/$USERNAME/.config && chown $USER_UID:$USER_GID /home/$USERNAME/.config

# Set up sudo
RUN echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
  && chmod 0440 /etc/sudoers.d/$USERNAME

# Install gz-harmonic (Gazebo Harmonic is the pairing for Jazzy)
RUN curl https://packages.osrfoundation.org/gazebo.gpg --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null \
    && apt-get update \
    && apt-get install -y gz-harmonic

ENV GZ_VERSION=harmonic
ENV PIP_BREAK_SYSTEM_PACKAGES=1

####################################################################################################
# Set up ROS 2 workspace
USER $USERNAME
WORKDIR /home/$USERNAME

RUN sudo apt install -y openjdk-17-jdk \
    && sudo apt-get install -y gitk git-gui \
    && sudo apt-get install -y gcc-arm-none-eabi

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

RUN cd ~/ \
    && git clone --recurse-submodules https://github.com/ardupilot/Micro-XRCE-DDS-Gen.git

RUN cd ~/Micro-XRCE-DDS-Gen \
    && ./gradlew assemble
ENV PATH="/home/ros/Micro-XRCE-DDS-Gen/scripts:$PATH"

RUN cd ~/ \
    && git clone https://github.com/ArduPilot/ardupilot.git \
    && cd ardupilot \
    && git submodule update --init --recursive \
    && git submodule init \
    && git submodule update \
    && git status

RUN cd ~/ardupilot \
    && ./waf distclean \
    && ./waf configure --board MatekF405-Wing

# Extra WS
RUN mkdir -p ~/ws/src

COPY extra.repos /home/${USERNAME}/ws/extra.repos

RUN cd ~/ws/ \
    && vcs import --recursive --input /home/${USERNAME}/ws/extra.repos src \
    && sudo apt update \
    && rosdep update \
    && /bin/bash -c "source /opt/ros/jazzy/setup.bash" \
    && rosdep install -y --from-paths src --ignore-src

# Build ws
RUN cd ~/ws \
    && colcon build || true
RUN /bin/bash -c "source ~/ws/install/setup.bash"

#### ROS2 WS

RUN mkdir -p ~/ros2_ws/src

COPY ros2.repos /home/${USERNAME}/ros2_ws/ros2.repos
COPY ros2_gz.repos /home/${USERNAME}/ros2_ws/ros2_gz.repos

RUN cd ~/ros2_ws/ \
    && vcs import --recursive --input /home/${USERNAME}/ros2_ws/ros2.repos src \
    && sudo apt update \
    && rosdep update \
    && /bin/bash -c "source /opt/ros/jazzy/setup.bash" \
    && rosdep install -y --from-paths src --ignore-src

# Build ardupilot_dds_tests 
RUN /bin/bash -c "source /opt/ros/jazzy/setup.bash \
    && cd ~/ros2_ws \
    && colcon build --packages-up-to ardupilot_dds_tests || true"
RUN /bin/bash -c "source ~/ros2_ws/install/setup.bash"

RUN sudo rm /home/${USERNAME}/ardupilot/Tools/environment_install/install-prereqs-ubuntu.sh
COPY install-prereqs-ubuntu.sh /home/${USERNAME}/ardupilot/Tools/environment_install/install-prereqs-ubuntu.sh

RUN cd ~/ardupilot \
    && sudo apt-get install -y python3-pip \
    && sudo pip3 install future --break-system-packages \
    && Tools/environment_install/install-prereqs-ubuntu.sh -y || true \
    && sudo apt-get install -y python3-pexpect \
    && ./waf clean \
    && ./waf configure --board sitl \
    && ./waf copter -v
ENV PATH="/home/ros/ardupilot/Tools/autotest:$PATH"

RUN cd ~/ardupilot/Tools/autotest \
    && sudo pip3 install MAVProxy --break-system-packages \
    && sudo pip3 install "MAVProxy[joystick]" --break-system-packages \
    && sudo apt-get install -y python3-wxgtk4.0 \
    && sudo pip3 install MAVProxy --upgrade --break-system-packages

# ROS2 with SITL
RUN /bin/bash -c "source /opt/ros/jazzy/setup.bash \
    && cd ~/ros2_ws \
    && MAKEFLAGS='-j1' colcon build \
        --packages-up-to ardupilot_sitl \
        --parallel-workers 1 --executor sequential"

# ROS2 with SITL in GAZEBO — import gz repos
RUN cd ~/ros2_ws \
    && vcs import --input /home/${USERNAME}/ros2_ws/ros2_gz.repos --recursive src || true \
    && sudo apt update \
    && rosdep update \
    && /bin/bash -c "source /opt/ros/jazzy/setup.bash" \
    && rosdep install -y --from-paths src --ignore-src -r || true

# Build up to ardupilot_gz_bringup
RUN /bin/bash -c "source /opt/ros/jazzy/setup.bash \
    && source ~/ros2_ws/install/setup.bash \
    && cd ~/ros2_ws \
    && MAKEFLAGS='-j1' colcon build \
        --packages-up-to ardupilot_gz_bringup \
        --parallel-workers 1 --executor sequential \
        --cmake-args \
            -Dyaml_cpp_LIBRARIES=/usr/lib/x86_64-linux-gnu/libyaml-cpp.so \
            -DCMAKE_EXE_LINKER_FLAGS='-lyaml-cpp' \
            -DCMAKE_SHARED_LINKER_FLAGS='-lyaml-cpp'"

# ardupilot_ros
RUN /bin/bash -c "source /opt/ros/jazzy/setup.bash \
    && cd ~/ros2_ws/src/ \
    && git clone https://github.com/ArduPilot/ardupilot_ros.git \
    && cd ~/ros2_ws/ \
    && rosdep install --from-paths src --ignore-src -r -y --skip-keys gazebo-ros-pkgs \
    && MAKEFLAGS='-j1' colcon build \
        --packages-up-to ardupilot_ros \
        --parallel-workers 1 --executor sequential"

# Install MAVROS and GeographicLib datasets
# Note: install_geographiclib_datasets.sh is called directly to avoid sudo stripping PATH
RUN sudo apt-get update \
    && sudo apt-get install -y \
        ros-jazzy-mavros \
        ros-jazzy-mavros-extras \
    && sudo rm -rf /var/lib/apt/lists/*

RUN sudo /opt/ros/jazzy/lib/mavros/install_geographiclib_datasets.sh

# Copy mavros launch/config/scripts into the container
COPY ./mavros/ /home/${USERNAME}/mavros/

# Copy local src folder to ros2_ws
COPY ./src/ /home/ros/ros2_ws/src/

####################################################################################################

# Copy the entrypoint and bashrc scripts
COPY entrypoint.sh /entrypoint.sh
COPY bashrc /home/${USERNAME}/.bashrc

ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]
CMD ["bash"]
