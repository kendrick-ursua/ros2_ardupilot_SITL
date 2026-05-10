# 🚁 ArduPilot ROS2 Docker Environment

<div align="center">


  <div align="center">
  <img src="https://github.com/farshidrayhancv/ROS2_ardupilot_Iris_docker/blob/main/Sample_1.png?raw=true" alt="ArduPilot ROS2 Docker Environment Preview" width="100%">
  <p>
    A complete Docker environment for developing with ArduPilot, ROS2 Humble, and Gazebo Harmonic
  </p>
</div>

</div>

## ✨ Features

- 🐳 Pre-configured Docker environment with ROS2 Humble
- 🛩️ ArduPilot SITL (Software In The Loop) integration
- 🌎 Gazebo Harmonic for simulation
- 🔌 ArduPilot-Gazebo plugins and integration
- 📡 DDS middleware for ArduPilot communication
- 🏗️ Ready-to-use development workspace
- 💻 Visual Studio Code DevContainer support for seamless development

## 🛠️ Prerequisites

- 🐳 Docker installed on your system
- 🔄 Docker Compose installed on your system
- 🖥️ X11 server running for GUI applications (on Linux this is normally running by default)
- 💻 Visual Studio Code with Remote - Containers extension (for DevContainer support)

## 🚀 Quick Start

### 🐳 Using Docker Compose

1. Clone this repository:
   ```bash
   git clone git@github.com:kendrick-ursua/ros2_ardupilot_SITL.git
   cd ros2_ardupilot_SITL
   ```

2. Build and start the container:
   ```bash
   docker compose build
   docker compose up -d
   ```

3. Connect to the container:
   ```bash
   docker compose exec ardupilot_ros bash
   ```

### 💻 Using Visual Studio Code DevContainer

1. Install the "Remote - Containers" extension in VS Code
2. Clone this repository and open it in VS Code
3. Click on the green button in the bottom-left corner of VS Code
4. Select "Reopen in Container" from the menu
5. VS Code will build the container and open it automatically

The DevContainer configuration is located in the `.devcontainer` directory, containing:
- `devcontainer.json`: Configuration for VS Code integration
- `docker-compose.yml`: Container configuration for the development environment

## 📂 Repository Structure

```
.
├── bashrc                     # Custom bashrc for the container
├── docker-compose.yml         # Docker Compose configuration
├── Dockerfile                 # Docker image definition
├── entrypoint.sh              # Container entrypoint script
├── install-prereqs-ubuntu.sh  # ArduPilot prerequisites installer
├── instruction.sh             # Additional instructions
├── ros2_gz.repos              # ROS2 Gazebo repos file
├── ros2.repos                 # ROS2 repos file
├── workspace                  # Shared workspace directory
└── .devcontainer/             # VS Code DevContainer configuration
    ├── devcontainer.json
    └── docker-compose.yml
```

## 🚁 Using ArduPilot SITL with ROS2

The container includes a helper script `~/Ardupilot_ROS.sh` that provides various testing commands for ArduPilot SITL and ROS2 integration.

### 🧪 Testing ArduPilot SITL

Run the following commands to test SITL with different options. After running, close all processes cleanly using Ctrl+C:

```bash
# Basic ArduCopter simulation
cd ~/ardupilot
./sim_vehicle.py -v ArduCopter -w

# ArduCopter with console and map
./sim_vehicle.py -v ArduCopter --console --map

# ArduCopter at San Francisco International Airport
./sim_vehicle.py -v ArduCopter -L KSFO --console --map


# ArduCopter in quadcopter configuration with console, map, and OSD
./sim_vehicle.py -v ArduCopter -f quadcopter --console --map --osd
```

### 🤖 Testing ROS2 with SITL

```bash
# Launch ROS2 with SITL using DDS over UDP
cd ~/ros2_ws
source install/setup.bash
ros2 launch ardupilot_sitl sitl_dds_udp.launch.py transport:=udp4 refs:=$(ros2 pkg prefix ardupilot_sitl)/share/ardupilot_sitl/config/dds_xrce_profile.xml synthetic_clock:=True wipe:=False model:=quad speedup:=1 slave:=0 instance:=0 defaults:=$(ros2 pkg prefix ardupilot_sitl)/share/ardupilot_sitl/config/default_params/copter.parm,$(ros2 pkg prefix ardupilot_sitl)/share/ardupilot_sitl/config/default_params/dds_udp.parm sim_address:=127.0.0.1 master:=tcp:127.0.0.1:5760 sitl:=127.0.0.1:5501
```

### 🌐 Final Simulation (Multi-Terminal)

Terminal 1 (Launch the container with GPU support):
```bash
cd ~/ros2_ardupilot_SITL
./run.sh
```
Terminal 1 (Launch the container without GPU support):
```bash
cd ~/ros2_ardupilot_SITL
./run_without_gpu.sh
```

Terminal 2 (Connect to the running container):
```bash
docker exec -it ardupilot_sitl /bin/bash
mavproxy.py --console --map --aircraft test --master=:14550
```

## 🐳 Container Environment

The Docker container includes:

- 🤖 ROS2 Humble Desktop
- 🌎 Gazebo Harmonic
- 🚁 ArduPilot source code with SITL capabilities
- 📡 MAVProxy
- 🔄 Micro-XRCE-DDS-Gen for DDS communication
- 📦 All ROS2 packages needed for ArduPilot-ROS2 integration
- 🔌 ROS2-Gazebo bridges and plugins

## 💡 Development Tips

1. **Source the workspace**: Always remember to source the workspace setup file before running ROS2 commands:
   ```bash
   source ~/ros2_ws/install/setup.bash
   ```

2. **Customizing ArduPilot parameters**: You can modify the default parameters in the `ros2_ws/src/ardupilot_sitl/config/default_params/` directory.

3. **Using tmuxinator**: A tmuxinator configuration is included for managing multiple terminal sessions:
   ```bash
   tmuxinator start -p ~/tmuxinator.yml
   ```

4. **Debugging**: To debug ROS2 nodes, you can use:
   ```bash
   ros2 run --prefix 'gdb -ex run --args' package_name node_name
   ```

## ⚙️ Customization

You can modify the `Dockerfile` to add additional dependencies or change the build configuration.

## 🔧 Troubleshooting

- **❌ X11 Display Issues**: If GUI applications don't appear, check that your X11 server is properly configured. You may need to run `xhost +local:docker` on the host.
  
- **⚠️ Permission Issues**: Ensure the USER_UID and USER_GID in the docker-compose.yml match your host system:
  ```bash
  echo "UID: $(id -u), GID: $(id -g)"
  ```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [🚁 ArduPilot](https://ardupilot.org/) - ArduPilot project
- [🤖 ROS2](https://docs.ros.org/en/humble/) - ROS2 Humble documentation
- [🌎 Gazebo](https://gazebosim.org/) - Gazebo simulation platform
- [🔌 ArduPilot-Gazebo-ROS2 Integration](https://github.com/ArduPilot/ardupilot_gz) - ArduPilot Gazebo integration
