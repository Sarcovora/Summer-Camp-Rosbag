if [ "$EUID" -ne 0 ]
  then echo "Please run as root"
  exit
fi
DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -yq --no-install-recommends
apt-get install -y usbutils
apt-get install -y ros-noetic-realsense2-camera
apt-get install -y curl

mkdir -p /etc/apt/keyrings

apt-key adv --keyserver keyserver.ubuntu.com --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
apt-get -y install software-properties-common
add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u
sudo apt-get update --fix-missing 
apt-get -y install librealsense2-utils

mkdir -p /catkin_ws/src

mkdir -p /etc/udev/rules.d/
cp 99-realsense-libusb.rules /etc/udev/rules.d/
cd /catkin_ws/src || exit
catkin_create_pkg collect_data_pkg std_msgs rospy roscpp sensor_msgs realsense2_camera
mkdir -p /catkin_ws/src/collect_data_pkg/launch

# ADD collect_data.py /catkin_ws/src/collect_data_pkg/src
echo "source /catkin_ws/devel/setup.bash" >> ~/.bashrc
./init.bash
mkdir -p /data
sudo apt-get install -y ros-noetic-imu-filter-madgwick
sudo apt-get install -y ros-noetic-robot-localization
sudo apt-get install -y ros-noetic-rtabmap-ros
sudo apt-get install -y ros-noetic-octomap-rviz-plugins
sudo apt-get install -y ros-noetic-compressed-image-transport
sudo apt-get install -y ros-noetic-compressed-depth-image-transport
pip install opencv-python
cp opensource_tracking.launch /opt/ros/noetic/share/realsense2_camera/launch/

# COPY opensource_tracking.launch /opt/ros/noetic/share/realsense2_camera/launch/
# COPY rs_camera.launch /opt/ros/noetic/share/realsense2_camera/launch/
# WORKDIR /data
