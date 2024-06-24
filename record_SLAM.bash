#!/usr/bin/env bash
NOW=$( date '+%F_%H:%M:%S' )
# roscore >/dev/null 2>&1 &
trap 'rosnode kill --all & exit' SIGINT
rosparam set use_sim_time false
roslaunch realsense2_camera opensource_tracking.launch > /dev/null 2>&1 &
echo "Press Enter to start recording"
read -n 1 -s
echo "Press [CTRL+C] to stop recording"
rosbag record -O SLAM_"$NOW".bag -b 0  /camera/aligned_depth_to_color/camera_info  /camera/aligned_depth_to_color/image_raw/compressedDepth /camera/color/camera_info /camera/color/image_raw/compressed /camera/imu /camera/imu_info /tf_static /tf