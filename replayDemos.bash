# roscore >/dev/null 2>&1 &
source /catkin_ws/devel/setup.bash
rosparam set use_sim_time true
roslaunch ./launch/realsense_load_from_map.launch database_path:=/home/abba/Summer-Camp-Rosbag/maps/recreated_maps/labroombetter.db offline:=true > /dev/null 2>&1 &
echo "Running image_transport"
rosrun image_transport republish compressed in:=/camera/color/image_raw raw out:=/camera/color/image_raw &

rosbag record -O recreatedDemos.bag -b 0 /camera/aligned_depth_to_color/camera_info /camera/aligned_depth_to_color/image_raw /camera/aligned_depth_to_color/image_raw/compressedDepth /camera/color/camera_info /camera/color/image_raw/compressed /camera/imu /camera/gyro/imu_info /camera/accel/imu_info /tf_static /tf /bariflex /bariflex_motion > /dev/null 2>&1 &

echo "Press Ctrl^C to exit"
trap 'rosnode kill --all' SIGINT

rosbag play "$1" --rate 0.5 --clock
sleep 3000h