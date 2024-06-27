# roscore >/dev/null 2>&1 &
source /catkin_ws/devel/setup.bash
rosparam set use_sim_time true
roslaunch realsense2_camera opensource_tracking.launch offline:=true > /dev/null 2>&1 &
echo "Running image_transport"
rosrun image_transport republish compressed in:=/camera/color/image_raw raw out:=/camera/color/image_raw &
rosrun image_transport republish compressedDepth in:=/camera/aligned_depth_to_color/image_raw raw out:=/camera/aligned_depth_to_color/image_raw &
echo "Press Ctrl^C to exit"
trap 'rosnode kill --all' SIGINT
rosbag play "$1" --rate 0.5 --clock
sleep 3000h