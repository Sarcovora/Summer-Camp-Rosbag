#!/usr/bin/env bash
NOW=$( date '+%F_%H-%M-%S' )
# roscore >/dev/null 2>&1 &
trap 'rosnode kill --all && exit' SIGINT
rosparam set use_sim_time false
roslaunch realsense2_camera opensource_tracking_SLAMless.launch > /dev/null 2>&1 &
echo "Press [CTRL+C] to exit"
sleep 3000h