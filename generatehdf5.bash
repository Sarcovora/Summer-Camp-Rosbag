# Usage: ./generatehdf5.bash /path/to/directory

if [ -z "$1" ]; then
    echo "Please provide a directory containing the bag files."
    exit 1
fi

# roscore >/dev/null 2>&1 &
source /catkin_ws/devel/setup.bash
rosparam set use_sim_time true

for file in "$1"/*; do
    if [ -f "$file" ]; then
        echo "Processing $file"

        if command -v python3 &>/dev/null; then
            PYTHON_CMD=python3
        elif command -v python &>/dev/null; then
            PYTHON_CMD=python
        else
            echo "Python does not appear to be installed on this system."
            exit 1
        fi

        $PYTHON_CMD call_this_bash.py "$file" &
        PYTHON_PID=$!


        echo "Press Ctrl^C to exit"
        trap 'rosnode kill --all' SIGINT

        # Get the duration of the bag file in seconds
        DURATION=$(rosbag info --yaml "$file" | grep duration | awk '{print $2}')
        
        SLEEP_TIME=$(echo "$DURATION / 0.5" | bc)
        
        sleep 2
        echo "about to play"
        rosbag play "$file" --rate 0.5 --clock &
        ROSBAG_PID=$!
        # Sleep for the calculated time
        sleep $SLEEP_TIME

        kill $ROSBAG_PID
        kill $PYTHON_PID
        pkill -f call_this_bash.py

    else
        echo "$file is not a regular file, skipping..."
    fi
done
