#!/bin/bash

# Init constants
mainpath="workingdir" # TODO change this to the actual absolute path of the folder containing the groupnames
groups=("group1" "group3" "group4" "group5")
path1="Summer-Camp-Rosbag/data/sample_demo_bags" # TODO check if this path (starting from Summer-Camp-Rosbag) is correct or not
path2="Summer-Camp-Rosbag/data/sample_demos" # TODO check if this path (starting from Summer-Camp-Rosbag) is correct or not

directories=()
for group in "${groups[@]}"; do
    echo "$group"
    directories+=("${mainpath}/${group}/${path1}")
    directories+=("${mainpath}/${group}/${path2}")
done

for directory in "${directories}"; do
    for file in "${directory}"/*; do
        if [[ -f "$file" && "$file" == *.bag ]]; then
            rosinf="$(rosbag info "$file")"
            echo "$rosinf" >| "junk.txt"
            bariflex=$(grep -q "/bariflex" "junk.txt")
            camera=$(grep -q "/camera/aligned_depth_to_color/image_raw/compressed" "junk.txt")

            if [[ -n "$bariflex" && -n "$camera" ]]; then
                echo "$file"
                echo "$rosinf"
            fi
        fi
rm junk.txt