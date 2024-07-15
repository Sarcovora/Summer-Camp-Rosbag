import os
out = os.popen('rosbag info data/saved_demo_bags/demo_labmapworks_2024-07-09__16_13_54.bag').read()
print(out)

