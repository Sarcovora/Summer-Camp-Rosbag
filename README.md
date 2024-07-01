# What files do I need to run?

- To get setup, run `./setup/setup.bash`
- To make a recording with **JUST** the camera, run `record_SLAM.py`
- To make a recording with **BOTH** the camera and the Arduino for the BaRiflex, run `record_SLAM_Gripper.py`
- To replay the bag in Rviz, run `replay_SLAM.bash`
- To parse the bag without playing it, run `nobagplayparsing.py`

## Extra notes

- Run `rosnode kill --all` to kill any remaining background processes if crtl-c did not work
