# Quick Reference Key

- `99-realsense-libusb.rules` & `init.bash` are both called by `setup.bash` so no need to worry about those.
- Run `setup.bash` to setup the system
- Run `rosnode kill --all` to kill any remaining background processes if crtl-c did not work
- `record_SLAM.bash` and `replay_SLAM.bash` record and replay respectively. Press enter for record to start recording, replay starts as soon as script is run. Kill both with crtl-c
