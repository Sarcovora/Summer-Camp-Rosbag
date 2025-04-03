cd ~/hsra_data/group1_final
cp ~/Summer-Camp-Rosbag/call_this_bash.py ~/Summer-Camp-Rosbag/generatehdf5.bash .
echo ${PWD}
rm -f rm data/rosbag.hdf5
./generatehdf5.bash data/saved_demo_bags/


cd ~/hsra_data/group4_final
cp ~/Summer-Camp-Rosbag/call_this_bash.py ~/Summer-Camp-Rosbag/generatehdf5.bash .
echo ${PWD}
rm -f rm data/rosbag.hdf5
./generatehdf5.bash data/saved_demo_bags/


cd ~/hsra_data/group6_final/ 
cp ~/Summer-Camp-Rosbag/call_this_bash.py ~/Summer-Camp-Rosbag/generatehdf5.bash .
echo ${PWD}
rm -f rm data/rosbag.hdf5
./generatehdf5.bash data/saved_demo_bags/

conda activate hsra
cd ~/Summer-Camp-Rosbag/scripts
python robomimic_train.py --config ../config/bariflex_bc.json
# python robomimic_train.py --config ../config/group2.json
# python robomimic_train.py --config ../config/group3.json
python robomimic_train.py --config ../config/group4.json
# python robomimic_train.py --config ../config/group5.json
python robomimic_train.py --config ../config/group6.json