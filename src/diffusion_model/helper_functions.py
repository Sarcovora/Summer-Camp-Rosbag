import h5py
import numpy as np

def flatten_states_and_actions(states_or_actions):
    flattened_data = []
    for entry in states_or_actions:
        position = np.array(entry['position']).flatten()
        orientation = np.array(entry['orientation']).flatten()
    flattened_data.append(np.concatenate((position, orientation)))
    return np.array(flattened_data)


def convert_hdf5(input_file, output_file):
    with h5py.File(input_file, 'r') as f_in:
        num_demonstrations = f_in['data'].attrs['num_demonstrations']
        with h5py.File(output_file, 'w') as f_out:
            f_out.attrs['date'] = ""
            f_out.attrs['time'] = ""
            f_out.attrs['repository_version'] = None
            f_out.attrs['env'] = None

            for i in range(num_demonstrations):
                demo_group = f_in['data/demo_{}'.format(i)]
                demo_out = f_out['data'].create_group(demo_name)

                demo_out = f_out.create_group(demo_name)
                demo_out.attrs['model_file'] = None
                demo_out.create_dataset('states', data=flatten_states_and_actions(demo_group['states']))
                demo_out.create_dataset('actions', data=flatten_states_and_actions(demo_group['action']))
                demo_out.create_dataset('observations', data=demo_group['observation'][()])
                demo_out.create_dataset('timestamps', data=demo_group['time_stamps'][()])

                demo_out.attrs['numsamples'] = demo_group.attrs['num_samples']
                demo_out.attrs['description'] = demo_group.attrs['description']

# def get_into_dataloader_format():
def get_into_dataloader_format(input_file, output_file=None):
    # Open the input HDF5 file
    data_list = []
    position_stats = {"min" : 1000, "max": -1000}
    orientation_stats = {"min" : 1000, "max": -1000}

    with h5py.File(input_file, 'r') as f:
        # demo_data = []
        for demo_number in f.keys():
            demo = f[demo_number]
            num_samples = demo.attrs['num_samples']
            # print(type(demo))
            # print(demo_number, num_samples)

            # Iterate through each demonstration group
            # for i in range(num_demonstrations):
            #     demo_group_name = f'demo_{i}'
            #     demo_group = f[demo_group_name]

                # num_samples = demo_group.attrs['num_samples']
            demo_data = []

                # Iterate through each sample in the demonstration
            # num_samples = f[demo].attrs['num_samples']
            for j in range(num_samples):
                sample_data = {}

                # Extract position and orientation from states dataset
                sample_data['position'] = np.array(demo['states'][j]['position'])
                sample_data['orientation'] = np.array(demo['states'][j]['orientation'])

                sample_data['position'] = np.expand_dims(sample_data['position'], axis=0)
                sample_data['orientation'] = np.expand_dims(sample_data['orientation'], axis=0)

                # Extract image from observation dataset
                sample_data['image'] = np.transpose(np.array(demo['observations'][j]), (2, 0, 1))

                sample_data['min_position'] = np.min(sample_data['position'])
                sample_data['max_position'] = np.max(sample_data['position'])

                sample_data['min_orientation'] = np.min(sample_data['orientation'])
                sample_data['max_orientation'] = np.max(sample_data['orientation'])

                position_stats["min"] = np.minimum(position_stats["min"], sample_data["min_position"])
                position_stats["max"] = np.maximum(position_stats["max"], sample_data["max_position"])

                orientation_stats["min"] = np.minimum(orientation_stats["min"], sample_data["min_orientation"])
                orientation_stats["max"] = np.maximum(orientation_stats["max"], sample_data["max_orientation"])
                demo_data.append(sample_data)
            # print(demo_data[0:2]['position'], demo_data[0:2]['orientation'])
            # print(demo_data[0])
            # # print(type(demo_data[0]['position']))
            # # print(type(demo_data[0]))
            # # print()
            # print(sample_data['position'].shape)
            # print(sample_data['orientation'].shape)
            # print(sample_data['image'].shape)

            data_list.append(demo_data)
    # print(position_stats)
    # print(orientation_stats)
    return (data_list, position_stats, orientation_stats)


    # uncomment to create hdf5 file
    # with h5py.File(output_file, 'w') as f:
    #     f.create_dataset('data', data=data_list)

if __name__ == "__main__":
    input_file = "test_1.hdf5"
    output_file = "output.h5"
    get_into_dataloader_format(input_file, output_file)
