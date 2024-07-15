import h5py
import json
import numpy as np

# Create a new HDF5 file
with h5py.File('dataset.hdf5', 'w') as f:
    # Create the data group
    data_group = f.create_group('data')
    
    # Add the 'total' attribute
    data_group.attrs['total'] = 400
    
    # Add the 'env_args' attribute
    env_args = {
        'env_name': 'ExampleEnv',
        'env_type': 'robosuite',
        'env_kwargs': {
            'arg1': 'value1',
            'arg2': 'value2'
        }
    }
    data_group.attrs['env_args'] = json.dumps(env_args)
    
    # Add datasets for the first trajectory
    N = 100  # Length of the trajectory
    D = 10   # Dimension of the state vector
    A = 4    # Dimension of the action space

    for i in range(4):
        # Add the first trajectory group 'demo_0'
        demo_0 = data_group.create_group(f'demo_{i}')
        demo_0.attrs['num_samples'] = N  # Number of state-action samples in this trajectory (example value)
    
        # demo_0.create_dataset('states', data=np.random.random((N, D)))
        demo_0.create_dataset('actions', data=np.random.random((N, A)))
        
        # Create the 'obs' group within 'demo_0'
        obs = demo_0.create_group('obs')
        obs.create_dataset('image_color', data=np.random.random((N, 84, 84, 3)))
        obs.create_dataset('image_depth', data=np.random.random((N, 84, 84, 1)))
        obs.create_dataset('bariflex', data=np.random.random((N,1)))
    
    # # Add the mask group
    # mask = data_group.create_group('mask')
    # mask.create_dataset('valid', data=np.array(['demo_0', 'demo_1'], dtype=h5py.special_dtype(vlen=str)))

print("HDF5 dataset created successfully.")
