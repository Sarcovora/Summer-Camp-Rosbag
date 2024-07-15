import h5py
import json
import numpy as np

# Create a new HDF5 file
with h5py.File('dataset.hdf5', 'w') as f:
    # Create the data group
    data_group = f.create_group('data')
    
    # Add the 'total' attribute
    data_group.attrs['total'] = 2  # Number of state-action samples in the dataset (example value)
    
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
    
    # Add the first trajectory group 'demo_0'
    demo_0 = data_group.create_group('demo_0')
    demo_0.attrs['num_samples'] = 100  # Number of state-action samples in this trajectory (example value)
    demo_0.attrs['model_file'] = '<xml_string>'  # Example MJCF MuJoCo model
    
    # Add datasets for the first trajectory
    N = 100  # Length of the trajectory
    D = 10   # Dimension of the state vector
    A = 4    # Dimension of the action space
    
    demo_0.create_dataset('states', data=np.random.random((N, D)))
    demo_0.create_dataset('actions', data=np.random.random((N, A)))
    demo_0.create_dataset('rewards', data=np.random.random(N))
    demo_0.create_dataset('dones', data=np.random.randint(0, 2, N))
    
    # Create the 'obs' group within 'demo_0'
    obs = demo_0.create_group('obs')
    obs.create_dataset('agentview_image', data=np.random.random((N, 84, 84, 3)))
    
    # Create the 'next_obs' group within 'demo_0'
    next_obs = demo_0.create_group('next_obs')
    next_obs.create_dataset('agentview_image', data=np.random.random((N, 84, 84, 3)))
    
    # Add the second trajectory group 'demo_1'
    demo_1 = data_group.create_group('demo_1')
    demo_1.attrs['num_samples'] = 80  # Number of state-action samples in this trajectory (example value)
    
    # Add datasets for the second trajectory
    N = 80  # Length of the trajectory
    
    demo_1.create_dataset('states', data=np.random.random((N, D)))
    demo_1.create_dataset('actions', data=np.random.random((N, A)))
    demo_1.create_dataset('rewards', data=np.random.random(N))
    demo_1.create_dataset('dones', data=np.random.randint(0, 2, N))
    
    # Create the 'obs' group within 'demo_1'
    obs = demo_1.create_group('obs')
    obs.create_dataset('agentview_image', data=np.random.random((N, 84, 84, 3)))
    
    # Create the 'next_obs' group within 'demo_1'
    next_obs = demo_1.create_group('next_obs')
    next_obs.create_dataset('agentview_image', data=np.random.random((N, 84, 84, 3)))
    
    # Add the mask group
    mask = data_group.create_group('mask')
    mask.create_dataset('valid', data=np.array(['demo_0', 'demo_1'], dtype=h5py.special_dtype(vlen=str)))

print("HDF5 dataset created successfully.")
