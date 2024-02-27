import d3rlpy
#import gym
import numpy as np
import pickle

def replace_cos_sin_with_angle(arr):
    # Determine the number of pairs of cos and sin values
    num_pairs = (arr.shape[1]) // 6
    
    # Initialize a list to collect the indices of columns to be removed
    cols_to_remove = []
    
    # Loop through each pair to calculate the angle and identify columns to remove
    for n in range(num_pairs):
        cos_col = 4 + 6*n
        sin_col = 5 + 6*n
        angle = np.arctan2(arr[:, sin_col], arr[:, cos_col])
        
        # Replace the cos values with the calculated angle
        arr[:, cos_col] = angle
        
        # Mark the sin column for removal
        cols_to_remove.append(sin_col)
    
    # Remove the marked columns
    arr2= np.delete(arr, cols_to_remove, axis=1)
    assert arr2.shape == (arr.shape[0], arr.shape[1] - num_pairs), (arr2.shape, (arr.shape[0], arr.shape[1] - num_pairs))
    return arr2

# Step 1: Load SB3 Replay Buffer Data from disk
with open('d3rlpy_data/sac_parking_replay_buffer_2_cars_any_car_termination.pkl', 'rb') as f:
    replay_buffer = pickle.load(f)

observations = np.concatenate([replay_buffer.observations["observation"].reshape(replay_buffer.buffer_size, -1), replay_buffer.observations["desired_goal"].reshape(replay_buffer.buffer_size, -1)], axis=1)

replay_buffer.observations = replace_cos_sin_with_angle(observations)

print(observations.shape, replay_buffer.observations.shape, replay_buffer.actions.shape)
# Convert to d3rlpy MDPDataset
#dataset = to_mdp_dataset(replay_buffer)

terminals_combined = np.logical_or(replay_buffer.timeouts, replay_buffer.dones)

dataset = d3rlpy.dataset.MDPDataset(
        observations=replay_buffer.observations,
        actions=replay_buffer.actions.reshape(replay_buffer.buffer_size, -1),
        rewards=replay_buffer.rewards,
        terminals=terminals_combined,
        #episode_terminals=replay_buffer.timeouts
    )

# save MDPDataset
dataset.dump('d3rlpy_data/sac_parking_replay_buffer_2_cars_any_car_termination_d3rlpy_angle.h5')
