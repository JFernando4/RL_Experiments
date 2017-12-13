import numpy as np


class Buffer:
    def __init__(self, buffer_size, observation_dimensions):
        self.buffer_size = buffer_size
        dim = [buffer_size]
        dim.extend(observation_dimensions)
        self.buffer_frames = np.zeros(shape=dim, dtype=float)
        self.buffer_actions = np.zeros(shape=[buffer_size, 1], dtype=int)
        self.buffer_labels = np.zeros(shape=[buffer_size], dtype=float)
        self.buffer_isampling = np.zeros(shape=[buffer_size], dtype=float)  # importance sampling
        self.current_buffer_size = 0

    def add_to_buffer(self, buffer_entry):
        if self.buffer_size > self.current_buffer_size:
            self.buffer_frames[self.current_buffer_size] = buffer_entry[0]
            self.buffer_actions[self.current_buffer_size] = buffer_entry[1]
            self.buffer_labels[self.current_buffer_size] = buffer_entry[2]
            self.buffer_isampling[self.current_buffer_size] = buffer_entry[3]
            self.current_buffer_size += 1
        else:
            random_entry = np.random.choice(self.buffer_size, replace=False)
            self.buffer_frames[random_entry] = buffer_entry[0]
            self.buffer_actions[random_entry] = buffer_entry[1]
            self.buffer_labels[random_entry] = buffer_entry[2]
            self.buffer_isampling[random_entry] = buffer_entry[3]

    def sample(self, batch_size):
        if self.current_buffer_size < batch_size:
            raise ValueError("Not enough entries in the buffer.")

        if self.buffer_size > self.current_buffer_size:
            sample_indices = np.random.choice(self.current_buffer_size, size=batch_size, replace=False)
        else:
            sample_indices = np.random.choice(self.buffer_size, size=batch_size, replace=False)

        sample_frames = self.buffer_frames[sample_indices]
        sample_actions = self.buffer_actions[sample_indices]
        sample_labels = self.buffer_labels[sample_indices]
        sample_isampling = self.buffer_isampling[sample_indices]

        return sample_frames, sample_actions, sample_labels, sample_isampling
