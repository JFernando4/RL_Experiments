import numpy as np


class Buffer:
    def __init__(self, buffer_size, dimensions):
        self.buffer_size = buffer_size
        n1, n2, d0, f1, d1, f2, d2, m1, m2 = dimensions
        self.buffer_frames = np.zeros(shape=[buffer_size, n1, n2, d0], dtype=float)
        self.buffer_actions = np.zeros(shape=[buffer_size, m2], dtype=int)
        self.buffer_labels = np.zeros(shape=[buffer_size, m1], dtype=float)
        self.buffer_priority = np.zeros(shape=buffer_size, dtype=int)
        self.current_buffer_size = 0

    def add_to_buffer(self, buffer_entry):
        if self.buffer_size > self.current_buffer_size:
            self.buffer_frames[self.current_buffer_size] = buffer_entry[0]
            self.buffer_actions[self.current_buffer_size] = buffer_entry[1]
            self.buffer_labels[self.current_buffer_size] = buffer_entry[2]
            self.buffer_priority[self.current_buffer_size] = buffer_entry[3]
            self.current_buffer_size += 1
        else:
            weights = softmax(-self.buffer_priority)
            random_entry = np.random.choice(np.arange(start=0, stop=self.buffer_size, dtype=int), p=weights)
            self.buffer_frames[random_entry] = buffer_entry[0]
            self.buffer_actions[random_entry] = buffer_entry[1]
            self.buffer_labels[random_entry] = buffer_entry[2]
            self.buffer_priority[random_entry] = buffer_entry[3]

    def sample(self, batch_size):
        if self.current_buffer_size < batch_size:
            raise ValueError("Not enough entries in the buffer.")

        if self.buffer_size > self.current_buffer_size:
            weights = softmax(self.buffer_priority[0:(self.current_buffer_size)])
            sample_indices = np.random.choice(np.arange(start=0, stop=self.current_buffer_size, dtype=int),
                                              size=batch_size, p=weights, replace=False)
        else:
            weights = softmax(self.buffer_priority)
            sample_indices = np.random.choice(np.arange(start=0, stop=self.buffer_size, dtype=int),
                                              size=batch_size, p=weights, replace=False)

        sample_frames = self.buffer_frames[sample_indices]
        sample_actions = self.buffer_actions[sample_indices]
        sample_labels = self.buffer_labels[sample_indices]

        return sample_frames, sample_actions, sample_labels


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
