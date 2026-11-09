"""Experience replay buffer: stores past transitions and returns random batches."""

import numpy as np


class ReplayBuffer:
    """Fixed-size buffer. Once it is full, new transitions overwrite the oldest ones.

    Frames are stored as uint8 (1 byte per pixel) instead of float32 (4 bytes),
    which uses 4x less RAM. Each transition stores two (4, 84, 84) states,
    so 100,000 transitions take about 5.6 GB.
    """

    def __init__(self, capacity, state_shape=(4, 84, 84)):
        self.capacity = capacity
        self.position = 0  # where the next transition is written
        self.size = 0  # how many transitions are stored so far

        # One array per part of the transition. (capacity, *state_shape)
        # is the same as (capacity, 4, 84, 84).
        self.states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        """Store one transition (s, a, r, s', done)."""
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done

        # Move to the next slot, wrapping around to 0 when the end is reached.
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """Return a random batch as numpy arrays: states, actions, rewards, next_states, dones.

        Random sampling breaks the correlation between consecutive frames,
        which makes training more stable.
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def __len__(self):
        # Lets us write len(buffer) to get the number of stored transitions
        return self.size
