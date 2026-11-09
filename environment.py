"""Creates the Pong environment with the standard DQN preprocessing.

We all use this function so that every experiment gets the same inputs.
"""

import ale_py
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, TransformReward

ENV_ID = "ALE/Pong-v5"

# Make the ALE Atari games available to gym.make()
gym.register_envs(ale_py)


def make_env(seed, render_mode=None):
    """Return a Pong environment whose observations are 4 stacked 84x84 grayscale frames.

    render_mode=None for training, "human" to open a game window (play.py).
    """
    # Pong-v5 already skips 4 frames by default. We set frameskip=1 here
    # so that AtariPreprocessing does the skipping, otherwise frames are skipped twice.
    env = gym.make(ENV_ID, frameskip=1, render_mode=render_mode)

    # Grayscale, resize to 84x84, repeat each action for 4 frames,
    # and start each episode with a random number of no-op actions.
    env = AtariPreprocessing(env, frame_skip=4, screen_size=84, grayscale_obs=True, noop_max=30)

    # Clip rewards to -1, 0 or +1 (Pong rewards are already in this range,
    # but we keep it so the setup matches the standard DQN recipe).
    env = TransformReward(env, lambda reward: float(np.sign(reward)))

    # Stack the last 4 frames so the network can see which way the ball is moving.
    # Observation shape: (4, 84, 84), dtype uint8.
    env = FrameStackObservation(env, stack_size=4)

    # Seed the environment and the action space so runs are repeatable.
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env
