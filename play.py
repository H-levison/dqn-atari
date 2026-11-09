"""Load the trained DQN and watch it play Pong in real time.

Usage:
    uv run play.py
    uv run play.py --model Alliane/exp03.pth --episodes 5
"""

import argparse

import torch

from environment import make_env
from network import DQN


def parse_args():
    parser = argparse.ArgumentParser(description="Watch a trained DQN agent play Pong.")
    parser.add_argument("--model", default="dqn_model.pth", help="path to the saved model weights")
    parser.add_argument("--episodes", type=int, default=3, help="number of games to play")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_model(model_path, n_actions):
    """Rebuild the same network used in training and load the saved weights into it."""
    model = DQN(n_actions)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()  # we are only playing, not training
    return model


def choose_greedy_action(model, state):
    """Always pick the action with the highest predicted Q-value (no exploration)."""
    with torch.no_grad():
        state_tensor = torch.as_tensor(state).unsqueeze(0)  # add a batch dimension
        q_values = model(state_tensor)
    return int(q_values.argmax(dim=1).item())


def main():
    args = parse_args()

    # render_mode="human" opens a window that shows the game in real time.
    env = make_env(args.seed, render_mode="human")
    model = load_model(args.model, env.action_space.n)

    for episode in range(1, args.episodes + 1):
        state, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = choose_greedy_action(model, state)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        # Pong score: +21 means the agent won every point, -21 means it lost every point.
        print(f"Episode {episode}: total reward = {total_reward:+.0f}")

    env.close()


if __name__ == "__main__":
    main()
