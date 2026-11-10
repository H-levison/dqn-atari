"""Train a DQN agent on ALE/Pong-v5.

The environment, the network and the replay buffer live in their own modules.
This file holds the parts that make them learn: action selection, the Bellman
update, the training loop and the CSV log.

Run one experiment:
    uv run train.py --lr 1e-4 --gamma 0.99 --out experiments/Anthony/exp01

Each stub keeps the signature the group agreed on. Other people's code calls
these, so do not change a signature without telling them.
"""

import argparse
import csv
import random
from pathlib import Path

import torch

from environment import make_env
from network import DQN
from replay_buffer import ReplayBuffer

# plot_results.py reads these columns, in this order. Do not rename them.
LOG_COLUMNS = ["episode", "step", "reward", "length", "epsilon", "loss"]


# --- Anthony: networks, learning step, training loop -----------------------

def build_networks(n_actions, device):
    """Create the online network and a target network holding the same weights.

    Returns (online_net, target_net).
    """
    # TODO (Anthony)
    raise NotImplementedError


def update_target(online_net, target_net):
    """Copy the online weights into the target network."""
    # TODO (Anthony)
    raise NotImplementedError


def learn(online_net, target_net, buffer, optimizer, batch_size, gamma, device):
    """Run one gradient step on a sampled batch.

    Predicts Q-values with the online network, builds the Bellman target
    r + gamma * max Q_target(next_state) for non-terminal transitions,
    computes the loss, backpropagates and steps the optimizer.

    Returns the loss as a float.
    """
    # TODO (Anthony)
    raise NotImplementedError


def train(args):
    """Run the training loop until the step budget is reached.

    Resets the environment, selects an action, stores the transition, learns
    once the buffer has warmed up, decays epsilon, syncs the target network
    periodically, writes one CSV row per episode, and saves the model.

    The pieces you need are already here:
        get_epsilon(step, args.eps_start, args.eps_end, args.eps_decay)
        select_action(state, online_net, epsilon, n_actions, device)
        log_file, writer = start_log(args.out)   # one row per episode
        writer.writerow([episode, step, reward, length, epsilon, loss])
        torch.save(online_net.state_dict(), Path(args.out) / "model.pth")
    """
    # TODO (Anthony)
    raise NotImplementedError


# --- HonourGod: action selection, epsilon decay, CLI, logging --------------

def get_epsilon(step, eps_start, eps_end, eps_decay):
    """Return the exploration rate at this step, decaying in a straight line.

    Epsilon falls from eps_start to eps_end over the first eps_decay steps and
    stays at eps_end afterwards. Early on the agent explores almost at random;
    later it mostly trusts its own Q-values.
    """
    fraction = min(step / eps_decay, 1.0)
    # Written this way round so that the result is exactly eps_end once the
    # decay is finished, rather than a floating-point value just beside it.
    return eps_end + (1.0 - fraction) * (eps_start - eps_end)


def select_action(state, online_net, epsilon, n_actions, device):
    """Epsilon-greedy: explore with probability epsilon, otherwise act greedily."""
    if random.random() < epsilon:
        return random.randrange(n_actions)

    # No gradients here: we are choosing an action, not learning from it.
    with torch.no_grad():
        # The network expects a batch, so add a leading dimension of 1.
        state_tensor = torch.as_tensor(state, device=device).unsqueeze(0)
        q_values = online_net(state_tensor)
    return int(q_values.argmax(dim=1).item())


def start_log(out_dir):
    """Create the run folder and open its CSV, writing the header row.

    Returns (file, writer). The caller writes one row per episode with
    writer.writerow([...]) in LOG_COLUMNS order, and closes the file at the end.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(out_dir) / "log.csv", "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(LOG_COLUMNS)
    return log_file, writer


def parse_args():
    """Read the hyperparameters for one experiment from the command line."""
    parser = argparse.ArgumentParser(description="Train a DQN agent on Pong.")

    # The six hyperparameters the experiment table asks about.
    parser.add_argument("--lr", type=float, default=1e-4, help="optimizer learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=32, help="transitions per update")
    parser.add_argument("--eps-start", type=float, default=1.0, help="starting exploration rate")
    parser.add_argument("--eps-end", type=float, default=0.01, help="final exploration rate")
    parser.add_argument("--eps-decay", type=int, default=200_000, help="steps to reach eps-end")

    # Fixed across experiments so results stay comparable. Change together.
    parser.add_argument("--steps", type=int, default=400_000, help="total environment steps")
    parser.add_argument("--buffer-size", type=int, default=50_000, help="replay buffer capacity")
    parser.add_argument("--warmup", type=int, default=10_000, help="random steps before learning")
    parser.add_argument("--learn-every", type=int, default=4, help="steps between updates")
    parser.add_argument("--target-sync", type=int, default=1_000, help="steps between target copies")
    parser.add_argument("--seed", type=int, default=0, help="random seed")

    parser.add_argument("--out", required=True, help="run folder, e.g. experiments/HonourGod/exp01")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
