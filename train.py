"""Train a DQN agent on ALE/Pong-v5.

The environment, the network and the replay buffer live in their own modules.
This file holds the parts that make them learn: action selection, the Bellman
update, the training loop and the CSV log.

Run one experiment:
    uv run train.py --lr 1e-4 --gamma 0.99 --out experiments/Anthony/exp01

Each stub keeps the signature the group agreed on. Other people's code calls
these, so do not change a signature without telling them.
"""

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
    """
    # TODO (Anthony)
    raise NotImplementedError


# --- HonourGod: action selection, epsilon decay, CLI, logging --------------

def get_epsilon(step, eps_start, eps_end, eps_decay):
    """Return the exploration rate at this step (linear decay)."""
    # TODO (HonourGod)
    raise NotImplementedError


def select_action(state, online_net, epsilon, n_actions, device):
    """Pick a random action with probability epsilon, else the greedy one."""
    # TODO (HonourGod)
    raise NotImplementedError


def parse_args():
    """Read the hyperparameters for one experiment from the command line."""
    # TODO (HonourGod): --lr --gamma --batch-size --eps-start --eps-end
    #                     --eps-decay --steps --buffer-size --target-sync
    #                     --seed --out
    raise NotImplementedError


if __name__ == "__main__":
    train(parse_args())
