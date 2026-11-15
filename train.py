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
    # The online network learns; the target network provides stable targets.
    online_net = DQN(n_actions).to(device)
    target_net = DQN(n_actions).to(device)
    update_target(online_net, target_net)
    target_net.eval()
    return online_net, target_net


def update_target(online_net, target_net):
    """Copy the online weights into the target network."""
    # Keep the target network synchronized at fixed intervals.
    target_net.load_state_dict(online_net.state_dict())


def learn(online_net, target_net, buffer, optimizer, batch_size, gamma, device):
    """Run one gradient step on a sampled batch.

    Predicts Q-values with the online network, builds the Bellman target
    r + gamma * max Q_target(next_state) for non-terminal transitions,
    computes the loss, backpropagates and steps the optimizer.

    Returns the loss as a float.
    """
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)

    states = torch.as_tensor(states, device=device)
    actions = torch.as_tensor(actions, dtype=torch.long, device=device)
    rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    next_states = torch.as_tensor(next_states, device=device)
    dones = torch.as_tensor(dones, dtype=torch.float32, device=device)

    # Select the Q-value for the action taken in each transition.
    predicted_q = online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Do not calculate gradients through the target network.
    with torch.no_grad():
        next_q = target_net(next_states).max(1).values
        target_q = rewards + gamma * next_q * (1 - dones)

    loss = torch.nn.functional.mse_loss(predicted_q, target_q)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return float(loss.item())


def train(args):
    """Run the training loop until the step budget is reached.

    Resets the environment, selects an action, stores the transition, learns
    once the buffer has warmed up, decays epsilon, syncs the target network
    periodically, writes one CSV row per episode, and saves the model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(args.seed)
    n_actions = env.action_space.n

    online_net, target_net = build_networks(n_actions, device)
    optimizer = torch.optim.Adam(online_net.parameters(), lr=args.lr)
    buffer = ReplayBuffer(args.buffer_size)
    log_file, writer = start_log(args.out)

    state, _ = env.reset(seed=args.seed)
    episode = 0
    episode_reward = 0.0
    episode_length = 0
    loss = 0.0

    for step in range(1, args.steps + 1):
        if step <= args.warmup:
            action = env.action_space.sample()
        else:
            epsilon = get_epsilon(step, args.eps_start, args.eps_end, args.eps_decay)
            action = select_action(state, online_net, epsilon, n_actions, device)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.add(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward
        episode_length += 1

        if step > args.warmup and step % args.learn_every == 0 and len(buffer) >= args.batch_size:
            loss = learn(online_net, target_net, buffer, optimizer, args.batch_size, args.gamma, device)

        if step % args.target_sync == 0:
            update_target(online_net, target_net)

        if done:
            episode += 1
            epsilon = get_epsilon(step, args.eps_start, args.eps_end, args.eps_decay)
            writer.writerow([episode, step, episode_reward, episode_length, epsilon, loss])
            log_file.flush()

            state, _ = env.reset()
            episode_reward = 0.0
            episode_length = 0

    torch.save(online_net.state_dict(), Path(args.out) / "model.pth")
    log_file.close()
    env.close()


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
