import argparse
import pickle
from pathlib import Path
from collections import deque

import torch
import random

from src.crafter_wrapper import Env
from src.drl_agent import DRLAgent
from src.icm import ICM


def _save_stats(episodic_returns, crt_step, path):
    # save the evaluation stats
    episodic_returns = torch.tensor(episodic_returns)
    avg_return = episodic_returns.mean().item()
    print(
        "[{:06d}] eval results: R/ep={:03.2f}, std={:03.2f}.".format(
            crt_step, avg_return, episodic_returns.std().item()
        )
    )
    with open(path + "/eval_stats.pkl", "ab") as f:
        pickle.dump({"step": crt_step, "avg_return": avg_return}, f)


def eval(agent, env, crt_step, opt):
    """ Use the greedy, deterministic policy, not the epsilon-greedy policy you
    might use during training.
    """
    episodic_returns = []
    for _ in range(opt.eval_episodes):
        obs, done = env.reset(), False
        episodic_returns.append(0)
        while not done:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            episodic_returns[-1] += reward

    _save_stats(episodic_returns, crt_step, opt.logdir)


def _info(opt):
    try:
        int(opt.logdir.split("/")[-1])
    except:
        print(
            "Warning, logdir path should end in a number indicating a separate"
            + "training run, else the results might be overwritten."
        )
    if Path(opt.logdir).exists():
        print("Warning! Logdir path exists, results can be corrupted.")
    print(f"Saving results in {opt.logdir}.")
    print(
        f"Observations are of dims ({opt.history_length},84,84),"
        + "with values between 0 and 1."
    )


def main(opt):
    _info(opt)
    # opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt.device = torch.device("cpu")
    env = Env("train", opt)
    eval_env = Env("eval", opt)
    input_shape = env.observation_space.shape[0]  # Adjust as per your environment
    num_actions = env.action_space.n

    # Training parameters
    num_episodes = 1000
    max_steps_per_episode = 100
    replay_buffer = deque(maxlen=50000)
    batch_size = 64
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    learning_rate = 1e-4
    gamma = 0.99  # Discount factor for future rewards

    # Agent and ICM setup
    agent = DRLAgent(input_shape, num_actions, learning_rate=learning_rate, gamma=gamma)
    icm = ICM(input_shape, num_actions, learning_rate=learning_rate)

    # Training loop
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** episode))

        for step in range(max_steps_per_episode):
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)

            # Store transition in the replay buffer
            replay_buffer.append((state, action, reward, next_state, done))

            # Update the agent if the replay buffer is sufficiently full
            if len(replay_buffer) > batch_size:
                batch = random.sample(replay_buffer, batch_size)
                agent.update_policy(batch, icm)

            state = next_state
            total_reward += reward

            if done:
                break

        print(f"Episode {episode}, Total Reward: {total_reward}")


def get_options():
    """ Configures a parser. Extend this with all the best performing hyperparameters of
        your agent as defaults.

        For devel purposes feel free to change the number of training steps and
        the evaluation interval.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="logdir/random_agent/0")
    parser.add_argument(
        "--steps",
        type=int,
        metavar="STEPS",
        default=1_000_000,
        help="Total number of training steps.",
    )
    parser.add_argument(
        "-hist-len",
        "--history-length",
        default=4,
        type=int,
        help="The number of frames to stack when creating an observation.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=100_000,
        metavar="STEPS",
        help="Number of training steps between evaluations",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=20,
        metavar="N",
        help="Number of evaluation episodes to average over",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(get_options())
