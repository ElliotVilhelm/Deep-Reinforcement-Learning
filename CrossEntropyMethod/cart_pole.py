import gym
from Models.nn import NeuralNetwork
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np

from tensorboardX import SummaryWriter


HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70

# Episode: A single episode stored as total undiscounted reward and a collection of 'EpisodeStep'
Episode = namedtuple('Episode', field_names=['reward', 'steps'])

# EpisodeStep: A single step our agent made in the episode.  It stores the observation from the environment
# and the action taken by our agent.
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


def iterate_batches(env, net, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()
    softmax_layer = nn.Softmax(dim=1)

    while True:
        observation_tensor = torch.FloatTensor([obs])
        action_probability_distribution = softmax_layer(net(observation_tensor))
        action = Categorical(action_probability_distribution).sample().item()
        next_obs, reward, done, _ = env.step(action)
        episode_reward += reward
        episode_steps.append(EpisodeStep(observation=obs, action=action))

        if done:
            batch.append(Episode(reward=episode_reward, steps=episode_steps))
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs

def filter_batch(batch, percentile):
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_obs = []  # input
    train_act = []  # targets
    for sample in batch:
        if sample.reward >= reward_bound:
            train_obs.extend(map(lambda step: step.observation, sample.steps))
            train_act.extend(map(lambda step: step.action, sample.steps))

    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.LongTensor(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean


def simulate(env, net, steps=200):
    obs = env.reset()
    step = 0
    while True:
        obs_v = torch.FloatTensor(obs)
        actions_v = net(obs_v)
        action = torch.argmax(actions_v).item()
        obs, reward, done, _ = env.step(action)
        env.render()
        step += 1
        if step == steps:
            break

def plot_results(loss, reward_means, reward_bounds):
    import matplotlib.pyplot as plt
    plt.plot(loss)
    plt.plot(reward_means)
    plt.plot(reward_bounds)
    plt.legend(['Loss', 'Reward Mean', 'Reward Bound'])
    plt.grid()
    plt.title('Cart Pole Cross-Entropy Method Training Results')
    plt.show()

if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    neural_net = NeuralNetwork(obs_size, HIDDEN_SIZE, n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=neural_net.parameters(), lr=0.01)
    writer = SummaryWriter(comment="-cartpole")

    losses = []
    reward_means = []
    reward_bounds = []

    for iter_no, batch in enumerate(iterate_batches(env, neural_net, BATCH_SIZE)):
        obs_v, actions_v, reward_bound, reward_mean = filter_batch(batch, PERCENTILE)
        optimizer.zero_grad()
        action_scores_v = neural_net(obs_v)
        loss_v = objective(action_scores_v, actions_v)
        loss_v.backward()
        optimizer.step()
        print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" % (
            iter_no, loss_v.item(), reward_mean, reward_bound))
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_bound, iter_no)
        writer.add_scalar("reward_mean", reward_mean, iter_no)
        losses.append(loss_v.item() * 100)
        reward_means.append(reward_mean)
        reward_bounds.append(reward_bound)
        if reward_mean > 199:
            print("Solved!")
            break
    writer.close()
    simulate(env, neural_net, 500)  # run and render for 500 steps, lets see our results!
    plot_results(losses, reward_means, reward_bounds)




