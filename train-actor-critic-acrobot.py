import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import gym
import numpy as np
from collections import deque

# Description of the gym environment
# https://www.gymlibrary.ml/environments/classic_control/acrobot/

# discount factor for future utilities
DISCOUNT_FACTOR = 0.95
# number of episodes to run
NUM_EPISODES = 1000
# max steps per episode
MAX_STEPS = 500
# score agent needs for environment to be solved
# For Acrobot-v1, this is -100
SOLVED_SCORE = -100
# device to run model on
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARN_RATE = 0.002
PATH_POLICY_MODEL = "outputs/acrobot-policy-model.bin"
PATH_VALUE_MODEL = "outputs/acrobot-value-model.bin"
PATH_SCORES = "outputs/scores-acrobot.txt"


# Using a neural network to learn our policy parameters
class PolicyNetwork(nn.Module):

    # Takes in observations and outputs actions
    def __init__(self, observation_space, action_space):
        super(PolicyNetwork, self).__init__()
        self.input_layer = nn.Linear(observation_space, 128)
        self.output_layer = nn.Linear(128, action_space)

    def forward(self, x):
        x = self.input_layer(x)
        x = F.relu(x)
        actions = self.output_layer(x)
        # get softmax for a probability distribution
        return F.softmax(actions, dim=1)

    def save(self, path):
        print(f"Saving Policy network in '{path}'")
        torch.save(self.state_dict(), path)

    def select_action(self, state):
        # make torch tensor of shape [BATCH x observation_size]
        state = torch.from_numpy(state).view(1, -1).to(DEVICE)

        # use network to predict action probabilities
        action_probs = self(state)

        # sample an action using the probability distribution
        m = Categorical(action_probs)
        # action will be a single value tensor: [0] or [1] or [2]
        action = m.sample()
        # return action as number and log probability
        return action.item(), m.log_prob(action)


# Using a neural network to learn state value
class StateValueNetwork(nn.Module):

    # Takes in state
    def __init__(self, observation_space):
        super(StateValueNetwork, self).__init__()

        self.input_layer = nn.Linear(observation_space, 128)
        self.output_layer = nn.Linear(128, 1)

    def save(self, path):
        print(f"Saving State-Value network in '{path}'")
        torch.save(self.state_dict(), path)

    # Expects X in shape [BATCH x observation_space]
    def forward(self, x):
        x = self.input_layer(x)
        x = F.relu(x)
        state_value = self.output_layer(x)
        return state_value


def save_scores(scores_list):
    print(f"Saving scores in '{PATH_SCORES}'.")
    np.savetxt(PATH_SCORES, scores_list, delimiter=',')


# Make environment
env = gym.make('Acrobot-v1')

# Init network
print(f"Observation space: {env.observation_space.shape[0]}")
print(f"Action space: {env.action_space.n}")
policy_network = PolicyNetwork(env.observation_space.shape[0],
                               env.action_space.n).to(DEVICE)
stateval_network = StateValueNetwork(env.observation_space.shape[0]).to(DEVICE)

# Init optimizer
policy_optimizer = optim.Adam(policy_network.parameters(), lr=LEARN_RATE)
stateval_optimizer = optim.Adam(stateval_network.parameters(), lr=LEARN_RATE)

# track scores
scores = []

# track recent scores
recent_scores = deque(maxlen=100)

# run episodes
for episode in range(NUM_EPISODES):

    # init variables
    state = env.reset()
    done = False
    score = 0
    I = 1

    # run episode, update online
    for step in range(MAX_STEPS):
        env.render()
        # get action and log probability
        action, lp = policy_network.select_action(state)

        # step with action
        new_state, reward, done, _ = env.step(action)

        # update episode score
        score += reward

        # convert to torch tensor [Batch x observationsize]
        state_tensor = torch.from_numpy(state).reshape(1, -1).to(DEVICE)
        state_val = stateval_network(state_tensor)

        # get state value of next state
        new_state_tensor = torch.from_numpy(new_state).view(1, -1).to(DEVICE)
        new_state_val = stateval_network(new_state_tensor)

        # if terminal state, next state val is 0
        if done:
            print(f"Episode {episode} finished after {step} timesteps, score={score}")
            new_state_val = torch.tensor([[0]]).to(DEVICE)

        # calculate value function loss with MSE
        val_loss = F.mse_loss(reward + DISCOUNT_FACTOR * new_state_val, state_val)
        val_loss *= I

        # calculate policy loss
        advantage = reward + DISCOUNT_FACTOR * new_state_val.item() - state_val.item()
        # lp is tensor of shape [1], advantage is scalar
        policy_loss = -lp * advantage
        policy_loss *= I

        # Back-propagate policy
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        # Back-propagate value
        stateval_optimizer.zero_grad()
        val_loss.backward()
        stateval_optimizer.step()

        if done:
          break

        # move into new state, discount I
        state = new_state
        I *= DISCOUNT_FACTOR

    # append episode score
    scores.append(score)
    recent_scores.append(score)

    # early stopping if we meet solved score goal
    if np.array(recent_scores).mean() >= SOLVED_SCORE:
        print(f"Learning is complete successfully.")
        policy_network.save(PATH_POLICY_MODEL)
        stateval_network.save(PATH_VALUE_MODEL)
        break
    if episode % 10 == 0:
       save_scores(scores)

save_scores(scores)
