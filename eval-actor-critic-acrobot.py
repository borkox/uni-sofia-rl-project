import torch
import torch.nn as nn
import torch.nn.functional as F
import gym

PATH_POLICY_MODEL = "outputs/acrobot-policy-model.bin"
PATH_VALUE_MODEL = "outputs/acrobot-value-model.bin"


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
        state = torch.from_numpy(state).view(1, -1)

        # use network to predict action probabilities
        action_probs = self(state)
        # This part is different from learning policy
        # There is no exploration part anymore
        return torch.argmax(action_probs)

# Make environment
env = gym.make('Acrobot-v1')

# Init network
print(f"Observation space: {env.observation_space.shape[0]}")
print(f"Action space: {env.action_space.n}")
policy_network = PolicyNetwork(env.observation_space.shape[0],
                               env.action_space.n)
policy_network.load_state_dict(torch.load(PATH_POLICY_MODEL))
policy_network.eval()

scores = []
# run episodes
for episode in range(5):

    # init variables
    state = env.reset()
    done = False
    score = 0

    # run episode, update online
    for step in range(500):
        env.render()
        # get action and log probability
        action = policy_network.select_action(state)

        # step with action
        state, reward, done, _ = env.step(action)

        # update episode score
        score += reward

        # if terminal state, next state val is 0
        if done:
            print(f"Episode {episode} finished after {step} timesteps, score={score}")
            break
    # append episode score
    scores.append(score)
