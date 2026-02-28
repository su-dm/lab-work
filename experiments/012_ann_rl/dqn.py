from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import random
import gymnasium as gym

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2500
TAU = 0.005
LR = 3e-4

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input = nn.Linear(input_dim, 10)
        self.h1 = nn.Linear(10, 10)
        self.h2 = nn.Linear(10, 10)
        self.output = nn.Linear(10, output_dim)

    def forward(self, x):
        x = torch.relu(self.input(x))
        x = torch.relu(self.h1(x))
        x = torch.relu(self.h2(x))
        x = self.output(x)
        return x

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def select_action(model, env, state):
    EPSILON = 0.1
    if random.random() < EPSILON:
        action = env.action_space.sample()
        return torch.tensor([[action]], device=device, dtype=torch.long)
    else:
        with torch.no_grad():
            return model(state.unsqueeze(0)).max(1).indices.view(1, 1)

def optimize(policy_net, target_net, memory, optimizer):
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)

    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device = device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    # output of policy_net is the values for each action, we index the column or action using the action index actually take  shown in action_batch
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    # no_grad because we're using the values for truth not deriving loss for this part of the equation, this is like the label
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # So we have the state_action_values according to pure policy net of the transitions in buffer/memory
    # We also have the expected_state_action_values which calculates it using reward + target_network(next_state)
    # We adjust the policy net with respect to the reward observed and the targets value calculation
    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    # unsqueeze because state_action_values is (batch_size,1) for the gathered action but expected_state_action_values is (batch_size,)
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad() # clear gradients instead of accumulating
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def evaluate(policy_net, env, episodes=5):
    print("\nEvaluating greedy policy...")
    print("-" * 55)
    rewards = []
    for e in range(episodes):
        state, _ = env.reset()
        state = torch.tensor(state, device=device)
        total_reward = 0.0
        done = False
        while not done:
            with torch.no_grad():
                action = policy_net(state.unsqueeze(0)).argmax().item()
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            state = torch.tensor(observation, device=device)
        rewards.append(total_reward)
        print(f"  Eval ep {e+1:>2}: reward = {total_reward:.2f}")
    mean_r = sum(rewards) / len(rewards)
    print(f"\nEval summary | Mean: {mean_r:.2f} | Min: {min(rewards):.2f} | Max: {max(rewards):.2f}")


def train(policy_net, target_net, optimizer, memory, env):
    EPISODES = 10
    MAX_STEPS = 50
    episode_rewards = []

    print(f"Device: {device} | Episodes: {EPISODES} | Max steps/ep: {MAX_STEPS}")
    print("-" * 55)

    for e in range(EPISODES):
        state, info = env.reset()
        state = torch.tensor(state, device=device)
        steps = 0
        total_reward = 0.0
        done = False
        while not done:
            action = select_action(policy_net, env, state)
            #action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action.item())
            total_reward += reward
            reward = torch.tensor([reward], device=device)
            if terminated or truncated or steps >= MAX_STEPS:
                done = True
                # we push None next_states to buffer because bellman equation resolves to Q being simply the reward at the end (no future)
                next_state = None
            else:
                next_state = torch.tensor(observation, device=device)
                steps += 1

            memory.push(state, action, next_state, reward)

            state = next_state

            optimize(policy_net, target_net, memory, optimizer)

            # Soft update of the target network's weights using Polyak averaging, used in DDPG
            # θ′ ← τ θ + (1 −τ )θ′
            # Original paper uses a hard update or full copy after 10k steps
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

        episode_rewards.append(total_reward)
        avg = sum(episode_rewards[-10:]) / min(len(episode_rewards), 10)
        print(f"Ep {e+1:>4}/{EPISODES} | steps: {steps:>3} | reward: {total_reward:>7.2f} | avg(10): {avg:>7.2f} | buf: {len(memory):>5}")

    print("-" * 55)
    print(f"Training done | best: {max(episode_rewards):.2f} | mean: {sum(episode_rewards)/len(episode_rewards):.2f}")

def main():

    env = gym.make('CartPole-v1')
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    buffer = ReplayMemory(10000)

    policy_net = DQN(input_dim, output_dim).to(device)
    target_net = DQN(input_dim, output_dim).to(device)

    # align the weights to have same start point
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

    train(policy_net, target_net, optimizer, buffer, env)
    evaluate(policy_net, env)
    env.close()

if __name__ == "__main__":
    main()
