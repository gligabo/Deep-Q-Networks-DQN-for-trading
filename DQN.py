import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def custom_reward_function(history):
    """
    Calcula a recompensa como sendo a diferença entre o preço do dia atual e do dia anterior,
    multiplicado pelo valor da ação.

    Fórmula da recompensa:
    rt = (price_t+1 - price_t) * action_t
    """
    t = history["step", -1]
    t_minus_1 = t - 1

    price_t = history["data_close", t_minus_1]
    price_t_plus_1 = history["data_close", t]
    action_t = history["position_index", t]

    if action_t == 0:
        reward = (price_t_plus_1 - price_t) * -1
    elif action_t == 1:
        reward = 0
    else:
        reward = (price_t_plus_1 - price_t) * 1

    return reward

class DQN(nn.Module):
    """
    Rede Neural para o agente DQN (Deep Q-Network).
    """
    def __init__(self, in_states, h1_nodes, h2_nodes, out_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(in_states, h1_nodes)
        self.fc2 = nn.Linear(h1_nodes, h2_nodes)
        self.fc3 = nn.Linear(h2_nodes, h1_nodes)
        self.out = nn.Linear(h1_nodes, out_actions)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)

class ReplayMemory():
    """
    Memória de Replay para armazenar experiências do agente.
    """
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)
    
    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

class GymTrading():
    """
    Classe principal para treinamento e teste do agente de negociação.
    """
    learning_rate_a = 1.0e-5
    discount_factor_g = 0.9
    network_sync_rate = 1000
    replay_memory_size = 20000
    mini_batch_size = 128
    loss_fn = nn.SmoothL1Loss()
    optimizer = None

    def train(self, episodes, data_train, p=1.0, z=0.01, c=0.5, window=1):
        """
        Treina o agente usando aprendizado por reforço Deep Q-Learning.
        """
        env_train = gym.make(
            "TradingEnv",
            name="AAPL",
            positions=[-1, 0, 1],
            df=data_train,
            dynamic_feature_functions=[],
            trading_fees=0.01 / 100,
            borrow_interest_rate=0.0003 / 100,
            reward_function=custom_reward_function,
            windows=window
        )
        
        num_states = env_train.observation_space.shape[0] * env_train.observation_space.shape[1]
        num_actions = env_train.action_space.n
        epsilon = p
        memory = ReplayMemory(self.replay_memory_size)

        policy_dqn = DQN(num_states, 128, 128, num_actions).to(device)
        target_dqn = DQN(num_states, 128, 128, num_actions).to(device)
        target_dqn.load_state_dict(policy_dqn.state_dict())
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        epsilon_history = []
        rewards = []

        for episode in range(episodes):
            state, _ = env_train.reset()
            terminated = False
            truncated = False
            total_reward = 0.0

            epsilon = max(0.01, np.exp(np.log(p) + c * episode * (np.log(z) - np.log(p)) / episodes))

            while not terminated and not truncated:
                state_tensor = self.state_to_dqn_input(state, num_states).unsqueeze(0)
                if random.random() < epsilon:
                    action = env_train.action_space.sample()
                else:
                    with torch.no_grad():
                        q_values = policy_dqn(state_tensor)
                        action = q_values.argmax(dim=1).item()
                new_state, reward, terminated, truncated, _ = env_train.step(action)
                memory.append((state, action, new_state, reward, terminated or truncated))
                state = new_state
                total_reward += reward
            
            if len(memory) > self.mini_batch_size:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)

            if episode % self.network_sync_rate == 0:
                target_dqn.load_state_dict(policy_dqn.state_dict())
            
            rewards.append(total_reward)
            epsilon_history.append(epsilon)
            print(f"Episódio {episode}, Epsilon: {epsilon:.4f}, Recompensa: {total_reward:.2f}")
        
        env_train.close()
        torch.save(policy_dqn.state_dict(), "gym_trading.pt")

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        """
        Atualiza os pesos da rede neural através do algoritmo de aprendizado DQN.
        """
        states, actions, next_states, rewards, dones = zip(*mini_batch)
        states = torch.tensor(np.array(states, dtype=np.float32), device=device).view(len(states), -1)
        next_states = torch.tensor(np.array(next_states, dtype=np.float32), device=device).view(len(next_states), -1)
        actions = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)
        
        current_q_values = policy_dqn(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = target_dqn(next_states).max(dim=1, keepdim=True)[0]
        targets = rewards + (1 - dones) * self.discount_factor_g * next_q_values
        
        loss = self.loss_fn(current_q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_dqn.parameters(), max_norm=1.0)
        self.optimizer.step()
    
    def state_to_dqn_input(self, state, num_states):
        """
        Converte o estado em um tensor para entrada na rede neural.
        """
        return torch.tensor(state, dtype=torch.float32, device=device).view(-1)