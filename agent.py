import random
import numpy as np

class QLearningAgent:
    def __init__(
        self,
        actions,
        learning_rate=0.1,
        discount_factor=0.99,
        exploration_decay=0.995, # 0.995 decays epsilon to ~0.005 after 1000 episodes
        state_converter=None,
    ):
        self.actions = list(actions)
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = 1.0  
        self.epsilon_decay = exploration_decay
        self.q_table = {}  # keys: (state_key, action)
        self.state_converter = state_converter if state_converter else (lambda s: s if isinstance(s, (int, str, tuple)) else tuple(np.array(s).ravel())) # Convert state to a hashable type
        self.training = True

    def _key(self, state):
        return self.state_converter(state)

    def get_q_value(self, state, action):
        return self.q_table.get((self._key(state), action), 0.0) # default Q-value is 0.0

    def choose_action(self, state):
        if random.random() < self.epsilon: # epsilon-greedy
            return random.choice(self.actions)
        else:
            q_values = [self.get_q_value(state, a) for a in self.actions]
            max_q = max(q_values)
            # break ties randomly
            best = [a for a, q in zip(self.actions, q_values) if q == max_q]
            return random.choice(best)

    def learn(self, state, action, reward, next_state, done=False):
        current_q = self.get_q_value(state, action)
        if done:
            target = reward
        else:
            future_qs = [self.get_q_value(next_state, a) for a in self.actions]
            target = reward + self.gamma * max(future_qs)
        new_q = (1.0 - self.lr) * current_q + self.lr * target
        self.q_table[(self._key(state), action)] = new_q

    def decay_exploration(self):
        self.epsilon *= self.epsilon_decay
    
    def train_mode(self):
        self.training = True

    def eval_mode(self):
        self.training = False



















    def update_epsilon_start_end_steps(self, episode, total_episodes, start_epsilon=1.0, end_epsilon=0.01):
        self.epsilon = end_epsilon + ((start_epsilon - end_epsilon) * (1 - episode / (total_episodes)))/2