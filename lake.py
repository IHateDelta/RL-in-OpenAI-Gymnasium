from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import gymnasium as gym
from agent import QLearningAgent
import time

map_size=16
map = generate_random_map(size=map_size, p=0.8)
env_name = 'FrozenLake-v1'
env = gym.make(env_name, desc=map, is_slippery=False)

agent = QLearningAgent(actions=range(env.action_space.n), exploration_decay=0.999)

training_episodes = 50000

agent.train_mode()


# training loop
for episode in range(training_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    state_list=[]
    while not done:
        action = agent.choose_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        #print(reward)
        if reward==1: reward=1000000
        else:
            if not(state in state_list):
                reward=state//map_size+state%map_size
                state_list.append(state)
        done = terminated or truncated

        agent.learn(state, action, reward, next_state)
        state = next_state
        total_reward += reward

    agent.decay_exploration()
    if episode % 100 == 0:
        print(f"Episode {episode}: Total Reward: {total_reward} Exploration Rate: {agent.epsilon:.4f}")




test_env = gym.make(env_name, desc=map, render_mode='human', is_slippery=False)
agent.eval_mode()
state, _ = test_env.reset()
done = False
total_reward = 0

while not done:
    action = agent.choose_action(state)
    next_state, reward, terminated, truncated, _ = test_env.step(action)
    done = terminated or truncated
    time.sleep(0.01)

    total_reward += reward
    state = next_state

print(f"Test Episode 1: Total Reward: {total_reward}")
