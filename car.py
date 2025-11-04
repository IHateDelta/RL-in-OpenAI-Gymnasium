import gymnasium as gym
from agent import QLearningAgent
import numpy as np
import joblib
import time

env_name = 'MountainCar-v0'
env = gym.make(env_name)


BIN_COUNT = 20
POS_SPACE = np.linspace(-1.2, 0.6, BIN_COUNT)  
VEL_SPACE = np.linspace(-0.07, 0.07, BIN_COUNT) 
def mountaincar_state_converter(state):
    pos, vel = state
    # np.digitize returns the index of the bin the value belongs to
    pos_bin = np.digitize(pos, POS_SPACE)
    vel_bin = np.digitize(vel, VEL_SPACE)
    return (pos_bin, vel_bin)

train = True
training_episodes = 60_000

if train:
    agent = QLearningAgent(
        actions=range(env.action_space.n), 
        state_converter=mountaincar_state_converter, 
        discount_factor=0.95
)
    history = []
    agent.train_mode()
    total_reward = 0

    # training loop
    for episode in range(training_episodes):

        state, _ = env.reset()
        done = False
        steps = 0

        while not done and steps < env._max_episode_steps:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.learn(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            steps += 1

        agent.update_epsilon_start_end_steps(episode, training_episodes, start_epsilon=1.0, end_epsilon=0.01)
        if (episode + 1) % 500 == 0:
            print(f"Episode {episode}: Total Reward (last 500): {total_reward}, Epsilon: {agent.epsilon:.4f}")
            total_reward = 0

    joblib.dump(agent.q_table, 'mountaincar_qtable.pkl')


    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(
        context='talk',
        style='whitegrid',
        rc={
            'figure.figsize': (10, 6),
            'axes.titlesize': 18,
            'axes.labelsize': 14,
            'lines.linewidth': 2,
            'lines.markersize': 6,
        }
    )

    sns.lineplot(data=history, marker='o')
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()
    plt.savefig('mountaincar_training_rewards.png')


test_env = gym.make(env_name, render_mode='human')
agent_table = joblib.load('mountaincar_qtable.pkl')
agent = QLearningAgent(actions=range(env.action_space.n), state_converter=mountaincar_state_converter)
agent.q_table = agent_table

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
