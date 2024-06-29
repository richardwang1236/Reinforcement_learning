import gymnasium as gym
import tensorflow as tf
from ddpg import ActorNetwork
import matplotlib.pyplot as plt

def choose_action(observation):
    state = tf.convert_to_tensor([observation], dtype=tf.float32)
    actions = actor.call(state)
    actions = tf.clip_by_value(actions, -3.0, 3.0)
    return actions[0]

ENV_NAME='InvertedPendulum-v4'
# Create the environment
env = gym.make(ENV_NAME, render_mode='human')
n_actions = env.action_space.shape[0]
actor = ActorNetwork(n_action=n_actions, name='actor', fc1_dims=400, fc2_dims=300)

actor.load_weights(f'{ENV_NAME}.weights.h5')
done=False
reward_history=[]
for i in range(10):
    reward1=0
    while not done:
        observation, _ = env.reset()
        action = choose_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        done=(truncated or terminated)
        reward1+=reward
    reward_history.append(reward1)
plt.plot(reward_history)
plt.title('reward curve after training')
plt.xlabel('Time')
plt.ylabel('Reward')
plt.savefig(f'{ENV_NAME}.png')
plt.show()


