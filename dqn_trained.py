import gymnasium as gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

ENV_NAME='CartPole-v1'
# Create the Frozen Lake environment
env = gym.make(ENV_NAME, render_mode='human')
q_network = tf.keras.models.load_model(f'{ENV_NAME}.h5')
reward_history=[]
for i in range(10):
    state = env.reset()[0]
    done = False
    reward1=0
    while not done:
        state = np.expand_dims(state, axis=0)
        q_values = q_network(state)
        action = np.argmax(q_values.numpy()[0])
        state, reward, terminated, truncated, info = env.step(action)
        done=(terminated or truncated)
        reward1+=reward
    reward_history.append(reward1)
plt.plot(reward_history)
plt.title('reward curve after training')
plt.xlabel('Time')
plt.ylabel('Reward')
plt.savefig(f'{ENV_NAME}_trained.png')
plt.show()

