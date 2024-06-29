import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import gymnasium as gym
#for continuous action space

class ReplayBuffer:
    def __init__(self, max_size, input_shape, num_actions):
        self.mem_size = max_size
        self.mem_cntr = 0  #center
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, num_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, new_state, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminals = self.terminal_memory[batch]
        return states, actions, rewards, new_states, terminals


class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims=512, fc2_dims=512, name='critic'):
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.model_name = name
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.q = Dense(1, activation=None)

    def call(self, state, action):
        action_value = self.fc1(tf.concat([state, action], axis=1))
        action_value = self.fc2(action_value)
        return self.q(action_value)


class ActorNetwork(keras.Model):
    def __init__(self, fc1_dims=512, fc2_dims=512, n_action=2, name='actor'):
        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.model_name = name
        self.n_action = n_action
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.mu = Dense(self.n_action, activation=None)

    def call(self, state):
        val = self.fc1(state)
        val = self.fc2(val)
        return self.mu(val)


class Agent:
    def __init__(self, input_dims, alpha=0.001, beta=0.002, env=None, gamma=0.99, n_action=2,
                 max_size=1000000, tau=0.005, fc1=400, fc2=300, batch_size=64, noise=0.1):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.memory = ReplayBuffer(max_size, input_dims, n_action)
        self.n_actions = n_action
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]
        self.noise = noise
        self.actor = ActorNetwork(fc1_dims=fc1, fc2_dims=fc2, n_action=n_action)
        self.critic = CriticNetwork(fc1_dims=fc1, fc2_dims=fc2)
        self.actor_target = ActorNetwork(fc1_dims=fc1, fc2_dims=fc2, n_action=n_action)
        self.critic_target = CriticNetwork(fc1_dims=fc1, fc2_dims=fc2)
        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic.compile(optimizer=Adam(learning_rate=beta))
        self.actor_target.compile(optimizer=Adam(learning_rate=alpha))
        self.critic_target.compile(optimizer=Adam(learning_rate=beta))
        self.update_network_parameters(tau)
        self.tau = tau

    def update_network_parameters(self, tau):
        weights = []
        targets = self.actor_target.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.actor_target.set_weights(weights)
        weights = []
        targets = self.critic_target.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.critic_target.set_weights(weights)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions = self.actor.call(state)
        actions += tf.random.normal(shape=[self.n_actions], mean=0.0, stddev=self.noise)
        actions = tf.clip_by_value(actions, self.min_action, self.max_action)
        return actions[0]

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)
        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)
        with tf.GradientTape() as tape:
            target_actions = self.actor_target.call(states_)
            critic_value_ = tf.squeeze(self.critic_target.call(
                states_, target_actions), 1)
            critic_value = tf.squeeze(self.critic.call(states, actions), 1)
            target = rewards + self.gamma * critic_value_ * (1 - done)
            critic_loss = keras.losses.MSE(target, critic_value)
        critic_network_gradient = tape.gradient(critic_loss,
                                                self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(
            critic_network_gradient, self.critic.trainable_variables))
        with tf.GradientTape() as tape:
            new_policy_actions = self.actor.call(states)
            actor_loss = -self.critic.call(states, new_policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss,
                                               self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(
            actor_network_gradient, self.actor.trainable_variables))

        self.update_network_parameters(self.tau)

    def save(self, filename):
        self.actor.save_weights(f'{filename}.weights.h5')


if __name__ == '__main__':
    env_name = "InvertedPendulum-v4"
    env = gym.make(env_name)
    agent = Agent(input_dims=env.observation_space.shape, env=env,
                  n_action=env.action_space.shape[0])
    EPISODES = 2000
    MAX_STEPS = 1000
    score_history = []
    start = time.time()
    for i in range(EPISODES):
        state, info1 = env.reset()
        done = False
        score = 0
        for t in range(MAX_STEPS):
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = (terminated or truncated)
            score += reward
            agent.remember(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            if done:
                break
        score_history.append(score)
        av_latest_points = np.mean(score_history[-100:])
        print(f"\rEpisode {i + 1} | Total point average of the last {100} episodes: {av_latest_points:.2f}",
              end="")

        if (i + 1) % 100 == 0:
            print(f"\rEpisode {i + 1} | Total point average of the last {100} episodes: {av_latest_points:.2f}")
        if av_latest_points >= 475:
            print(f"\n\nEnvironment solved in {i + 1} episodes!")
            break
    tot_time = time.time() - start
    print(f"\nTotal Runtime: {tot_time:.2f} s ({(tot_time / 60):.2f} min)")
    agent.save(env_name)
    plt.plot(score_history)
    plt.title('Training Curve')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig(f'{env_name}.png')
    plt.show()
