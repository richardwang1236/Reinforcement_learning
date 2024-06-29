import random
import time
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam


def compute_loss(experiences, gamma, q_network, target_q_network):
    """
    Calculates the loss.

    Args:
      experiences: (tuple) tuple of ["state", "action", "reward", "next_state", "done"] namedtuples
      gamma: (float) The discount factor.
      q_network: (tf.keras.Sequential) Keras model for predicting the q_values
      target_q_network: (tf.keras.Sequential) Keras model for predicting the targets

    Returns:
      loss: (TensorFlow Tensor(shape=(0,), dtype=int32)) the Mean-Squared Error between
            the y targets and the Q(s,a) values.
    """
    states, actions, rewards, next_states, done_vals = experiences
    max_qsa = tf.reduce_max(target_q_network(next_states), axis=-1)
    y_targets = rewards + gamma * max_qsa * (1 - done_vals)
    q_values = q_network(states)
    q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]),
                                                tf.cast(actions, tf.int32)], axis=1))
    loss = MSE(y_targets, q_values)
    return loss


@tf.function
def agent_learn(experiences, gamma):
    """
    Updates the weights of the Q networks.

    Args:
      experiences: (tuple) tuple of ["state", "action", "reward", "next_state", "done"] namedtuples
      gamma: (float) The discount factor.

    """
    TAU = 1e-3
    # Calculate the loss
    with tf.GradientTape() as tape:
        loss = compute_loss(experiences, gamma, q_network, target_q_network)

    # Get the gradients of the loss with respect to the weights.
    gradients = tape.gradient(loss, q_network.trainable_variables)

    # Update the weights of the q_network.
    optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))

    # update the weights of target q_network
    for target_weights, q_net_weights in zip(
            target_q_network.weights, q_network.weights
    ):
        target_weights.assign(TAU * q_net_weights + (1.0 - TAU) * target_weights)


if __name__ == '__main__':
    E_DECAY = 0.995  # ε-decay rate for the ε-greedy policy.
    E_MIN = 0.01  # Minimum ε value for the ε-greedy policy.
    MINIBATCH_SIZE = 64
    MEMORY_SIZE = 100_000  # size of memory buffer
    GAMMA = 0.995  # discount factor
    ALPHA = 1e-3  # learning rate
    NUM_STEPS_FOR_UPDATE = 4  # perform a learning update every C time steps
    ENV_NAME = 'LunarLander-v2'
    EXPECTED_REWARD = 200
    env = gym.make(ENV_NAME)
    state_size = env.observation_space.shape
    num_actions = env.action_space.n
    q_network = Sequential([
        Input(shape=state_size),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_actions, activation='linear'),
    ])
    target_q_network = Sequential([
        Input(shape=state_size),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_actions, activation='linear'),
    ])
    optimizer = Adam(learning_rate=ALPHA)
    start = time.time()
    experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    num_episodes = 2000
    max_num_timesteps = 1000

    total_point_history = []

    num_p_av = 100  # number of total points to use for averaging
    epsilon = 1.0  # initial ε value for ε-greedy policy

    # Create a memory buffer D with capacity N
    memory_buffer = deque(maxlen=MEMORY_SIZE)

    # Set the target network weights equal to the Q-Network weights
    target_q_network.set_weights(q_network.get_weights())

    for i in range(num_episodes):

        # Reset the environment to the initial state and get the initial state
        state, __ = env.reset()
        total_points = 0

        for t in range(max_num_timesteps):
            # From the current state S choose an action A using an ε-greedy policy
            state_qn = np.expand_dims(state, axis=0)  # state needs to be the right shape for the q_network
            q_values = q_network(state_qn)
            if random.random() > epsilon:
                action = np.argmax(q_values.numpy()[0])
            else:
                action = random.choice(np.arange(4))

            # Take action A and receive reward R and the next state S'
            next_state, reward, done, _, info = env.step(action)

            # Store experience tuple (S,A,R,S') in the memory buffer.
            # We store the done variable as well for convenience.
            memory_buffer.append(experience(state, action, reward, next_state, done))

            if (t + 1) % NUM_STEPS_FOR_UPDATE == 0 and len(memory_buffer) > MINIBATCH_SIZE:
                # Sample random mini-batch of experience tuples (S,A,R,S') from D
                experiences = random.sample(memory_buffer, k=MINIBATCH_SIZE)
                states = tf.convert_to_tensor(
                    np.array([e.state for e in experiences if e is not None]), dtype=tf.float32
                )
                actions = tf.convert_to_tensor(
                    np.array([e.action for e in experiences if e is not None]), dtype=tf.float32
                )
                rewards = tf.convert_to_tensor(
                    np.array([e.reward for e in experiences if e is not None]), dtype=tf.float32
                )
                next_states = tf.convert_to_tensor(
                    np.array([e.next_state for e in experiences if e is not None]), dtype=tf.float32
                )
                done_vals = tf.convert_to_tensor(
                    np.array([e.done for e in experiences if e is not None]).astype(np.uint8),
                    dtype=tf.float32,
                )
                experiences = (states, actions, rewards, next_states, done_vals)

                # Set the y targets, perform a gradient descent step,
                # and update the network weights.
                agent_learn(experiences, GAMMA)

            state = next_state.copy()
            total_points += reward

            if done:
                break

        total_point_history.append(total_points)
        av_latest_points = np.mean(total_point_history[-num_p_av:])

        # Update the ε value
        epsilon = max(E_MIN, E_DECAY * epsilon)

        print(f"\rEpisode {i + 1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}",
              end="")

        if (i + 1) % num_p_av == 0:
            print(f"\rEpisode {i + 1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}")

        # We will consider that the environment is solved if we get an
        # average of 200 points in the last 100 episodes.
        if av_latest_points >= EXPECTED_REWARD:
            print(f"\n\nEnvironment solved in {i + 1} episodes!")
            break

    tot_time = time.time() - start

    print(f"\nTotal Runtime: {tot_time:.2f} s ({(tot_time / 60):.2f} min)")
    q_network.save(f'{ENV_NAME}.h5')
    plt.plot(total_point_history)
    plt.title('Training Curve')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig(f'{ENV_NAME}.png')
    plt.show()