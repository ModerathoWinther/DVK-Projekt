from random import random, randrange

import tensorflow as tf
from environment import Environment
from src.action_space import ACTION_SPACE

NUM_INPUTS = 4
NUM_OUTPUTS = 7

EXPLORATION = 0.05
LEARNING_RATE = 0.001
DISCOUNT = 0.99

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
loss_object = tf.keras.losses.MeanSquaredError()


def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(NUM_INPUTS,)),
        tf.keras.layers.Dense(24, activation=tf.nn.relu),
        tf.keras.layers.Dense(12, activation=tf.nn.relu),
        tf.keras.layers.Dense(NUM_OUTPUTS)])


def loss(model, x, y_true):
    y_pred = model(x)
    return loss_object(y_true=y_true, y_pred=y_pred)


# inputs = state, targets = true reward + future Q
def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def train(atr=False, macd=False, rsi=False):
    env = Environment("train", 5, atr=atr, macd=macd, rsi=rsi)
    q_network = build_model()
    target_network = build_model()
    replay_buffer = []

    while env.has_next():
        state = env.get_current_state()
        if random() <= EXPLORATION:
            action_index = randrange(len(ACTION_SPACE))
        else:
            action_index = q_network(state)

        action = ACTION_SPACE[action_index]
        env.perform_action(action)
        reward = env.get_reward_and_clear_trades()
        new_state = env.get_current_state()

        # todo store transition object in replay buffer

        # todo sample random transition from replay buffer and apply gradient to q-network using target_network as reference

        target_network = q_network


"""
loss_value, grads = grad(target_network, state, reward)
optimizer.apply_gradients(zip(grads, q_network.trainable_variables))
"""
