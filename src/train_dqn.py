from dataclasses import dataclass
from random import random, randrange

import yaml
import numpy
import tensorflow as tf
from environment import Environment
from src.action_space import Action, ACTION_SPACE

NUM_TRADES = 5
NUM_INPUTS = 5 + NUM_TRADES * 4
NUM_OUTPUTS = len(ACTION_SPACE)

EXPLORATION = 0.05
LEARNING_RATE = 0.001
DISCOUNT = 0.99

TN_UPDATE_INTERVAL = 100

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
loss_object = tf.keras.losses.MeanSquaredError()


@dataclass
class Transition:
    state: numpy.ndarray
    action: Action
    reward: float
    next_state: numpy.ndarray

def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(NUM_INPUTS,)),
        tf.keras.layers.Dense(24, activation=tf.nn.relu),
        tf.keras.layers.Dense(12, activation=tf.nn.relu),
        tf.keras.layers.Dense(NUM_OUTPUTS)])


def loss(model, transition, target):
    q_values = model(transition.state)
    q = q_values[transition.action.index]
    return loss_object(y_true=target, y_pred=q)


def grad(model, transition):
    target = transition.reward + DISCOUNT * tf.reduce_max(model(transition.next_state))
    with tf.GradientTape() as tape:
        loss_value = loss(model, transition, target)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def train(atr=False, macd=False, rsi=False):
    env = Environment("train", NUM_TRADES, atr=atr, macd=macd, rsi=rsi)
    q_network = build_model()
    target_network = build_model()
    replay_buffer = []
    counter = 0

    while env.has_next():
        counter += 1
        state = env.get_current_state()
        state = state.reshape((1,-1))
        if random() <= EXPLORATION:
            action_index = randrange(len(ACTION_SPACE))
        else:
            action_index = q_network(state).argmax()

        action = ACTION_SPACE[action_index]
        env.perform_action(action)
        reward = env.get_reward_and_clear_trades()
        new_state = env.get_current_state()

        replay_buffer.append(Transition(state, action, reward, new_state))

        transition = replay_buffer[randrange(len(replay_buffer))]
        loss_value, grads = grad(target_network, transition)
        print(loss_value)
        optimizer.apply_gradients(zip(grads, q_network.trainable_variables))

        if counter % TN_UPDATE_INTERVAL == 0:
            target_network = q_network

if __name__ == '__main__':
    train(atr=False, macd=False, rsi=False)