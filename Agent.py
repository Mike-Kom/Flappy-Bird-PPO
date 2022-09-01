import numpy as np
import tensorflow as tf
import gym
from collections import deque
from PPO_SeparateNet import Actor, Critic
import random


class Agent:
    def __init__(self, action_space, save_name, dl1=32, dl2=32, lr=1e-3, gamma=0.99, ent_coef=0.01, v_coef=0.5,
                 max_grad_norm=0.5, epsilon=1e-5, alpha=0.99, one_net=True):
        self.actor = Actor(n_actions=action_space, l1=dl1, l2=dl2)
        self.actor.compile(optimizer=tf.optimizers.Adam(lr))
        self.critic = Critic(l1=dl1, l2=dl2)
        self.critic.compile(optimizer=tf.optimizers.Adam(lr * 2))
        self.action_space = action_space
        self.gamma = gamma
        self.save_name = save_name
        self.ent_coef = ent_coef
        self.clip_ratio = 0.2
        self.actor_loss = deque(maxlen=1000)
        self.critic_loss = deque(maxlen=1000)
        self.entropy_loss = deque(maxlen=1000)
        self.best_score = -1000
        self.probs = None

    def save_models(self):
        print("...saving model...")
        self.actor.save_weights(f"{self.actor.chkpt_dir}/Actor_{self.save_name}.h5")
        self.critic.save_weights(f"{self.critic.chkpt_dir}/Critic_{self.save_name}.h5")

    def load_models(self):
        print("...loading model...")
        self.actor.load_weights(f"{self.actor.chkpt_dir}/Actor_{self.save_name}.h5")
        self.critic.load_weights(f"{self.critic.chkpt_dir}/Critic_{self.save_name}.h5")

    @tf.function(jit_compile=True)
    def choose_action(self, state):
        logits = self.actor(state)
        action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
        return logits, action

    @tf.function(jit_compile=True)
    def logprobabilities(self, logits, action):
        # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
        logprobabilities_all = tf.nn.log_softmax(logits)
        logprobability = tf.reduce_sum(
            tf.one_hot(action, self.action_space) * logprobabilities_all, axis=1
        )
        return logprobability

    @tf.function(jit_compile=True)
    def cat_entropy(self, logits):
        a0 = logits - tf.reduce_max(logits, 1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, 1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.math.log(z0) - a0), 1)

    # @tf.function(jit_compile=True)
    def train_policy(self, observation_buffer, action_buffer, logprobability_buffer, advantage_buffer):
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            logits = self.actor(observation_buffer)
            ratio = tf.exp(self.logprobabilities(logits, action_buffer)
                           - logprobability_buffer)

            entropy = self.cat_entropy(logits) * self.ent_coef

            min_advantage = tf.where(
                advantage_buffer > 0,
                (1 + self.clip_ratio) * advantage_buffer,
                (1 - self.clip_ratio) * advantage_buffer)

            actor_loss = -tf.reduce_mean(
                tf.minimum(ratio * advantage_buffer, min_advantage)) - entropy
        self.entropy_loss.append(entropy)
        self.actor_loss.append(actor_loss)
        self.probs = tf.nn.softmax(logits)

        actor_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_gradient, self.actor.trainable_variables))

        kl = tf.reduce_mean(
            logprobability_buffer
            - self.logprobabilities(self.actor(observation_buffer), action_buffer)
        )
        kl = tf.reduce_sum(kl)
        return kl

    # @tf.function(jit_compile=True)
    def train_value_function(self, observation_buffer, return_buffer, value_buffer):
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            val = self.critic(observation_buffer)

            # print("Return Buffer:")
            # print(return_buffer.shape)
            # print(return_buffer[50:70])  
            # print("Val:")
            # print(val.shape)
            # print(val[50:70])
            critic_loss = tf.reduce_mean((return_buffer - val) ** 2)

            # value_prediction = self.critic(observation_buffer)
            # value_prediction_clipped = value_prediction[:-1] +\
            #                            tf.clip_by_value(value_prediction[1:] - value_buffer[:-1],
            #                                                                 - self.clip_ratio, self.clip_ratio)
            # critic_loss_unclipped = tf.square(value_prediction[1:] - return_buffer[1:])
            # critic_loss_clipped = tf.square(value_prediction_clipped - return_buffer[1:])
            # critic_loss = tf.reduce_mean(tf.maximum(critic_loss_unclipped, critic_loss_clipped))

        self.critic_loss.append(critic_loss)
        value_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(value_grads, self.critic.trainable_variables))

    def losses(self):
        critic_loss = np.mean(self.critic_loss)
        actor_loss = np.mean(self.actor_loss)
        entropy_loss = np.mean(self.entropy_loss)
        return critic_loss, actor_loss, entropy_loss
