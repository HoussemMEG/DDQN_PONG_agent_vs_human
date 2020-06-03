# Import dependencies
import tensorflow as tf
import numpy as np
import gym
import retro
import cv2 as cv
import matplotlib.pyplot as plt
from time import time, sleep
from random import sample, randint, uniform
import keyboard
import operator

# Training booleans
PLAYERS_NUMBER = 1
hi = False
IS_TRAINABLE = hi
IS_SHOW = not hi
IS_SAVEABLE = hi
IS_LOADABLE = True
enable_no_op = True

# Image and memory parameters
memory = []
MEMORY_MIN = 3000
MEMORY_MAX = 10000
IMAGE_WIDTH, IMAGE_HEIGHT, DEPTH = 84, 84, 4

# Training parameters
EPISODE_NUMBER = 10000
UPDATE = 10000
ACTION_REPEAT = 4
best_score_mean = -15

# Agent parameters
ACTION_SIZE = 3
epsilone = 0.47
EPSILONE_DECAY = 4e-6
EPSILONE_MIN = 0.1
BATCH_SIZE = 32
DISCOUT_FACTOR = 0.99
LEARNING_RATE = 0.0001
NO_OP_MAX = 30

# Function that I need to migrate
def execution_time(function):
    def my_function(*args, **kwargs):
        tic = time()
        result = function(*args, **kwargs)
        toc = time()
        if toc - tic > 1e-2:
            print("Temps d'éxecution de", function.__name__, ": {0:.3f} (s)".format(toc - tic))
        else:
            print("Temps d'éxecution de", function.__name__, ": {0:.3f} (ms)".format(1000 * (toc - tic)))
        return result
    return my_function

def action_encoder(action, enable):
    #model output 1 2 3
    if not enable :
        return action
    action_out = [0] * 8 * PLAYERS_NUMBER
    if PLAYERS_NUMBER == 1:
        if action == 2:
            action_out[4] = 1
        if action == 3:
            action_out[5] = 1

    elif PLAYERS_NUMBER == 2:
        action_out[0] = 1
        action_out[-1] = 1
    # in that case the output would be [1, 2]
        if action[0] == 2:
            action_out[4] = 1
        if action[0] == 3:
            action_out[5] = 1
        if action[1] == 2:
            action_out[6] = 1
        if action[1] == 3:
            action_out[7] = 1
    return action_out

def show(image):
    plt.imshow(image, cmap=plt.cm.gray)
    plt.show()

def pre_process(images):
    state = np.empty((4, 84, 84), dtype=np.float32)
    for index in range(images.shape[0]):
        image = cv.rectangle(images[index][35:194, :], (0, 0), (20, 160), (143, 73, 14), -1)
        image_gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        image = (cv.resize(image_gray, (84, 84)) / 255.0).astype(np.float32)
        state[index] = image
    return state

# pre processing without ommiting the second player part
# def pre_process(images):
#         state = np.empty((4, 84, 84), dtype=np.float32)
#         for index in range(images.shape[0]):
#             image = images[index][35:194, :]
#             image_gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
#             (_, bwimage) = cv.threshold(image_gray, 127, 255, cv.THRESH_BINARY)
#             bwimage = cv.rectangle(bwimage, (0, 0), (20, 160), (0, 0, 0), -1)
#             image = (cv.resize(bwimage, (84, 84)) / 255.0).astype(np.float32)
#             state[index] = image
#         return state

# @execution_time
def add_to_memory(state, observation, action, reward, done):
    if len(memory) >= MEMORY_MAX:
        memory.pop(0)
    next_state = np.append(np.delete(state, obj=0, axis=0), [observation], axis=0)
    memory.append([state, action, reward, next_state, done])
    return next_state

class Frameskip(gym.Wrapper):
    def __init__(self, env, skip=2):
        super().__init__(env)
        self._skip = skip
    def reset(self):
        return self.env.reset()
    def step(self, act):
        total_rew = 0.0
        done = None
        for i in range(self._skip):
            obs, rew, done, info = self.env.step(act)
            total_rew += rew
            if done:
                break
        return obs, total_rew, done, info

# Agent parameters
class DDQN():
    def __init__(self, EPSILONE_DECAY, EPSILONE_MIN, BATCH_SIZE, DISCOUT_FACTOR, LEARNING_RATE, \
                 INPUT_SHAPE=[84, 84, 4], ACTION_SIZE=3, FC_SIZE=512, epsilone=1):
        self.INPUT_SHAPE = INPUT_SHAPE
        self.ACTION_SIZE = ACTION_SIZE
        self.FC_SIZE = FC_SIZE
        self.epsilone = epsilone
        self.EPSILONE_DECAY = EPSILONE_DECAY
        self.EPSILONE_MIN = EPSILONE_MIN
        self.BATCH_SIZE = BATCH_SIZE
        self.DISCOUT_FACTOR = DISCOUT_FACTOR
        self.LEARNING_RATE = LEARNING_RATE

        self.model = self.model_creation()
        self.target_model = self.model_creation()
        self.loss_object = tf.keras.losses.Huber()
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.LEARNING_RATE)

    def model_creation(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(32, (8, 8), strides=4, input_shape=self.INPUT_SHAPE, activation='relu'))
        model.add(tf.keras.layers.Conv2D(64, (4, 4), strides=2, activation='relu'))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), strides=1, activation='relu'))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(self.FC_SIZE, activation='relu'))
        model.add(tf.keras.layers.Dense(self.ACTION_SIZE, activation='linear'))
        return model

    def learn(self, memory_batch):
        self.epsilone_update()
        states, targets = np.empty((0, 84, 84, 4), dtype=np.float32), np.empty((0, self.ACTION_SIZE), dtype=np.float32)
        for state, action, reward, next_state, done in memory_batch:
            state = np.rollaxis(pre_process(state), 0, 3)
            next_state = np.rollaxis(pre_process(next_state), 0, 3)
            target = reward
            state = np.expand_dims(state, axis=0)
            next_state = np.expand_dims(next_state, axis=0)
            if not done:
                target += self.DISCOUT_FACTOR * np.amax(self.target_model(next_state))
            target_f = self.model(state).numpy()
            target_f[0][action - 1] = target
            states = np.append(states, state, axis=0)
            targets = np.append(targets, target_f, axis=0)
        self.train(states, targets)

    @tf.function
    def train(self, states, targets):
        with tf.GradientTape() as tape:
            predictions = self.model(states)
            loss = self.loss_object(targets, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.train_loss(loss)

    def copy_model(self):
        self.target_model.set_weights(self.model.get_weights())
        print('...Model updated...')

    def load_model(self):
        try:
            self.model.load_weights('./model_weights.h5')
            print('...Model loaded...')
            self.copy_model()
        except:
            print('...No model to load...')

    def act(self, state):
        if uniform(0, 1) <= self.epsilone:
            return np.random.randint(1, 4)
        else:
            state = pre_process(state)
            state = np.rollaxis(state, 0, 3)
            state = np.expand_dims(state, axis=0)
        return np.argmax(self.model(state)) + 1

    def epsilone_update(self):
        if self.epsilone > self.EPSILONE_MIN:
            self.epsilone -= self.EPSILONE_DECAY
        else:
            self.epsilone = self.EPSILONE_MIN

    def save_model(self):
        self.model.save('drive/My Drive/Collab Notebook/my_model.h5')
        self.model.save_weights('drive/My Drive/Collab Notebook/model_weights.h5')
        print('...Model saved...')


agent = DDQN(EPSILONE_DECAY, EPSILONE_MIN, BATCH_SIZE, DISCOUT_FACTOR, LEARNING_RATE, \
             INPUT_SHAPE=[84, 84, 4], ACTION_SIZE=3, FC_SIZE=512, epsilone=epsilone)

# Games creation part
env = retro.make(game='Pong-Atari2600', players=PLAYERS_NUMBER,
                 use_restricted_actions=retro.Actions.ALL)
enable = True
env = Frameskip(env)

state = np.asarray([env.reset()] * DEPTH)

step = 0
scores = []

if IS_LOADABLE:
    agent.load_model()

# Initialize memory buffer
while len(memory) < MEMORY_MIN:
    action = np.random.randint(1, 4)  # Check action space for this [1 2 3]
    observation, reward, done, info = env.step(action_encoder(action, enable))
    state = add_to_memory(state, observation, action, reward, done)

for episode in range(EPISODE_NUMBER):
    state = np.asarray([env.reset()] * DEPTH)
    done = False
    episode_reward = 0
    if enable_no_op:
        for _ in range(9):
            obs, rew, done, info = env.step(action_encoder(2, enable))
        no_op = randint(1, NO_OP_MAX+1)
        i = 0
        for _ in range(no_op):
            observation, reward, done, info = env.step(action_encoder(1, enable))
            i += 1
            if i % 4 == 0:
                state = add_to_memory(state, observation, 1, 0, done)
                i = 0

    while not done:
        step += 1

        if IS_SHOW:
            env.render()
            sleep(0.01666)

        if IS_TRAINABLE:
            action = agent.act(state)
            reward_actions = 0
            for _ in range(ACTION_REPEAT):
                observation, reward, done, info = env.step(action_encoder(action, enable))
                reward_actions += reward
            episode_reward += reward_actions
            state = add_to_memory(state, observation, action, reward_actions, done)
            memory_batch = sample(memory, BATCH_SIZE)
            agent.learn(memory_batch)

            # Copy model to target model
            if step % UPDATE == 0:
                print('update model', step)
                agent.copy_model()
        else:
            #agent.epsilone = agent.EPSILONE_MIN
            agent.epsilone = 0.1
            action = agent.act(state)
            for _ in range(ACTION_REPEAT):
                if IS_SHOW:
                    env.render()
                    sleep(0.01)
                observation, reward, done, info = env.step(action_encoder(action, enable))
            state = add_to_memory(state, observation, action, reward, done)

        if IS_SAVEABLE and step % 9999 == 0:
            agent.save_model()

    # Sauvgarder le model si amélioration a la fin de l'épisode
    if IS_SAVEABLE:
        scores.append(episode_reward)
        avg_score = np.mean(scores[-5:])
        if best_score_mean <= avg_score:
            best_score_mean = avg_score
            agent.save_model()
            print("This episode score is", episode_reward, ", average score is", avg_score, "\n")

    # Display some infos
    print('Episode: %i, Reward: %i, Average reward: %.1f, Epsilone: %.3f, steps: %i' % (episode, episode_reward, avg_score, agent.epsilone, step))

env.close()
