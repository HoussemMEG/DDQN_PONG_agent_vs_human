# Import dependencies
import tensorflow as tf
import numpy as np
import gym
import retro
import cv2 as cv
from time import time, sleep
import keyboard
from PIL import Image
from random import sample, randint, uniform
import operator

# interface images
img1=cv.imread('1.png')
img2=cv.imread('2.png')
img3=cv.imread('3.png')
img4=cv.imread('4.png')
img5=cv.imread('5.png')
img6=cv.imread('6.png')
img1 = cv.resize(img1,(900,700))
img2 = cv.resize(img2,(900,700))
img3 = cv.resize(img3,(900,700))
img4 = cv.resize(img4,(900,700))
img5 = cv.resize(img5,(900,700))
img6 = cv.resize(img6,(900,700))



# Training booleans
PLAYERS_NUMBER = 2
enable_no_op = False
END_REWARD = 21
END_SLEEP = 3

# Image and memory parameters
IMAGE_WIDTH, IMAGE_HEIGHT, DEPTH = 84, 84, 4

# Agent parameters
ACTION_REPEAT = 4
EPSILONE_MIN = 0.0
NO_OP_MAX = 20


def Mode():
    i = 0
    mode = 0
    while not keyboard.is_pressed('enter'):
        if keyboard.is_pressed('DOWN'):
            i += 1
        if keyboard.is_pressed('UP'):
            i -= 1
        if i > 3:
            i = 0
        if i < 0:
            i = 3

        if i == 0:
            cv.imshow('', img1)
            mode = 1
        elif i == 1:
            cv.imshow('', img2)
            mode = 2
        elif i == 2:
            cv.imshow('', img3)
            mode = 3
        else:
            cv.imshow('', img4)
            mode = 4
        cv.waitKey(0)
    return mode


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


def action_encoder(action):
    #model output 1 2 3
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

#hichem pre process
def pre_process(images):
    state = np.empty((4, 84, 84), dtype=np.float32)
    for index in range(images.shape[0]):
        image = cv.rectangle(images[index][35:194, :], (0, 0), (20, 160), (143, 73, 14), -1)
        image_gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        image = (cv.resize(image_gray, (84, 84)) / 255.0).astype(np.float32)
        state[index] = image
    return state

# def pre_process(images):
#     state = np.empty((4, 84, 84), dtype=np.float32)
#     for index in range(images.shape[0]):
#         image = images[index][35:194, :]
#         image_gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
#         (_, bwimage) = cv.threshold(image_gray, 127, 255, cv.THRESH_BINARY)
#         bwimage = cv.rectangle(bwimage, (0, 0), (20, 160), (0, 0, 0), -1)
#         image = (cv.resize(bwimage, (84, 84)) / 255.0).astype(np.float32)
#         state[index] = image
#     return state

class Frameskip(gym.Wrapper):
    def __init__(self, env, skip=2):
        super().__init__(env)
        self._skip = skip
    def reset(self):
        return self.env.reset()
    def step(self, act):
        total_rew = [0.0, 0.0]
        done = None
        for i in range(self._skip):
            obs, rew, done, info = self.env.step(act)
            total_rew = list(map(operator.add, total_rew, rew))
            if done:
                break
        return obs, total_rew, done, info


# Agent parameters
class DDQN():
    def model_load(self, difficulty):
        try:
            FRAME_SLEEP = 0.01
            if difficulty == 1:
                self.model = tf.keras.models.load_model('./my_model.h5', compile=False)
                print('...Model {} loaded...'.format(difficulty))
            elif difficulty == 3:
                self.model = tf.keras.models.load_model('./my_model.h5', compile=False)
                print('...Model {} loaded...'.format(difficulty))
            elif difficulty == 4:
                FRAME_SLEEP = FRAME_SLEEP / 2
                self.model = tf.keras.models.load_model('./my_model.h5', compile=False)
                print('...Model {} loaded...'.format(difficulty))
            else:
                self.model = tf.keras.models.load_model('./my_model.h5', compile=False)
                print('...Model {} loaded...'.format(2))
            return FRAME_SLEEP
        except:
            print('...No model to load...')

    def act(self, state):
        if uniform(0, 1) <= EPSILONE_MIN:
            return np.random.randint(1, 4)
        else:
            state = np.expand_dims(np.rollaxis(pre_process(state), 0, 3), axis=0)
        return np.argmax(self.model(state)) + 1


# Creation of the agent
agent = DDQN()

# Games creation part
env = retro.make(game='Pong-Atari2600', players=PLAYERS_NUMBER,
                 use_restricted_actions=retro.Actions.ALL)
env = Frameskip(env)

while True:
    FRAME_SLEEP = agent.model_load(Mode())
    episode_reward = [0.0, 0.0]
    done = False
    while not done:
        state = np.asarray([env.reset()] * DEPTH)
        # initialise l'environnement (paddle at the middle)
        for _ in range(9):
            observation, reward, done, info = env.step(action_encoder([2, 2]))

        # Noop part (maybe we should disable this to see)
        if enable_no_op:
            no_op = randint(1, NO_OP_MAX + 1)
            i = 0
            for _ in range(no_op):
                observation, reward, done, info = env.step(action_encoder([1, 1]))
                i += 1
                if i % 4 == 0:
                    state = np.append(np.delete(state, obj=0, axis=0), [observation], axis=0)
                    i = 0

        # here the boi is playing
        while not done:
            action = agent.act(state)
            for _ in range(ACTION_REPEAT):
                tic = time()
                left_up = keyboard.is_pressed('UP')
                left_down = keyboard.is_pressed('DOWN')
                player_action = 1
                if left_up:
                    player_action = 2
                if left_down:
                    player_action = 3
                observation, reward, done, info = env.step(action_encoder([action, player_action]))

                if reward[0] == 1:
                    episode_reward[0] += 1
                elif reward[1] == 1:
                    episode_reward[1] += 1

                img = Image.fromarray(observation, "RGB")
                img = img.resize((900, 700))
                img = cv.cvtColor(np.array(img), cv.COLOR_RGB2GRAY)
                (_, img) = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
                toc = time()
                try:
                    sleep(FRAME_SLEEP - (toc - tic))
                except:
                    pass
                sleep(FRAME_SLEEP)
                cv.imshow('', np.array(img))
                if cv.waitKey(1) & keyboard.is_pressed('ESC'):
                    done = True
                    break

                if episode_reward[0] == END_REWARD:
                    done = True
                    cv.imshow('', img5)
                    cv.waitKey(1)
                    sleep(END_SLEEP)
                    break
                if episode_reward[1] == END_REWARD:
                    done = True
                    cv.imshow('', img6)
                    cv.waitKey(1)
                    sleep(END_SLEEP)
                    break
            state = np.append(np.delete(state, obj=0, axis=0), [observation], axis=0)
