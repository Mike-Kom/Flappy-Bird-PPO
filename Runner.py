import numpy as np
import gym
from collections import deque
from inputs import jump, click
import time
from Window_capture import capture_window, process, main_menu, game_over
import cv2
from collections import deque
from numba import jit

buffer_time = deque(maxlen=500)


def timer(function):
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = function(*args, **kwargs)
        t2 = time.time()
        print(f"Time spent on {function.__name__} = {t2 - t1}")
        buffer_time.append(t2 - t1)
        print(f"Average time = {np.mean(buffer_time)}, max = {np.max(buffer_time)}")

        return result

    return wrapper


class Runner:
    def __init__(self, agent, env_name):
        self.agent = agent
        self.env_name = env_name
        self.score_history = []
        self.avg_score_history = []
        self.avg_score = -100
        self.score = 0
        self.buffer = deque(maxlen=int(5e4))
        self.windowname = "Flappy Bird New"
        self.last_frame = None
        self.length = None

    def game_start(self):
        screenshot = capture_window(processing=False)
        if game_over(screenshot):
            jump()
            time.sleep(0.03)
            jump()
        elif main_menu(screenshot):
            jump()
        else:
            raise KeyError("game_start() while playing")
        time.sleep(0.05)
        jump()
        time.sleep(0.2)
        for x in range(5):
            jump()
            time.sleep(0.1)

    @jit(nopython=False, cache=True)
    def step(self, action):
        t1 = time.time()
        screenshot = capture_window(processing=False)
        done = game_over(screenshot)
        if done:
            action = 0
        screenshot = process(screenshot)
        if done:
            assert self.last_frame.any() != None, "runner.last frame used before assignment"
            screenshot = self.last_frame
            reward = 0
        else:
            self.last_frame = screenshot
            if action == 1:
                jump()
            if self.length > 24:
                reward = 2
            else:
                reward = 0.025

        while time.time() - t1 < 0.04:
            pass

        return np.array(screenshot, dtype=np.float32), reward, done

    def force_game_over(self):
        screenshot = capture_window(processing=False)
        done = game_over(screenshot)
        if done:
            return 0
        if not done:
            jump()
            time.sleep(0.05)
            jump()
            time.sleep(0.05)
        countre = 0
        while not done:
            time.sleep(2)
            if main_menu():
                done = True
                break
            screenshot = capture_window(processing=False)
            done = game_over(screenshot)
            countre += 1
            if countre > 2:
                cv2.imwrite("img\out.bmp", screenshot[270:280, 760:900, :])
                raise KeyError("Cannot restart the game")

    def open_main_menu(self):
        attempt = 0
        while attempt < 2:
            screenshot = capture_window(processing=False)
            if main_menu(screenshot):
                return 0
            elif game_over(screenshot):
                jump()
                if main_menu():
                    return 0
            else:
                time.sleep(2)
                attempt += 1
        screenshot = capture_window(processing=False)
        cv2.imwrite("img\out.bmp", screenshot)
        raise KeyError("Cannot open main menu")

    def first_state(self):
        self.open_main_menu()
        self.game_start()
        action = 1
        frame, reward, done = self.step(action)
        state = np.expand_dims(frame, axis=0)
        state = np.expand_dims(state, axis=0)
        assert state.shape == (1, 1, 128, 128)
        for i in range(3):
            next_frame, rewards, dones = self.step(0)
            next_frame = np.expand_dims(next_frame, axis=0)
            next_frame = np.expand_dims(next_frame, axis=0)
            state = np.column_stack((state, next_frame))
        assert state.shape == (1, 4, 128, 128), f"States shape is {state.shape}"
        # cv2.imwrite("1.jpg", state[0][0]*255)
        # cv2.imwrite("2.jpg", state[0][1]*255)
        # cv2.imwrite("3.jpg", state[0][2]*255)
        # cv2.imwrite("4.jpg", state[0][3]*255)
        # exit()
        return np.array(state, dtype=np.float32)

    def next_state(self, action):
        # if main_menu():
        #     raise KeyError("Unexpected main_menu on the next_state call")
        frame, rewards, done = self.step(action)
        dones = [done]
        next_state = np.expand_dims(frame, axis=0)
        next_state = np.expand_dims(next_state, axis=0)
        assert next_state.shape == (1, 1, 128, 128)
        for i in range(3):
            next_frame, reward, done = self.step(0)
            rewards += reward
            dones.append(done)
            next_frame = np.expand_dims(next_frame, axis=0)
            next_frame = np.expand_dims(next_frame, axis=0)
            next_state = np.column_stack((next_state, next_frame))
        if any(dones):
            done = True
        else:
            done = False
        assert next_state.shape == (1, 4, 128, 128), f"States shape is {next_state.shape}"
        next_state = np.array(next_state, dtype=np.float32)
        rewards = np.clip(rewards, -1, 2)
        if done:
            rewards = 0
        # cv2.imwrite("1.jpg", next_state[0][0]*255)
        # cv2.imwrite("2.jpg", next_state[0][1]*255)
        # cv2.imwrite("3.jpg", next_state[0][2]*255)
        # cv2.imwrite("4.jpg", next_state[0][3]*255)
        # exit()
        return next_state, rewards, done
