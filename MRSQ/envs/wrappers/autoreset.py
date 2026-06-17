import gymnasium as gym
from typing import Any


class Autoreset(gym.Wrapper):
    """
    Brax-like AutoResetWrapper.
    When done, next obs will be the first obs of the next episode, the reward will be the last reward of the current episode.. 
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        done = terminated or truncated

        if done:
            obs, _ = self.env.reset()

        return obs, reward, terminated, truncated, info