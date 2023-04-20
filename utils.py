#!/usr/bin/env python
# coding: utf-8

import random
import numpy as np
import sys
from itertools import cycle
from shutil import get_terminal_size
from threading import Thread
from time import sleep

def invert_polarity(signal):
    return -1 * signal

def random_gain(signal, min_gain, max_gain):
    rand_gain = random.uniform(min_gain, max_gain)
    return rand_gain * signal

def noise(signal, noise_factor):
    noise = np.random.normal(0, signal.std(), signal.size)
    aug_signal = signal + noise * noise_factor
    return aug_signal


class Loader:
    def __init__(self, desc="Loading...", end="Done!", timeout=0.1):
        """
        A loader-like context manager

        Args:
            desc (str, optional): The loader's description. Defaults to "Loading...".
            end (str, optional): Final print. Defaults to "Done!".
            timeout (float, optional): Sleep time between prints. Defaults to 0.1.
        """
        self.desc = desc
        self.end = end
        self.timeout = timeout

        self._thread = Thread(target=self._animate, daemon=True)
        self.steps = ["⢿", "⣻", "⣽", "⣾", "⣷", "⣯", "⣟", "⡿"]
        self.done = False

    def start(self):
        self._thread.start()
        return self

    def _animate(self):
        for c in cycle(self.steps):
            if self.done:
                break
            print(f"\r{self.desc} {c}", flush=True, end="")
            sleep(self.timeout)

    def __enter__(self):
        self.start()

    def stop(self):
        self.done = True
        cols = get_terminal_size((80, 20)).columns
        print("\r" + " " * cols, end="", flush=True)
        print(f"\r{self.end}", flush=True)

    def __exit__(self, exc_type, exc_value, tb):
        # handle exceptions with those variables ^
        self.stop()
