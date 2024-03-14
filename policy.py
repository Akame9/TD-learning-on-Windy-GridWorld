from collections import defaultdict
import random
from typing import Callable, Sequence, Tuple

import numpy as np

from env import Action, RandomWalkAction


def select_random_action():
    """
    Random selection policy
    """
    return random.choice(list(RandomWalkAction))

def create_epsilon_policy(Q: defaultdict, epsilon: float) -> Callable:
    """Creates an epsilon soft policy from Q values.

    A policy is represented as a function here because the policies are simple. More complex policies can be represented using classes.

    Args:
        Q (defaultdict): current Q-values
        epsilon (float): softness parameter
    Returns:
        get_action (Callable): Takes a state as input and outputs an action
    """

    def get_action(state: Tuple) -> int:
        p = np.random.random()
        if p < epsilon:
            action = random.choice(range(len(Action)))  
        else:
            action = argmax(Q[state])

        return action
    
    def argmax(arr: Sequence[float]) -> int:
    
        max_value = max(arr)
        argmax_i = [i for i, value in enumerate(arr) if value == max_value]
        random_index = random.choice(argmax_i)

        return random_index

    return get_action