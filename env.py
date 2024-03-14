from enum import IntEnum
import random
from typing import Tuple, Optional, List
from gym import Env, spaces
from gym.utils import seeding
from gym.envs.registration import register

class RandomWalkAction(IntEnum):
    """Action"""

    LEFT = -1
    RIGHT = 1

class States(IntEnum):

    L = 0
    A = 1
    B = 2
    C = 3
    D = 4
    E = 5
    F = 6
    G = 7
    H = 8
    I = 9
    J = 10
    K = 11
    M = 12
    N = 13
    O = 14
    P = 15
    Q = 16
    S = 17
    T = 18
    U = 19
    R = 20

class RandomWalk():
    def __init__(self):
        self.agent_pos = States.J
        
    def reset(self):
        self.agent_pos = States.J
        return self.agent_pos
    
    def step(self, action: RandomWalkAction) -> Tuple[Tuple[int, int], float, bool]:

        reward = 0
        done = False
        self.agent_pos += action
        if self.agent_pos == States.L:
            done = True
            reward = -1
        elif self.agent_pos == States.R:
            done = True
            reward = 1
        return self.agent_pos, reward, done


def register_env() -> None:
    """Register custom gym environment so that we can use `gym.make()`

    In your main file, call this function before using `gym.make()` to use the Four Rooms environment.
        register_env()
        env = gym.make('WindyGridWorld-v0')

    There are a couple of ways to create Gym environments of the different variants of Windy Grid World.
    1. Create separate classes for each env and register each env separately.
    2. Create one class that has flags for each variant and register each env separately.

        Example:
        (Original)     register(id="WindyGridWorld-v0", entry_point="env:WindyGridWorldEnv")
        (King's moves) register(id="WindyGridWorldKings-v0", entry_point="env:WindyGridWorldEnv", **kwargs)

        The kwargs will be passed to the entry_point class.

    3. Create one class that has flags for each variant and register env once. You can then call gym.make using kwargs.

        Example:
        (Original)     gym.make("WindyGridWorld-v0")
        (King's moves) gym.make("WindyGridWorld-v0", **kwargs)

        The kwargs will be passed to the __init__() function.

    Choose whichever method you like.
    """
    # TODO


class Action(IntEnum):
    """Action"""

    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3
    #King's Moves
    #D_UP_LEFT = 4
    #D_UP_RIGHT = 5
    #D_DOWN_RIGHT = 6
    #D_DOWN_LEFT = 7
    #NO_MOVE = 8


def actions_to_dxdy(action: Action) -> Tuple[int, int]:
    """
    Helper function to map action to changes in x and y coordinates
    Args:
        action (Action): taken action
    Returns:
        dxdy (Tuple[int, int]): Change in x and y coordinates
    """
    mapping = {
        Action.LEFT: (-1, 0),
        Action.DOWN: (0, -1),
        Action.RIGHT: (1, 0),
        Action.UP: (0, 1),
        #King's Moves
        #Action.D_UP_LEFT: (-1,1),
        #Action.D_UP_RIGHT: (1,1),
        #Action.D_DOWN_LEFT: (-1,-1),
        #Action.D_DOWN_RIGHT: (1,-1),
        #Action.NO_MOVE: (0,0)

    }
    return mapping[action]


class WindyGridWorldEnv(Env):
    def __init__(self):
        """Windy grid world gym environment
        This is the template for Q4a. You can use this class or modify it to create the variants for parts c and d.
        """

        # Grid dimensions (x, y)
        self.rows = 10
        self.cols = 7

        # Wind
        self.wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(self.rows), spaces.Discrete(self.cols))
        )

        # Set start_pos and goal_pos
        self.start_pos = (0, 3)
        self.goal_pos = (7, 3)
        self.agent_pos = None

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Fix seed of environment

        In order to make the environment completely reproducible, call this function and seed the action space as well.
            env = gym.make(...)
            env.seed(seed)
            env.action_space.seed(seed)

        This function does not need to be used for this assignment, it is given only for reference.
        """

        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.agent_pos = self.start_pos
        return self.agent_pos

    def step(self, action: Action) -> Tuple[Tuple[int, int], float, bool, dict]:
        """Take one step in the environment.

        Takes in an action and returns the (next state, reward, done, info).
        See https://github.com/openai/gym/blob/master/gym/core.py#L42-L58 foand r more info.

        Args:
            action (Action): an action provided by the agent

        Returns:
            observation (object): agent's observation after taking one step in environment (this would be the next state s')
            reward (float) : reward for this transition
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning). Not used in this assignment.
        """

        # TODO
        reward = -1
        done = False
        
        next_dxdy = actions_to_dxdy(action)
        next_pos = tuple(map(sum, zip(self.agent_pos,next_dxdy)))
        if 0 <= next_pos[0] < self.rows : 
            wind_strength = self.wind[self.agent_pos[0]]
            #stochastic winds
            #if wind_strength!=0:
            #    stochastic_winds = [wind_strength-1,wind_strength,wind_strength+1]
            #    wind_strength = random.choice(stochastic_winds)

            new_y = max(min(next_pos[1] + wind_strength, self.cols - 1), 0)
            next_pos = (next_pos[0],new_y)
            if 0 <= next_pos[1] < self.cols:
                self.agent_pos = next_pos
        
        if self.agent_pos == self.goal_pos:
            done = True
            reward = 0.0

        return self.agent_pos, reward, done, {}
