import gym
from typing import Callable, List, Optional
from collections import defaultdict
import numpy as np
from tqdm import trange

from env import RandomWalk, States, WindyGridWorldEnv
from policy import create_epsilon_policy

#Q3
def initialize_V(V: defaultdict):
    V = {States.L: 0,States.A: 0, States.B: 0, States.C: 0, States.D: 0, States.E: 0,
                    States.F: 0,States.G: 0,States.H:0,States.I:0,States.J: 0,
                    States.K: 0, States.M:0, States.N: 0, States.O:0, States.P: 0,
                    States.Q:0, States.S:0, States.T:0, States.U: 0, States.R: 0}
    return V

def generate_episodes(env: RandomWalk, policy: Callable, num_episodes: int):
    all_episodes= []
    for _ in trange(num_episodes, desc="Episode"):
        state = env.reset()
        episode = [] 
        while True:
            action = policy()
            next_state, reward, done = env.step(action)
            episode.append((next_state, reward, done))
            if done:
                break
        all_episodes.append(episode)
    return all_episodes

#Q3
def TD_0(env: RandomWalk, policy: Callable, step_size: float, gamma: float, num_episodes: int):

    V = defaultdict(float)
    V = initialize_V(V)
    all_Vs = []
    all_episodes= []
    for _ in trange(num_episodes, desc="Episode"):
        state = env.reset()
        episode = []
        while True:
            action = policy()
            next_state, reward, done = env.step(action)
            episode.append((next_state, reward, done))
            V[state] = V.get(state) + step_size * (reward + (gamma * V.get(next_state)) - V.get(state))
            state = next_state
            if done:
                break
        all_episodes.append(episode)
        all_Vs.append(V.copy())
            
    return V, all_Vs, all_episodes

def n_step_TD(env: RandomWalk, policy: Callable, alpha: float, gamma: float, n: int, num_episodes: int, all_episodes: List):
    V = defaultdict(float)
    V = initialize_V(V)
    all_Vs = []

    for episode in all_episodes:
        states = [None] * (n + 1)
        rewards = [0] * (n + 1)
        T = len(episode) 
        tau = 0
        t = 0
        state = env.reset()
        states[t % (n + 1)] = state

        while tau != T - 1:
            if t < T:
                next_state, reward, done = episode[t] 
                states[(t + 1) % (n + 1)] = next_state
                rewards[(t + 1) % (n + 1)] = reward

                if done:
                    T = t + 1
            
            tau = t - n + 1
            if tau >= 0:
                G = sum([gamma ** (i - tau - 1) * rewards[(i) % (n + 1)] 
                         for i in range(tau+1, min(tau+n, T)+1)])
                if tau + n < T:
                    G += (gamma ** n) * V[states[(tau + n) % (n + 1)]]
                V[states[tau % (n + 1)]] += alpha * (G - V[states[tau % (n + 1)]])
            
            t += 1
        all_Vs.append(V.copy())

    return V, all_Vs

def sarsa(env: gym.Env, num_steps: int, gamma: float, epsilon: float, step_size: float):
    """SARSA algorithm.

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = create_epsilon_policy(Q, epsilon)
    state = env.reset()
    action = policy(state)
    episode_count = 0
    epsiodes = []
    for _ in trange(num_steps, desc="Steps"):
        
        next_state, reward, done, _ = env.step(action)
        next_action = policy(next_state)
        Q[state][action] = Q[state][action] + step_size * ((reward + (gamma * Q[next_state][next_action])) - Q[state][action])
        state = next_state
        action = next_action
        policy = create_epsilon_policy(Q, epsilon)
        if done:
            episode_count+=1
            state = env.reset()
            action = policy(state)
        epsiodes.append(episode_count)

    return Q, epsiodes


def nstep_sarsa(
    env: gym.Env,
    num_steps: int,
    gamma: float,
    epsilon: float,
    step_size: float,
):
    """N-step SARSA

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = create_epsilon_policy(Q, epsilon)
    n = 4
    states = [None] * (n + 1)
    rewards = [0] * (n + 1)
    actions = [None] * (n+1)
    episode_count = 0
    epsiodes = []
    while len(epsiodes)<num_steps:
        T = num_steps-len(epsiodes)
        t = 0
        tau = 0
        state = env.reset()
        states[t % (n + 1)] = state
        actions[t % (n+1)] = policy(state)
        while tau != T - 1:
            if t < T:
                action = actions[t % (n+1)]
                next_state, reward, done, _ = env.step(action)
                states[(t + 1) % (n + 1)] = next_state
                rewards[(t + 1) % (n + 1)] = reward

                if done:
                    episode_count+=1
                    state = env.reset()
                    actions[(t + 1) % (n + 1)] = policy(state)
                    T = t + 1
                else:
                    actions[(t + 1) % (n + 1)] = policy(next_state)
                epsiodes.append(episode_count)
            tau = t - n + 1
            if tau >= 0:
                G = sum([gamma ** (i - tau - 1) * rewards[(i) % (n + 1)] 
                            for i in range(tau+1, min(tau+n, T)+1)])
                if tau + n < T:
                        G += (gamma ** n) * Q[states[(tau + n) % (n + 1)]][actions[(tau + n) % (n + 1)]]
                Q[states[tau % (n + 1)]][actions[tau % (n + 1)]] += step_size * (G - Q[states[tau % (n + 1)]][actions[tau % (n + 1)]])
                policy = create_epsilon_policy(Q, epsilon)
            t += 1
            

    return Q, epsiodes
    

def policy_fn(Q, observation, epsilon, num_actions):
    A = np.ones(num_actions, dtype=float) * epsilon / num_actions
    best_action = np.argmax(Q[observation])
    A[best_action] += (1.0 - epsilon)
    return A
    

def exp_sarsa(
    env: gym.Env,
    num_steps: int,
    gamma: float,
    epsilon: float,
    step_size: float,
):
    """Expected SARSA

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = create_epsilon_policy(Q, epsilon)
    state = env.reset()
    episode_count = 0
    epsiodes = []
    for _ in trange(num_steps, desc="Steps"):
        action = policy(state)
        next_state, reward, done, _ = env.step(action)
        next_action_probs = policy_fn(Q, next_state, epsilon, env.action_space.n)
        expected_value_next_state = np.sum(next_action_probs * Q[next_state])
        Q[state][action] += step_size * (reward + gamma * expected_value_next_state - Q[state][action])
        state = next_state
        policy = create_epsilon_policy(Q, epsilon)
        if done:
            episode_count+=1
            state = env.reset()
        epsiodes.append(episode_count)

    return Q, epsiodes


def q_learning(
    env: gym.Env,
    num_steps: int,
    gamma: float,
    epsilon: float,
    step_size: float,
):
    """Q-learning

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = create_epsilon_policy(Q, epsilon)
    state = env.reset()
    episode_count = 0
    epsiodes = []
    for _ in trange(num_steps, desc="Steps"):
        action = policy(state)
        next_state, reward, done, _ = env.step(action)
        Q[state][action] = Q[state][action] + step_size * ((reward + (gamma * max(Q[next_state]))) - Q[state][action])
        state = next_state
        policy = create_epsilon_policy(Q, epsilon)
        if done:
            episode_count+=1
            state = env.reset()
        epsiodes.append(episode_count)

    return Q, epsiodes

def generate_episode(env: gym.Env, policy: Callable, es: bool = False):
    """A function to generate one episode and collect the sequence of (s, a, r) tuples

    This function will be useful for implementing the MC methods

    Args:
        env (gym.Env): a Gym API compatible environment
        policy (Callable): A function that represents the policy.
        es (bool): Whether to use exploring starts or not
    """
    episode = []
    state = env.reset()
    counter = 0
    while counter<=8000: #459
        counter+=1
        action = policy(state)
        next_state, reward, done, _ = env.step(action)
        episode.append((state, action, reward))
        if done:
            break
        state = next_state
        
    return episode

def on_policy_mc_control_epsilon_soft(
    env: gym.Env, num_steps: int, gamma: float, epsilon: float
):
    """On-policy Monte Carlo policy control for epsilon soft policies.

    Args:
        env (gym.Env): a Gym API compatible environment
        num_episodes (int): Number of episodes
        gamma (float): Discount factor of MDP
        epsilon (float): Parameter for epsilon soft policy (0 <= epsilon <= 1)
    Returns:

    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = create_epsilon_policy(Q, epsilon)
    episode_count = 0
    episodes = []
    step_count = 0
    while step_count <= num_steps:
        episode = generate_episode(env, policy)
        episodes +=[episode_count] * len(episode)
        episode_count+=1
        G = 0
        for t in range(len(episode) - 1, -1, -1):
            G = gamma * G + episode[t][2]
            curr_state = episode[t][0]
            curr_action = episode[t][1]
            if (curr_state, curr_action) not in [(x[0], x[1]) for x in episode[:t]]:
                N[curr_state][curr_action] += 1
                Q[curr_state][curr_action] = Q[curr_state][curr_action] + (1/N[curr_state][curr_action]) * (G - Q[curr_state][curr_action])
        policy = create_epsilon_policy(Q, epsilon)
        step_count = len(episodes)

    return Q,episodes[:num_steps]

def td_prediction(env: gym.Env, gamma: float, episodes, alpha, n=1) -> defaultdict:
    """TD Prediction

    This generic function performs TD prediction for any n >= 1. TD(0) corresponds to n=1.

    Args:
        env (gym.Env): a Gym API compatible environment
        gamma (float): Discount factor of MDP
        episodes : the evaluation episodes. Should be a sequence of (s, a, r) tuples or a dict.
        n (int): The number of steps to use for TD update. Use n=1 for TD(0).
    """
    V = defaultdict(float)
    for i in range(env.rows):
        for j in range(env.cols):
            V[(i,j)]=0
    for episode in episodes:
        states = [None] * (n + 1)
        rewards = [0] * (n + 1)
        T = len(episode)-1
        tau = 0
        t = 0
        state = env.reset()
        states[t % (n + 1)] = state
        while tau != T - 1:
            if t < T:
                state, action, reward = episode[t]
                states[(t + 1) % (n + 1)] = episode[t+1][0] #next_state
                rewards[(t + 1) % (n + 1)] = reward
            
            tau = t - n + 1
            if tau >= 0:
                G = sum([gamma ** (i - tau - 1) * rewards[(i) % (n + 1)] 
                         for i in range(tau+1, min(tau+n, T)+1)])
                if tau + n < T:
                    G += (gamma ** n) * V[states[(tau + n) % (n + 1)]]
                V[states[tau % (n + 1)]] += alpha * (G - V[states[tau % (n + 1)]])
            t += 1
    print("V : ",V)
    return V

"""
def TD_0_evaluation(env: WindyGridWorldEnv, gamma: float, episodes, alpha):

    V = defaultdict(float)
    for i in range(env.rows):
        for j in range(env.cols):
            V[(i,j)]=0
    for episode in episodes:
        for i in range(len(episode)-1):
            state, action, reward = episode[i]
            next_state = episode[i+1][0]
            #V[state] = reward + V.get(next_state)
            V[state] = V.get(state) + alpha * (reward + (gamma * V.get(next_state)) - V.get(state))
    print("V : ", V)   
    print("V[0]: ",V[(0,3)])  
    return V
"""


def on_policy_mc_evaluation(
    env: gym.Env, episodes: List, gamma: float, epsilon: float
):

    V = defaultdict(float)
    N = defaultdict(int)

    for episode in episodes:
        G = 0
        print("Episode : ",len(episode))
        for t in range(len(episode) - 1, -1, -1):
            G = gamma * G + episode[t][2]
            curr_state = episode[t][0]
            visited = False
            for i in range(0,t):
                if curr_state == episode[i][0]:
                    visited = True
                    break
            if not visited:
                N[curr_state] += 1
                V[curr_state] = V[curr_state] + (1/N[curr_state] * (G - V[curr_state]))
    return V

def learning_targets(
    V: defaultdict, gamma: float, episodes, n: Optional[int] = None
) -> np.ndarray:
    """Compute the learning targets for the given evaluation episodes.

    This generic function computes the learning targets for Monte Carlo (n=None), TD(0) (n=1), or TD(n) (n=n).

    Args:
        V (defaultdict) : A dict of state values
        gamma (float): Discount factor of MDP
        episodes : the evaluation episodes. Should be a sequence of (s, a, r) tuples or a dict.
        n (int or None): The number of steps for the learning targets. Use n=1 for TD(0), n=None for MC.
    """
    # TODO
    targets = np.zeros(len(episodes))
    return_count = 0
    for episode in episodes:
        expected_return=0
        if n!=None:
            for i in range(n):
                reward = episode[i][2]
                expected_return+=reward
            next_state = episode[n][0]
            if(next_state not in V.keys()):
                    V[next_state] = 0
            expected_return += gamma * V.get(next_state)
        else:
            for i in range(len(episode)):
                reward = episode[i][2]
                expected_return+=reward
        targets[return_count] = expected_return
        return_count+=1
    return targets
