from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
from algorithms import TD_0, exp_sarsa, generate_episode, generate_episodes, learning_targets, n_step_TD, nstep_sarsa, on_policy_mc_control_epsilon_soft, on_policy_mc_evaluation, q_learning, sarsa, td_prediction

from env import RandomWalk, States, WindyGridWorldEnv
from policy import create_epsilon_policy, select_random_action 

def calculate_rms_error(V: defaultdict, true_values: defaultdict):
    """
    Calculate the RMS error given the estimated values and the true values.
    """
    errors = np.array([V.get(s) - true_values.get(s) for s in true_values.keys()])
    rms_error = np.sqrt(np.mean(np.square(errors)))
    return rms_error

def rms_errors(all_Vs, ground_truth):
    rms_errors = []
    for V in all_Vs:
        error = calculate_rms_error(V,ground_truth)
        rms_errors.append(error)
    return rms_errors

def run_experiment(env, policy, true_values, alphas, ns, num_episodes, num_repetitions, samples):
    errors = np.zeros((len(ns), len(alphas)))
    
    for i, n in enumerate(ns):
        for j, alpha in enumerate(alphas):
            rms = 0
            #for _ in trange(num_repetitions, desc=f"n={n}, alpha={alpha}"):
            for sample in samples:
                V, all_Vs = n_step_TD(env, policy, alpha, gamma=1.0, n=n, num_episodes=num_episodes,all_episodes=sample) 
                rms+=sum(rms_errors(all_Vs, true_values))/num_episodes
            errors[i, j] = rms / num_repetitions
    return errors

def plot_timesteps_vs_episodes(ax, num_steps, all_episodes, trials, label):
    x = np.arange(num_steps)
    y = np.mean(all_episodes, axis=0)
    plt.plot(x, y, label=label)

    #Confidence bands
    std_error_curve = np.std(all_episodes, axis=0)
    std_error = (1.96 * std_error_curve) / np.sqrt(trials)
    ax.fill_between(x, y - std_error, y + std_error, alpha=0.3)

def plot_histogram(U_t,title):
    plt.hist(U_t, bins=30)
    plt.title(title)
    plt.show()

def main():
    
    #Q2
    alphas = np.linspace(0, 1, 100) 
    ns = [1 , 2, 4, 8, 16, 32, 64, 128, 256, 512]

    # Run the experiment
    num_episodes = 10
    num_repetitions = 100
    samples = []
    env = RandomWalk()
    for i in range(num_repetitions):
        samples.append(generate_episodes(env, select_random_action,num_episodes))
    #Ground Truths for reward = 0
    #ground_truth = {States.A: 0.05, States.B: 0.1, States.C: 0.15, States.D: 0.2, States.E: 0.25,
    #                States.F: 0.3,States.G: 0.35,States.H:0.4,States.I:0.45,States.J: 0.5,
    #                States.K: 0.55, States.M:0.6, States.N: 0.65, States.O:0.7, States.P: 0.75,
    #                States.Q:0.8, States.S:0.85, States.T:0.9, States.U: 0.95}
    ground_truth = {States.A: -0.9, States.B: -0.8, States.C: -0.7, States.D: -0.6, States.E: -0.5,
                    States.F: -0.4,States.G: -0.3,States.H:-0.2,States.I:-0.1,States.J: 0.0,
                    States.K: 0.1, States.M:0.2, States.N: 0.3, States.O:0.4, States.P: 0.5,
                    States.Q:0.6, States.S:0.7, States.T:0.8, States.U: 0.9}
    errors = run_experiment(env, select_random_action, ground_truth, alphas, ns, num_episodes, num_repetitions,samples)

    # Plot the results
    plt.figure(figsize=(10, 6))
    for i, n in enumerate(ns):
        y = []
        for e in errors[i]:
            if e>0.55:
                e = 0.55
            y.append(e)
        plt.plot(alphas, y, label=f"n={n}")
    plt.xlabel('α (alpha)')
    plt.ylabel('Average RMS error over 19 states and first 10 episodes')
    plt.title('Performance of n-step TD methods')
    plt.legend()
    plt.show()
    

    #Q3
    env = WindyGridWorldEnv()
    num_trials = 10
    num_steps = 8000
    sarsa_episodes = []
    q_learn_episodes = []
    expected_sarsa = []
    n_step_sarsa = []
    on_policy_mc = []
    for i in range(num_trials):
        Q, episodes_1 = sarsa(env,num_steps,1.0,0.1,0.5)
        Q, episodes_2 = exp_sarsa(env,num_steps,1.0,0.1,0.5)
        Q, episodes_3 = q_learning(env,num_steps,1.0,0.1,0.5)
        Q, episodes_4 = nstep_sarsa(env,num_steps,1.0,0.1,0.5)
        Q, episodes_5 = on_policy_mc_control_epsilon_soft(env,num_steps,1.0,0.1)
        sarsa_episodes.append(episodes_1)
        expected_sarsa.append(episodes_2)
        q_learn_episodes.append(episodes_3)
        n_step_sarsa.append(episodes_4)
        on_policy_mc.append(episodes_5)

    #plt.figure(figsize=(10, 6))
    _, ax = plt.subplots()
    plot_timesteps_vs_episodes(ax, num_steps, sarsa_episodes, num_trials,"sarsa")
    plot_timesteps_vs_episodes(ax, num_steps, expected_sarsa, num_trials,"exp_sarsa")
    plot_timesteps_vs_episodes(ax, num_steps, q_learn_episodes, num_trials,"q_learning")
    plot_timesteps_vs_episodes(ax, num_steps, n_step_sarsa, num_trials,"n_step_sarsa")
    plot_timesteps_vs_episodes(ax, num_steps, on_policy_mc, num_trials,"on_policy_mc")
    plt.xlabel('Time Steps')
    plt.ylabel('Episodes')
    plt.title('Performance of Different Methods')
    plt.legend()
    plt.show()

    
    #Q4
    env = WindyGridWorldEnv()
    num_steps = 8000
    Q, _ = q_learning(env,num_steps,1.0,0.1,0.5)
    policy = create_epsilon_policy(Q, 0.1)
    n_episodes = []
    N = [1,10,50]  
    for i in range(len(N)):
        episodes = []
        for _ in range(N[i]):
            episode = generate_episode(env, policy)
            episodes.append(episode)
        n_episodes.append(episodes)
    evaluation_episodes = []
    evaluations = 100
    for i in range(evaluations):
        episode = generate_episode(env, policy)
        evaluation_episodes.append(episode)
    
    for i in range(len(N)):
        V1 = td_prediction(env,1,n_episodes[i],0.5,n=1)
        V2 = td_prediction(env,1,n_episodes[i],0.5,n=4)
        V3 = on_policy_mc_evaluation(env,n_episodes[i],1,0.1)
        U_t1 = learning_targets(V1,1,evaluation_episodes,n=1)
        U_t2 = learning_targets(V2,1,evaluation_episodes,n=4)
        U_t3 = learning_targets(V3, 1, evaluation_episodes, n=None)
        plot_histogram(U_t1, "TD(0) : N = "+str(N[i]))
        plot_histogram(U_t2, "nstep-TD(n=4) : N = "+str(N[i]))
        plot_histogram(U_t3, "MC : N = "+str(N[i]))
    plt.ylabel('Evaluation Episodes')
    plt.xlabel('Learning targets')


if __name__ =="__main__":
    main()
