# MIT License
# Copyright (c) 2025 Brett Daley and Prabhat Nagarajan
# Copyright (c) 2026 Prabhat Nagarajan
#
# Portions derived from https://github.com/prabhatnagarajan/reg-duel-q:
# `generate_step_sizes`

import os
import table_rl
import numpy as np
from pdb import set_trace
import matplotlib.pyplot as plt

from datetime import datetime

import matplotlib.cm as cm
import matplotlib.colors as colors

import multiprocessing
import matplotlib.patches as mpatches

plt.rcParams["font.family"] = "Georgia"
plt.rcParams["font.serif"] = ["Georgia"]

# Define normalization and colormap
norm = colors.Normalize(vmin=-1, vmax=5)
cmap = cm.plasma  # try: plasma, inferno, magma, tab10


def value_to_color(x):
    return cmap(norm(x))


def generate_step_sizes():
    start = -6
    end = 0
    incr = 0.1
    n = int((end - start) / incr) + 1
    alogs = np.linspace(start, end, n)
    return tuple([round(np.exp(x), 4) for x in alogs])


class MeanResidualQLearning(table_rl.learners.ExpectedSarsa):
    """Class that implements our method"""

    def __init__(
        self,
        num_states,
        num_actions,
        step_size_schedule,
        explorer,
        multiplier,
        discount=0.99,
        initial_val=0.0,
    ):
        self.num_actions = num_actions
        self.explorer = explorer
        self.step_size_schedule = step_size_schedule
        assert multiplier >= 0.0, "Multiplier must be non-negative"
        self.multiplier = multiplier
        self.residuals = np.full((num_states, num_actions), initial_val, dtype=float)
        self.discount = discount

    @property
    def q(self):
        return (
            self.residuals
            + self.multiplier
            * np.sum(self.residuals, axis=1, keepdims=True)
            / self.num_actions
        )

    def update_q(self, obs, action, reward, terminated, next_obs):
        target = (
            reward if terminated else reward + self.discount * np.max(self.q[next_obs])
        )
        estimate = self.q[obs, action]
        step_size = self.step_size_schedule.step_size(obs, action)
        td_error = target - estimate
        multiplier_vec = self.multiplier * np.ones(self.num_actions) / self.num_actions
        multiplier_vec[action] += 1.0  # Adjust for the action taken
        try:
            self.residuals[obs] += (
                step_size * td_error * multiplier_vec
            )  # Update all actions' residuals
        except:
            set_trace()


def measure_value_error(q_vec, optimal_values, state_distribution):
    return np.linalg.norm(q_vec - optimal_values)


def plot_value_error_curve(value_errors, timestep_interval=50, agent="Q-learning"):
    import matplotlib.pyplot as plt

    plt.plot(value_errors)
    plt.xlabel(f"Timestep (x{timestep_interval})")
    plt.ylabel("Value Error (L2 Norm)")
    plt.title(f"Value Error over Time for {agent}")
    plt.show()


def timed_episodes_experiment(env, agent, max_timesteps=5000):
    observation, _ = env.reset()
    optimal_values = table_rl.dp.dp.value_iteration(
        env.num_states, 4, env.R, env.T, 0.95, 10000
    )

    accumulated_value_errors = []
    episode_completions = 0
    episode_completion_times = []
    data_check_interval = 50
    for timestep in range(max_timesteps):
        if timestep % data_check_interval == 0:
            # print(f"Timestep {timestep}")
            thing = np.amax(agent.q, axis=1)
            thing_as_list = thing.tolist()
            thing_as_list.append(0.0)
            q_vec = np.array(thing_as_list)
            value_error = measure_value_error(
                np.amax(agent.q, axis=1), optimal_values, None
            )
            # value_error = measure_value_error(q_vec, optimal_values, None)
            accumulated_value_errors.append(value_error)
            # print(f"Value error: {value_error}")
        action = agent.act(observation, True)
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated:
            episode_completions += 1
            episode_completion_times.append(timestep + 1)
        agent.observe(observation, reward, terminated, truncated, training_mode=True)
        if terminated or truncated:
            observation, info = env.reset()
    return episode_completion_times


def episode_counting_experiment(env, agent, time_to_episode, max_timesteps=800_000):
    observation, _ = env.reset()
    optimal_values = table_rl.dp.dp.value_iteration(
        env.num_states, 4, env.R, env.T, 0.99, 10000
    )

    accumulated_value_errors = []
    episode_completions = 0
    episode_completion_times = []
    data_check_interval = 50
    for timestep in range(max_timesteps):
        if timestep % data_check_interval == 0:
            # print(f"Timestep {timestep}")
            thing = np.amax(agent.q, axis=1)
            thing_as_list = thing.tolist()
            thing_as_list.append(0.0)
            q_vec = np.array(thing_as_list)
            value_error = measure_value_error(
                np.amax(agent.q, axis=1), optimal_values, None
            )
            # value_error = measure_value_error(q_vec, optimal_values, None)
            accumulated_value_errors.append(value_error)
            # print(f"Value error: {value_error}")
        action = agent.act(observation, True)
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated:
            episode_completions += 1
            episode_completion_times.append(timestep + 1)
        if episode_completions == time_to_episode:
            break
        agent.observe(observation, reward, terminated, truncated, training_mode=True)
        if terminated or truncated:
            observation, info = env.reset()
    if episode_completions == time_to_episode:
        return episode_completion_times[-1] - episode_completion_times[0]
    else:
        return None


def run_many_experiments(agent_type, num_experiments, time_to_episode, goal_reward):
    step_size = table_rl.step_size_schedulers.ConstantStepSize(0.05)
    explorer = table_rl.explorers.ConstantEpsilonGreedy(0.1, 4)
    discount_rate = 0.95
    finish_times = []
    for _ in range(num_experiments):
        env = table_rl.envs.ControlGridworldEnv(
            width=5,
            height=5,
            goal_reward=goal_reward,
            noise_prob=0.25,
            truncation_limit=None,
        )
        if agent_type == "q_learning":
            agent = table_rl.learners.QLearning(
                env.num_states,
                env.num_actions,
                step_size,
                explorer,
                discount=discount_rate,
                initial_val=0.0,
            )
        elif agent_type == "mean_residual_q_learning":
            agent = MeanResidualQLearning(
                env.num_states,
                env.num_actions,
                step_size,
                explorer,
                multiplier=1.0,
                discount=discount_rate,
                initial_val=0.0,
            )
        finish_time = episode_counting_experiment(
            env, agent, time_to_episode, max_timesteps=800_000
        )
        finish_times.append(finish_time)
    mean_finish_time = np.mean([time for time in finish_times if time is not None])
    std_finish_time = np.std(
        [time for time in finish_times if time is not None], ddof=1
    )
    print(
        f"Mean finish time over {num_experiments} experiments: {mean_finish_time} +/- {std_finish_time}"
    )


def run_many_timed_experiments(
    agent_type,
    num_experiments,
    max_timesteps,
    goal_reward,
    step_size,
    resolution,
    **kwargs,
):
    step_size_schedule = table_rl.step_size_schedulers.ConstantStepSize(step_size)
    explorer = table_rl.explorers.ConstantEpsilonGreedy(0.1, 4)
    discount_rate = 0.95
    episode_completions_list = []
    aligned_counts = []
    for _ in range(num_experiments):
        env = table_rl.envs.ControlGridworldEnv(
            width=5,
            height=5,
            goal_reward=goal_reward,
            noise_prob=0.25,
            truncation_limit=None,
        )
        if agent_type == "q_learning":
            agent = table_rl.learners.QLearning(
                env.num_states,
                env.num_actions,
                step_size_schedule,
                explorer,
                discount=discount_rate,
                initial_val=0.0,
            )
        elif agent_type in ["IBQ"]:
            multiplier = kwargs["k"]
            agent = MeanResidualQLearning(
                env.num_states,
                env.num_actions,
                step_size_schedule,
                explorer,
                multiplier=multiplier,
                discount=discount_rate,
                initial_val=0.0,
            )
        episode_completions = timed_episodes_experiment(
            env, agent, max_timesteps=max_timesteps
        )
        episode_completions_list.append(episode_completions)
    common_time_axis = np.arange(0, max_timesteps + 1, resolution)
    for episode_completions in episode_completions_list:
        counts = np.searchsorted(episode_completions, common_time_axis)
        aligned_counts.append(counts)
    mean_episodes = np.mean(aligned_counts, axis=0)
    ste_episodes = np.std(aligned_counts, ddof=1, axis=0) / np.sqrt(num_experiments)
    return mean_episodes, ste_episodes


def plot_k_results(k_best_results, log_intervals, plot_type):
    assert plot_type in ["percent_increase", "raw"]

    plt.figure(figsize=(8, 5))
    for log_interval in log_intervals:
        baseline = None
        for k, mean, ste in k_best_results[log_interval]:
            if k == 0:
                baseline = mean
                break
        assert baseline is not None, "Baseline (k=0) result not found."
        xs, ys, stes = zip(*k_best_results[log_interval])
        xs = np.array(xs)
        if plot_type == "percent_increase":
            ys = np.array([(mean - baseline) / baseline * 100 for mean in ys])
        elif plot_type == "episode_diff":
            ys = np.array([mean - baseline for mean in ys])
        elif plot_type == "raw":
            ys = np.array(ys)
        stes = np.array(stes)

        # 1. Add the label here so the legend picks up the solid line color
        line = plt.plot(xs, ys, label=f"{log_interval} Timesteps")[0]
        plt.fill_between(
            xs,
            ys - 1.96 * stes,
            ys + 1.96 * stes,
            color=line.get_color(),
            alpha=0.2,
        )
    plt.xlabel(r"$k$", fontsize=16)
    # 1. Grab the default handles (the lines) and their labels
    handles, labels = plt.gca().get_legend_handles_labels()
    # 2. Convert each line handle into a solid Patch block
    block_handles = [
        mpatches.Patch(color=h.get_color(), label=l) for h, l in zip(handles, labels)
    ]
    # 3. Pass the new blocks to the legend
    plt.legend(handles=block_handles, loc="best", fontsize=12)
    # plt.legend(loc='best', fontsize=12)
    if plot_type == "percent_increase":
        plt.ylabel("% Increase in Completed Episodes", fontsize=16)
    elif plot_type == "episode_diff":
        plt.ylabel("Increase in Episodes Completed", fontsize=16)
    elif plot_type == "raw":
        plt.ylabel("Episodes Completed", fontsize=16)
    # plt.title('Best Step Size Performance by k Value', fontsize=18)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.yticks(fontsize=12)
    ax = plt.gca()
    ax.set_xscale("symlog")
    ax.set_xlim(left=0)
    ax.set_xticks(xs)

    def fmt_k(x):
        if x == 0.0:
            return "0"
        if 0.0 < x < 1.0:
            return str(x).lstrip("0")
        return str(x)

    ax.set_xticklabels([fmt_k(x) for x in xs], fontsize=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    if not os.path.exists("tabular/figs"):
        os.makedirs("tabular/figs")
    figname = f"tabular/figs/{plot_type}_performance.png"
    plt.savefig(figname)
    print("Saved figure into {}".format(figname))


def experiment_worker(args):
    """
    Wrapper function to unpack arguments and run the experiment.
    Returns: (k, step_size, mean_episodes, ste_episodes) or None on error.
    """
    (
        k,
        step_size,
        num_experiments,
        max_timesteps,
        log_intervals,
        goal_reward,
        resolution,
        task_num,
        num_tasks,
    ) = args

    try:
        # print(f"Starting: k={k}, step_size={step_size}") # Optional: noisy with many threads
        mean, ste = run_many_timed_experiments(
            "IBQ",
            num_experiments,
            max_timesteps,
            goal_reward,
            step_size,
            resolution=resolution,
            k=k,
        )
        print(
            "Completed: k={}, step_size={}, task_num={}/{}".format(
                k, step_size, task_num, num_tasks
            )
        )
        return (k, step_size, mean, ste)
    except ZeroDivisionError:
        return None


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    num_experiments = 128
    max_timesteps = 5000
    log_intervals = [
        500,
        1000,
        2500,
        5000,
    ]  # Timesteps at which to log aggregate results
    resolution = 10  # frequency in timesteps of checking data
    goal_reward = 5.0

    step_sizes = generate_step_sizes()
    k_vals = np.array([0.0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8])
    # Prepare Tasks
    # Create a list of all (k, step_size) combinations to run
    tasks = []
    for k in k_vals:
        for step_size in step_sizes:
            tasks.append(
                (
                    k,
                    step_size,
                    num_experiments,
                    max_timesteps,
                    log_intervals,
                    goal_reward,
                    resolution,
                    len(tasks) + 1,
                    len(k_vals) * len(step_sizes),
                )
            )

    print(f"Generated {len(tasks)} experimental tasks.")
    print(f"Starting parallel execution using {multiprocessing.cpu_count()} cores...")

    # Run in Parallel
    num_cores = multiprocessing.cpu_count()

    with multiprocessing.Pool(processes=num_cores) as pool:
        # map returns results in the same order as tasks
        all_results = pool.map(experiment_worker, tasks)

    print("Experiments finished. Processing results...")

    # Process Results ---
    # We need to group results by k to find the best step_size for each k

    # Dictionary to hold all valid results: results_by_k[k] = list of (step_size, mean, ste)
    results_by_k_storage = {k: [] for k in k_vals}

    for res in all_results:
        if res is None:
            continue
        k, step_size, mean, ste = res
        for log_interval in log_intervals:
            num_points = int(log_interval / resolution)
            interval_mean = mean[: num_points + 1]
            interval_ste = ste[: num_points + 1]
            results_by_k_storage[k].append(
                (step_size, log_interval, interval_mean, interval_ste)
            )

    # Containers for final best results
    results = {}
    best_step_sizes = {}
    k_best_results = {}

    for log_interval in log_intervals:
        k_best_results[log_interval] = []
        results[log_interval] = {}
        best_step_sizes[log_interval] = {}
        for k in k_vals:
            k_data = results_by_k_storage.get(k, [])
            if not k_data:
                print(f"No valid results for k={k}")
                continue
            interval_data = [
                (step_size, mean, ste)
                for (step_size, li, mean, ste) in k_data
                if li == log_interval
            ]
            if not interval_data:
                print(f"No valid results for k={k} at log_interval={log_interval}")
                continue
            best_entry = max(
                interval_data, key=lambda x: x[1][-1]
            )  # Get the entry with the highest mean at the last point
            best_step_size, best_mean, best_ste = best_entry
            best_step_sizes[log_interval][k] = best_step_size
            print(
                f"Best step size for k={k} at log_interval={log_interval}: {best_step_size}"
            )
            results[log_interval][f"Implicit Baseline Q-learning (k={k})"] = (
                best_mean,
                best_ste,
            )
            k_best_results[log_interval].append((k, best_mean[-1], best_ste[-1]))

    # Plot
    plot_k_results(k_best_results, log_intervals, plot_type="percent_increase")
    plot_k_results(k_best_results, log_intervals, plot_type="raw")
