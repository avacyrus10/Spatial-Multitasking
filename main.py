import csv
import os
import random
import itertools
from typing import Dict, List

from plot import Plot
import matplotlib.pyplot as plt


class Task:
    def __init__(self, num_core, avrg_t, min_t, max_t, power, energy, energy_in_window, name):
        self.num_core = int(num_core)
        self.avrg_t = avrg_t
        self.min_t = min_t
        self.max_t = max_t
        self.power = power
        self.energy = energy
        self.energy_in_window = energy_in_window
        self.name = name
        self.color = self.generate_random_color()

    def generate_random_color(self):
        return "#{:06x}".format(random.randint(0, 0xFFFFFF))


def try_atof(val):
    try:
        return float(val.replace(',', ''))
    except ValueError:
        return None


def initialize(directory_path):
    tasks = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r') as file:
                reader = csv.reader(file, delimiter=',')
                next(reader)

                for row in reader:
                    row = [try_atof(entry) for entry in row]
                    if None in row:
                        print(f"Skipping row {row} in file {filename} due to conversion error.")
                        continue

                    num_core, avrg_t, min_t, max_t, power, energy, energy_in_window = row
                    task = Task(num_core, avrg_t, min_t, max_t, power, energy, energy_in_window, filename.split('.')[0])
                    while len(tasks) <= int(task.num_core):
                        tasks.append([])
                    tasks[int(task.num_core)].append(task)

    return tasks


def cooperative(tasks_list):
    """
       Assigns tasks to cores using a cooperative scheduling approach where all tasks are assigned
       to all cores. This is a simple, non-optimized approach for balancing the workload across cores.

       :param tasks_list: List of tasks per core from the initialization phase.
       :return: A list of tasks assigned to each core, and the total energy consumption of the tasks.
       """
    tasks = tasks_list[6]
    cores = [tasks] * 6
    total_energy = sum(task.energy for task in tasks)
    return cores, total_energy


def calculate_schedule_metrics(assignment, task_list):
    """
    Calculates key scheduling metrics including makespan, total power, total energy, and peak power
    based on the task assignments.

    :param assignment: Dictionary with core IDs as keys and a list of assigned tasks as values.
    :param task_list: List of all tasks used for the scheduling.
    :return: Tuple containing the makespan, total power, total energy, and peak power.
    """
    num_cores = 6
    core_times = [0] * num_cores
    total_power = 0
    total_energy = 0
    peak_power = 0

    for core_id, tasks in assignment.items():
        for task in tasks:
            core_times[core_id - 1] += task.max_t
            total_power += task.power
            total_energy += task.energy
            peak_power = max(peak_power, task.power)

    makespan = max(core_times)
    return makespan, total_power, total_energy, peak_power


def beam_search(task_list, beam_width=15_000_000):
    """
    Performs beam search to assign tasks to cores in an optimal manner, minimizing makespan and power consumption.

    :param task_list: List of tasks for each core from the initialization phase.
    :param beam_width: Integer specifying the width of the beam search.
                       It controls how many solutions are retained at each step.
    :return: The best task assignment found during the search.
    """
    num_cores = 6
    initial_assignment = {core_id: [] for core_id in range(1, num_cores + 1)}
    assigned_tasks_set = set()
    beam = [initial_assignment]
    all_tasks = list(itertools.chain.from_iterable(task_list[1:]))
    sorted_tasks = sorted(all_tasks, key=lambda task: task.max_t, reverse=True)

    for task in sorted_tasks:
        new_beam = []
        for assignment in beam:
            for core_id in range(1, num_cores + 1):
                assigned_tasks = assignment[core_id]
                if task not in assigned_tasks and task not in assigned_tasks_set:
                    new_assignment = {key: value.copy() for key, value in assignment.items()}
                    new_assignment[core_id].append(task)
                    new_beam.append(new_assignment)

        new_beam = sorted(new_beam, key=lambda assgn: calculate_schedule_metrics(assgn, task_list))
        beam = new_beam[:beam_width]
        assigned_tasks_set.add(task)

    return min(beam, key=lambda assgn: calculate_schedule_metrics(assgn, task_list))


def best(task_list):
    """
    Implements the 'best' scheduling strategy that optimizes task assignment by using beam search
    and returning the solution with the minimal makespan and energy consumption.

    :param task_list: List of tasks for each core from the initialization phase.
    :return: Tuple containing the makespan and total energy for the best task assignment.
    """
    assignment = beam_search(task_list)
    makespan, total_power, total_energy, peak_power = calculate_schedule_metrics(assignment, task_list)
    cores = [set(task.name for task in tasks) for tasks in assignment.values()]

    plot = Plot()
    plot.draw_assignment(cores, task_list, makespan, output_file='profile.png')
    return makespan, total_energy


def calculate_aggregated_speedup(assignment, remaining_tasks):
    """
    Calculates the aggregated speedup based on the current task assignment and remaining tasks.

    :param assignment: Dictionary with core IDs as keys and lists of assigned tasks as values.
    :param remaining_tasks: List of tasks that have not been assigned yet.
    :return: The aggregated speedup as a float.
    """
    total_speedup = 0
    for core_id, tasks in assignment.items():
        for task in tasks:
            speedup = task.max_t / (len(tasks) + 1e-9)
            total_speedup += speedup ** (1 / len(remaining_tasks) if remaining_tasks else 1)
    return total_speedup


def profile(tasks, assigned_cores, time):
    """
    Implements the 'profile' scheduling that maximizes aggregated speedup by considering
    the current time and task assignments.

    :param tasks: List of tasks to be scheduled.
    :param assigned_cores: Current task assignment for each core.
    :param time: The current time in the scheduling process.
    :return: The best task assignment found based on the profile algorithm.
    """
    remaining_tasks = [task for core_tasks in assigned_cores.values() for task in core_tasks if
                       task not in assigned_cores]
    available_cores = [i for i in range(6) if i not in assigned_cores]
    best_assignment = {i: [] for i in range(6)}
    best_speedup_sum = float('inf')

    for core_id in available_cores:
        for task in remaining_tasks:
            effective_time = time + task.max_t

            new_assignment = {key: value.copy() for key, value in assigned_cores.items()}
            new_assignment[core_id].append(task)

            speedup_sum = calculate_aggregated_speedup(new_assignment, remaining_tasks)

            if speedup_sum < best_speedup_sum:
                best_speedup_sum = speedup_sum
                best_assignment = new_assignment

    return best_assignment


def simulate_energy_for_tasks_range(start_tasks, end_tasks, step, algorithm_functions, tasks_list):
    num_tasks_range = range(start_tasks, end_tasks + 1, step)
    energy_values = {alg.__name__: [] for alg in algorithm_functions}

    for num_tasks in num_tasks_range:
        simulated_tasks = [
            Task(num_core=random.randint(1, 6), avrg_t=1, min_t=1, max_t=10, power=1, energy=1, energy_in_window=1,
                 name=f"Task{i}")
            for i in range(num_tasks)]

        for algorithm_function in algorithm_functions:
            assignment = algorithm_function(tasks_list, simulated_tasks)
            makespan, _, total_energy = calculate_schedule_metrics(assignment, tasks_list)

            energy_values[algorithm_function.__name__].append(total_energy)

    return num_tasks_range, energy_values


def plot_energy_vs_tasks(algorithm_functions, algorithm_names, tasks_list, start_tasks, end_tasks, step):
    num_tasks_range, energy_values = simulate_energy_for_tasks_range(start_tasks, end_tasks, step, algorithm_functions,
                                                                     tasks_list)

    for algorithm_name in algorithm_names:
        plt.plot(num_tasks_range, energy_values[algorithm_name], label=algorithm_name)

    plt.xlabel('Number of Tasks')
    plt.ylabel('Total Energy Consumption')
    plt.title('Energy Consumption vs Number of Tasks')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    time = 0
    directory_path = 'gpu_t400'
    tasks_list = initialize(directory_path)
    cores, baseline_total_energy = cooperative(tasks_list)
    p = Plot()
    baseline_makespan = p.draw_schedule_avrg(cores)
    assigned_cores = {i: [] for i in range(6)}

    best_makespan, best_total_energy = best(tasks_list)
    profile_assignment = profile(tasks_list, assigned_cores, time)

    pr_makespan, pr_total_power, pr_total_energy, _ = calculate_schedule_metrics(profile_assignment, tasks_list)

    cores_for_plot = [set(task.name for task in tasks) for _, tasks in profile_assignment.items()]

    plot = Plot()
    plot.draw_assignment(cores_for_plot, tasks_list, pr_makespan, output_file='profile--.png')
