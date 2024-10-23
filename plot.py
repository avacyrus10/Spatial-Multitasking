import matplotlib.pyplot as plt
from itertools import cycle
import itertools


class Plot:
    def __init__(self):
        self.t_current = 0

    def draw_schedule_avrg(self, cores, output_file='baseline.png'):
        fig, ax = plt.subplots(figsize=(10, 6))
        legend_items = set()

        for i, core_tasks in enumerate(cores, start=1):
            self.t_current = 0
            for task_instance in core_tasks:
                if task_instance.name not in legend_items:
                    legend_items.add(task_instance.name)
                    ax.broken_barh([(self.t_current, 0)], (0, 0), facecolors=(task_instance.color),
                                   edgecolor='none', linewidth=0, label=task_instance.name)

                ax.broken_barh([(self.t_current, task_instance.max_t)],
                               (i - 0.4, 0.8),
                               facecolors=(task_instance.color),
                               edgecolor='black', linewidth=1)

                self.t_current += task_instance.max_t

        total_time = self.t_current

        ax.set_yticks(range(1, len(cores) + 1))
        ax.set_yticklabels([f'Core {i}' for i in range(1, len(cores) + 1)])
        ax.set_xlabel('Time (ms)')
        ax.set_title('Task Schedule')

        ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1))

        ax.axvline(total_time, color='red', linestyle='--', label='All tasks finished')

        ax.text(total_time + 10, len(cores) + 1, f'Total Time: {total_time} ms', color='red')

        plt.savefig(output_file, bbox_inches='tight')
        return total_time

    def draw_assignment(self, cores, task_list, makespan, output_file='task_assignment.png'):
        fig, ax = plt.subplots(figsize=(10, 6))
        legend_items = set()
        for i, core_tasks in enumerate(cores, start=1):
            self.t_current = 0
            for task_name in core_tasks:

                matching_task = next(
                    (task for task in itertools.chain.from_iterable(task_list) if task.name == task_name), None)

                if matching_task:
                    color = matching_task.color

                    if task_name not in legend_items:
                        legend_items.add(task_name)

                        ax.broken_barh([(self.t_current, 0)], (0, 0), facecolors=color,
                                       edgecolor='none', linewidth=0, label=task_name)

                    max_t = matching_task.max_t
                    ax.broken_barh([(self.t_current, max_t)],
                                   (i - 0.4, 0.8),
                                   facecolors=color,
                                   edgecolor='black', linewidth=1)

                    self.t_current += max_t

        ax.set_yticks(range(1, len(cores) + 1))
        ax.set_yticklabels([f'Core {i}' for i in range(1, len(cores) + 1)])
        ax.set_xlabel('Time (ms)')
        ax.set_title('Task Assignment')

        ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1))

        plt.savefig(output_file, bbox_inches='tight')
