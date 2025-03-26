import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")

def draw_gantt_tasks_ordered(assignments, schedule):
    """
    Draws a Gantt chart with each row representing a task in the order:
    F1, G1, F2, G2, …, Fn, Gn.
    
    The color of each bar indicates the worker (e.g., Zu, Ga) who performed the task.
    
    Parameters:
      - assignments: list of dictionaries for each sword mapping task ('F' or 'G') to a worker.
      - schedule: list of dictionaries for each sword mapping task ('F' or 'G') to [start, end] times.
    """
    # Define colors for each worker.
    worker_colors = {'Zu': 'tab:blue', 'Ga': 'tab:orange'}
    
    # Build a list of tasks in the desired order.
    tasks_list = []
    for sword_idx, (assign, times) in enumerate(zip(assignments, schedule)):
        # Process tasks in fixed order: F then G.
        for task in ['F', 'G']:
            worker = assign[task]
            start, end = times[task]
            duration = end - start
            label = f"{task}{sword_idx + 1}"
            tasks_list.append({
                'label': label,
                'worker': worker,
                'start': start,
                'duration': duration
            })
    
    # Create the plot.
    fig, ax = plt.subplots(figsize=(10, 6))
    row_height = 8
    row_gap = 10  # vertical spacing between rows
    
    # Plot each task.
    for row_idx, task_item in enumerate(tasks_list):
        y_pos = row_idx * row_gap
        ax.broken_barh(
            [(task_item['start'], task_item['duration'])],
            (y_pos, row_height),
            facecolors=worker_colors.get(task_item['worker'], 'gray'),
            edgecolor='black', alpha=0.8
        )
        # Add the task label inside the bar.
        ax.text(
            task_item['start'] + task_item['duration'] / 2, 
            y_pos + row_height / 2,
            task_item['label'], ha='center', va='center', color='white', fontsize=9
        )
    
    # Set y-ticks and labels.
    ax.set_yticks([i * row_gap + row_height / 2 for i in range(len(tasks_list))])
    ax.set_yticklabels([task['label'] for task in tasks_list])
    ax.set_xlabel("Time")
    ax.set_title("Gantt Chart (Ordered as F1, G1, F2, G2, …)")
    
    # Create a legend mapping colors to workers.
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=worker_colors[w]) for w in worker_colors]
    ax.legend(legend_handles, worker_colors.keys(), title="Worker", loc='upper right')
    
    plt.tight_layout()
    plt.show()