# Adjusting the code to set fixed colors for the curves and markers
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Specify the folder where TensorBoard logs are located
log_dir = "./assemblygame_desktop_random_figure_v9"

# Initialize an empty list to store the scalar data
scalar_data = []

# Function to recursively search for event files in subdirectories
def find_event_files(root_dir):
    event_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if "tfevents" in file:  # Check for event files
                event_files.append(os.path.join(root, file))
    return event_files

# Get all event files from the root directory and its subfolders
event_files = find_event_files(log_dir)

# Iterate through each event file and extract data
for event_file in event_files:
    # Load the event file using EventAccumulator
    event_acc = EventAccumulator(event_file)
    event_acc.Reload()  # Load the event file
    # Get all scalar tags
    scalar_tags = event_acc.Tags().get('scalars', [])
    # Extract data for 'info/best_reward' and its corresponding 'info/best_reward_std'
    if 'info/best_reward' in scalar_tags and 'info/best_reward_std' in scalar_tags:
        scalar_mean = event_acc.Scalars('test/returns_stat/mean')
        scalar_max = event_acc.Scalars('test/returns_stat/max')
        scalar_min = event_acc.Scalars('test/returns_stat/min')

        for mean, max, min in zip(scalar_mean, scalar_max, scalar_min):
            # Append the data, including the run (file name)
            scalar_data.append({
                'step': mean.step,
                'wall_time': mean.wall_time,
                'run': event_file.split("/")[2],
                'mean': mean.value,
                'max': max.value,
                'min': min.value
            })

# Convert the list of scalars to a DataFrame
df = pd.DataFrame(scalar_data)

# Filter the DataFrame to only include runs that end with "/validation"
df_validation = df

# Get the optimizer value for each row (extracting from the 'run' string)
optimizer_validation = df_validation['run'].apply(lambda run: run.split(",")[0])

# Apply a rolling window to smooth the 'mean', 'min', and 'max' columns
window_size = 100  # You can adjust this window size
df_validation['mean_smooth'] = df_validation.groupby('run')['mean'].transform(
    lambda x: x.rolling(window=window_size, min_periods=1).mean())
df_validation['min_smooth'] = df_validation.groupby('run')['min'].transform(
    lambda x: x.rolling(window=window_size, min_periods=1).mean())
df_validation['max_smooth'] = df_validation.groupby('run')['max'].transform(
    lambda x: x.rolling(window=window_size, min_periods=1).mean())

# Set different markers for each optimizer
markers = ['o', 'v', 's', 'P', 'D', '*', 'X']

# Define fixed colors for the 7 lines
colors = ['blue', 'red','green', 'purple', 'orange', 'brown', 'pink']

# Create a plot
plt.figure(figsize=(6, 4.5))

# Plot the smoothed 'mean' against 'step' and include standard deviation shading
for (key, grp), marker, color in zip(df_validation.groupby(optimizer_validation), markers, colors):
    # Plot every 50000 steps for marking
    grp_subsampled = grp[grp['step'] % 50000 == 0]

    # Line plot with fixed color for the curve and marker
    line = sns.lineplot(
        data=grp,
        x="step",
        y="mean_smooth",
        marker=None,  # No markers for the full curve
        # label=key,
        linewidth=2,
        ci=None,
        color=color  # Use the fixed color from the array
    )

    # Plot markers for every 50000 steps with the same fixed color
    plt.scatter(
        grp_subsampled['step'],
        grp_subsampled['mean_smooth'],
        marker=marker,
        # label=f'{key}',
        color=color,  # Use the fixed color
        s=40
    )

    # Draw standard deviation as shaded area using the smoothed 'min' and 'max'
    plt.fill_between(
        grp['step'],
        grp['min_smooth'],
        grp['max_smooth'],
        alpha=0.1,
        color=color  # Use the same fixed color for the shaded area
    )

# Customize the plot
plt.ylim(0, 50)
plt.xlim(0, 800000)
plt.xlabel("Steps")
plt.ylabel("Reward")
plt.grid(True)
# Adjust legend placement to avoid overlap
# plt.legend(bbox_to_anchor=(1.05, 1), loc='lower right')
plt.tight_layout()

# Display the plot
plt.show()
