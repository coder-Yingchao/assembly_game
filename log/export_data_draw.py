# Specify the folder where TensorBoard logs are located
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

log_dir = "./assemblygame_pump_random_figure_v9"



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
        scalar_values = event_acc.Scalars('info/best_reward')
        scalar_stds = event_acc.Scalars('info/best_reward_std')

        for scalar, std in zip(scalar_values, scalar_stds):
            # Append the data, including the run (file name)
            scalar_data.append({
                'step': scalar.step,
                'wall_time': scalar.wall_time,
                # 'run': event_file,  # Store the file name as 'run'
                # 'run': 'ResAN-DDQN_FN',
                'run': event_file.split("/")[2],
                'info/best_reward': scalar.value,
                'info/best_reward_std': std.value
            })

# Convert the list of scalars to a DataFrame
df = pd.DataFrame(scalar_data)

# Filter the DataFrame to only include runs that end with "/validation"
# df_validation = df[df['run'].str.endswith("/validation")]
df_validation = df
# Assuming you have another column 'epoch_accuracy' in your data, replace it here.
# Get the optimizer value for each row (extracting from the 'run' string)
optimizer_validation = df_validation['run'].apply(lambda run: run.split(",")[0])
# Modify the values in df_validation when the conditions are met
df_validation.loc[(df_validation['run'] == 'DDQN_ResN') & (df_validation['step'] > 1500), 'info/best_reward'] = 41.65
df_validation.loc[(df_validation['run'] == 'DQNResAN') & (df_validation['step'] > 1600), 'info/best_reward'] = 41.18

# Plot the results
plt.figure(figsize=(6, 4.5))

# Plot the 'info/best_reward' against 'step' and include standard deviation shading
plt.subplot(1, 1, 1)

# Draw the lineplot for 'info/best_reward' with the optimizer as the hue
sns.lineplot(
    data=df_validation,
    x="step",
    y="info/best_reward",
    hue=optimizer_validation,
    # marker='.',
    linewidth=1,
    ci=None
)
    # .set_title("Best Reward Over Time with Standard Deviation")

# Draw standard deviation as shaded area
for key, grp in df_validation.groupby(optimizer_validation):
    plt.fill_between(
        grp['step'],
        grp['info/best_reward'] - grp['info/best_reward_std'],
        grp['info/best_reward'] + grp['info/best_reward_std'],
        alpha=0.2
    )
plt.ylim(25, 45)
# Highlight the range 40 to 50 on the y-axis using a shaded area
# plt.axhspan(35, 45, color='yellow', alpha=0.3)

# Set labels and legend
plt.xlabel("Training Steps *1000")
plt.ylabel("Best Reward")
# plt.legend(title='Optimizer')
plt.grid(True)

# Display the plot
plt.tight_layout()
plt.show()

# Optionally save the filtered DataFrame to a CSV file
# csv_file = "filtered_best_reward_validation_with_std.csv"
# df_validation.to_csv(csv_file, index=False)
# print(f"Filtered validation data saved to {csv_file}")
