# graph file for analysis

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde

import json

root_path = os.getcwd()

# read characteristics of landscapes
land_path = os.path.join(root_path, "landscape/Grid5_Lambda")
land_lambda = ["0.5", "1", "2"]

land_sc = {}
for land_l in land_lambda:
    path = os.path.join(land_path+land_l, "Grid5_Lambda"+land_l)
    land_sc[land_l] = {}
    for land_n in range(5):
        with open(path+"_"+str(land_n)+".json", "r") as f:
            data = json.load(f)
        land_ = data["Landscape"]
        land_sc[land_l][land_n] = [np.mean(land_), np.std(land_), np.max(land_)]
    # print("Landscape Lambda: ", land_l, ": ", land_sc[land_l])


# read agent_decision file
path = os.path.join(root_path, "output/agent_decision(original_value).csv")
df = pd.read_csv(path)
df["agent_lambda"] = np.round(df['agent_lambda'], 2)

# read group_mapping file
# use this file to check which landscape is used in which round
mappingpath = os.path.join(root_path, "output/group_mapping.csv")
mapdf = pd.read_csv(mappingpath)

# use group_mapping to map landscape characteristics to agent_decision
land_mean, land_std = [], []
land_max = []
for idx, row in df[['group_number', 'round']].iterrows():
    gp, rd = row['group_number'], row['round']
    land_s = mapdf['land_number'][(mapdf['group_number'] == gp) & (mapdf['round_number'] == rd)].values
    temp = land_s[0].split("a")[-1]
    land_l, land_n = temp.split('_')
    land_n = int(land_n)
    land_mean.append(land_sc[land_l][land_n][0])
    land_std.append(land_sc[land_l][land_n][1])
    land_max.append(land_sc[land_l][land_n][2])

# store landscape characterstics to a new agent_decision_z.csv
df["land_mean"] = np.array(land_mean)
df["land_std"] = np.array(land_std)
df['land_max'] = np.array(land_max)
df['agent_z'] = (df['agent_max'] - df['land_mean'])/df['land_std']

df.to_csv(os.path.join(root_path, "output/agent_decision_z.csv"))

# Distribution plot for different agent_lambdas
# Define the range of agent_alphas and agent_lambdas
agent_alphas = np.round(np.linspace(0.1, 1, 10), 2)
agent_lambdas = [0.3, 0.6]

for ag_lambda in agent_lambdas:
    # Create a figure with subplots (arranged as 5 rows and 2 columns)
    fig, axes = plt.subplots(5, 2, figsize=(12, 15))  # Adjust the number of rows/cols based on your preferences

    # Flatten axes for easier indexing in the loop
    axes = axes.flatten()

    # Loop over different values of 'a'
    for idx, alpha in enumerate(agent_alphas):
        # Filter data for each value of 'a' and the rounds
        for r in range(-3, 1):
            am = df["agent_z"][(df["agent_alpha"] == alpha) & (df["round"] == r) & (df["agent_lambda"] == ag_lambda)]
            
            # Plot the histogram on the corresponding subplot
            axes[idx].hist(am, bins=10, alpha=0.5, label=f'Round {r}', density=True)  # Use density=True for KDE
            
            # Calculate KDE and plot it
            if len(am) > 1:  # Only plot KDE if there's enough data points
                kde = gaussian_kde(am)
                x_vals = np.linspace(min(am), max(am), 100)  # Generate x values for KDE plot
                axes[idx].plot(x_vals, kde(x_vals), label=f'KDE Round {r}')

        # Set the title and labels for each subplot
        axes[idx].set_title(f"Histogram and KDE for a = {alpha}")
        axes[idx].set_xlabel("agent_z")
        axes[idx].set_ylabel("Density")
        axes[idx].legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the figure with the name "agent_z_{alpha}.png"
    plt.savefig(os.path.join(root_path, f"output/agent_z_Lambda{ag_lambda}.png"))

    # Display the plot
    plt.show()

