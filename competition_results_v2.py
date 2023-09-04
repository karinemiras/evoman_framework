
import sys
import matplotlib.pyplot as plt
from math import pi
from matplotlib.ticker import AutoLocator
from matplotlib.offsetbox import AnchoredText


from evoman.environment import Environment
from demo_controller import player_controller
import numpy as np
import pandas as pd
import pickle as pkl
import os
import pygame
os.environ["SDL_VIDEODRIVER"] = "dummy"

###### CREATE A FOLDER CALLED solutions IN THE SAME DIRECTORY AS THIS SCRIPT  AND PASTE ALL SOLUTION TXTs THERE ! #####

mode = "test"  # Can be test for generating competition files, or demo to just present the winners

######

experiment_name = 'test'
n_enemies = 8
n_hidden = 10

# Switch for demo
if mode == "demo":
    repetitions = 1
    speed = "normal"
    fullscreen = True
    sound = "on"
else:
    repetitions = 5
    speed = "fastest"
    fullscreen = False
    sound = "off"

if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Run each enemy n times for each group and record the data
df = pd.DataFrame(columns=["fitness", "player_life", "enemy_life", "time", "group", "repetition", "enemy"])
enemies = range(1, n_enemies + 1)
index = 0
for file in os.listdir("solutions"):
    if file.endswith(".txt"):
        group_name = file.replace(".txt", "")
        try:
            solution = np.loadtxt("solutions/" + file)
            print("File of group " + str(group_name) + " was read")
        except:
            print("File of group "+str(group_name)+" could NOT be read")

        for enemy in enemies:
            env = Environment(
                experiment_name=experiment_name,
                enemies=[enemy],
                playermode="ai",
                fullscreen=fullscreen,
                player_controller=player_controller(n_hidden),
                enemymode="static",
                level=2,
                sound=sound,
                speed=speed)

            n_vars = (env.get_num_sensors() + 1) * n_hidden + (n_hidden + 1) * 5  # multilayer with 50 neurons
            for n in range(repetitions):
                try:
                    f, p, e, t = env.play(pcont=solution)
                    df.loc[index,] = [f, p, e, t, group_name, n, enemy]
                    index += 1
                except:
                    print('bad solutioon')

if mode == "test":
    # Convert time to time left for sorting
    df["time"] = 3000 - df["time"]
    df["gain"] = df["player_life"] - df["enemy_life"]

    # Calculate gain and aggregate data
    df_final = pd.DataFrame(columns=["group", "enemies_slain", "gain", "player_life", "enemy_life", "time"])
    for i, group in enumerate(list(set(df["group"]))):
        this_group = df["group"] == group
        dead_enemies = np.count_nonzero(df["enemy_life"].loc[this_group] == 0) / repetitions
        gain = sum(df["player_life"].loc[this_group] - df["enemy_life"].loc[this_group]) / repetitions
        plife = sum(df["player_life"].loc[this_group]) / repetitions / n_enemies
        elife = sum(df["enemy_life"].loc[this_group]) / repetitions / n_enemies
        time = sum(df["time"].loc[this_group]) / repetitions / n_enemies
        df_final.loc[i] = {"group": group, "enemies_slain": dead_enemies, "gain": gain, "player_life": plife, "enemy_life": elife, "time": time}

    # Determine and print winners
    winners = pd.DataFrame(columns=["slain", "gain"])
    winners_slain = df_final.sort_values(by=["enemies_slain", "player_life", "time"], ascending=False).reset_index()
    winners_gain = df_final.sort_values(by="gain", ascending=False).reset_index()
    print("Winner for slain enemies: \n", winners["slain"].head(n=3))
    print("Winner for gain measure: \n", winners["gain"].head(n=3))
    # Index as ranks
    winners_slain["time"] = 3000 - winners_slain["time"]
    winners_gain["time"] = 3000 - winners_gain["time"]
    pd.concat([winners_slain, winners_gain], axis=1).to_csv("winners.csv")
    # Prepare data for radar chart and make plots of winners and whole class
    # adapted from: https://python-graph-gallery.com/391-radar-chart-with-several-individuals/
    for winner in [winners_slain["group"], winners_gain["group"], ["whole_class"]]:  # !!!! [:3]
        for group in winner:
            this_group = df["group"] == group
            if group == "whole_class":
                df_plot = df.drop(["group", "repetition"], axis=1).apply(pd.to_numeric).groupby(
                    "enemy").mean().transpose()
            else:
                df_plot = df.loc[this_group].drop(["group", "repetition"], axis=1).apply(pd.to_numeric).groupby(
                    "enemy").mean().transpose()
            df_plot = df_plot.reset_index().rename(columns={"index": "group"})
            # Build radar chart
            categories = list(df_plot)[1:]
            N = len(categories)
            # Determine angle
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]

            # Initialise radar chart
            ax = plt.subplot(111, polar=True)
            plt.title(group)

            # If you want the first axis to be on top:
            ax.set_theta_offset(pi / 2)
            ax.set_theta_direction(-1)

            # Draw one axe per variable + add labels labels yet

            g = str(round(df_plot.drop("group", axis=1).loc[4].sum(), 2))

            text_box = AnchoredText("Gain: " + g, frameon=False, loc=8, pad=-3.5)
            plt.setp(text_box.patch, facecolor='white', alpha=0.5)
            plt.gca().add_artist(text_box)

            plt.xticks(angles[:-1], categories)

            # Draw ylabels
            ax.set_rlabel_position(0)

            labels = ["gain"]
            indices = [4]

            for lab, col in zip(labels, indices):
                values = df_plot.loc[col].drop('group').values.flatten().tolist()
                values += values[:1]
                ax.plot(angles, values, linewidth=1, linestyle='solid', label="energy")
                ax.fill(angles, values, 'b', alpha=0.1)
                if group == "whole_class" and lab == "gain":
                    print(lab, values)
                    ax.yaxis.set_major_locator(AutoLocator())
                    if lab == "player life":
                        continue
                    # Next line is to prevent that there is no plain in the plot when almost all values are 0 and one or two
                    # are really high
                    plt.ylim(bottom=min(values) - 10)
                    plt.legend(loc='lower right', bbox_to_anchor=(0.1, 0.1))
                    plt.savefig(group + "_energy.png", dpi=300)
                    plt.close()

                    ax = plt.subplot(111, polar=True)
                    ax.set_theta_offset(pi / 2)
                    ax.set_theta_direction(-1)
                    plt.xticks(angles[:-1], categories)
                    ax.set_rlabel_position(0)

            if group != "whole_class":
                plt.legend(loc='lower right', bbox_to_anchor=(0.1, 0.1))
                plt.savefig(group + "_energy.png", dpi=300)
                plt.close()

    plt.close()
    plt.hist(pd.to_numeric(df_final["gain"]))
    plt.title("Distribution of gain\n(whole class)")
    plt.savefig("gain_hist_whole_group.png", dpi=300)

