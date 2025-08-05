import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
import numpy as np
from datetime import datetime
from rliable import metrics
from rliable.library import get_interval_estimates
from rliable.plot_utils import plot_sample_efficiency_curve


def convert_timestamp_to_time(timestamp):
    dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
    
    # Formatieren: "17. Juli 2025 18:30"
    return dt.strftime("%d. %B %Y %H:%M")

class Auswerter():
    def __init__(self, timestamps):
        self.timestamps = list(timestamps.values())
        self.algorithms = list(timestamps.keys())
        
        try:
            heatmap_dir = os.path.join("RAW_Data", self.timestamps[0], "heatmap_data")
            self.heatmap_paths = []
            for filename in os.listdir(heatmap_dir):
                self.heatmap_paths.append(os.path.join(heatmap_dir, filename))

            training_data_dir = os.path.join("RAW_Data", self.timestamps[0], "training_data")
            self.training_data_paths = []
            for filename in os.listdir(training_data_dir):
                self.training_data_paths.append(os.path.join(training_data_dir, filename))
        except FileNotFoundError:
            print("No files found for the given first timestamp.")

        if self.timestamps[1] is not None:
            try:
                training_data_dir_2 = os.path.join("RAW_Data", self.timestamps[1], "training_data")
                self.training_data_paths_2 = []
                for filename in os.listdir(training_data_dir_2):
                    self.training_data_paths_2.append(os.path.join(training_data_dir_2, filename))
            except FileNotFoundError:
                print("No files found for the given second timestamp.")
                self.training_data_paths_2 = None
        else:
            self.training_data_paths_2 = None
                
        if self.timestamps[2] is not None:
            try:
                training_data_dir_3 = os.path.join("RAW_Data", self.timestamps[2], "training_data")
                self.training_data_paths_3 = []
                for filename in os.listdir(training_data_dir_3):
                    self.training_data_paths_3.append(os.path.join(training_data_dir_3, filename))
            except FileNotFoundError:
                print("No files found for the given third timestamp.")
                self.training_data_paths_3 = None
        else:
            self.training_data_paths_3 = None


    def simpleplot(self, save=False):
        try:
            for path in self.training_data_paths:
                df = pd.read_csv(path)
                plt.plot(df["steps"], df["rewards"], label="Reward")
                plt.xlabel("steps")
                plt.ylabel("Reward")
                plt.title(f"Reward per Episode ({convert_timestamp_to_time(timestamp)})")
                plt.legend()
                plt.grid()
            if save:
                plt.savefig(f"plots/training/reward_plot_{self.timestamp}.png")
            plt.show()

        except FileNotFoundError:
            print("No training data files found for the given timestamp.")

    def iqmplot(self, save=False):
        # Same as simpleplot, but using rliable to calculate IQM
        # Some parts are copied from rliable_example.py given in the exercises
        try:
            i = 0
            dfs_1 = []
            for path in self.training_data_paths:
                _df = pd.read_csv(path)
                _df["seed"] = i
                dfs_1.append(_df)
                i += 1

            i = 0
            dfs_2 = []
            if self.training_data_paths_2 is not None:
                for path in self.training_data_paths_2:
                    _df = pd.read_csv(path)
                    _df["seed"] = i
                    dfs_2.append(_df)
                    i += 1

            i = 0
            dfs_3 = []
            if self.training_data_paths_3 is not None:
                for path in self.training_data_paths_3:
                    _df = pd.read_csv(path)
                    _df["seed"] = i
                    dfs_3.append(_df)
                    i += 1

            df_1 = pd.concat(dfs_1, ignore_index=True)
            df_1 = df_1.sort_values(["seed", "steps"])

            n_seeds = df_1["seed"].nunique()
            episodes_per_seed = df_1.groupby("seed").size().min()
            steps = df_1[df_1["seed"] == 0]["steps"].values[:episodes_per_seed]

            rewards_1 = np.vstack([
                g["rewards"].values[:episodes_per_seed]
                for _, g in df_1.groupby("seed")
            ])

            train_scores = {
                self.algorithms[0]: rewards_1,
            }

            if dfs_2:
                df_2 = pd.concat(dfs_2, ignore_index=True)
                df_2 = df_2.sort_values(["seed", "steps"])
                rewards_2 = np.vstack([
                    g["rewards"].values[:episodes_per_seed]
                    for _, g in df_2.groupby("seed")
                ])
                train_scores[self.algorithms[1]] = rewards_2

            if dfs_3:
                df_3 = pd.concat(dfs_3, ignore_index=True)
                df_3 = df_3.sort_values(["seed", "steps"])
                rewards_3 = np.vstack([
                    g["rewards"].values[:episodes_per_seed]
                    for _, g in df_3.groupby("seed")
                ])
                train_scores[self.algorithms[2]] = rewards_3

            iqm = lambda scores: np.array(  # noqa: E731
                [metrics.aggregate_iqm(scores[:, eval_idx]) for eval_idx in range(scores.shape[-1])]
            )

            iqm_scores, iqm_cis = get_interval_estimates(
                train_scores,
                iqm,
                reps=2000,
            )

            plot_sample_efficiency_curve(
                steps + 1,
                iqm_scores,
                iqm_cis,
                algorithms=list(train_scores.keys()),
                xlabel=r"Number of Frames",
                ylabel="IQM Normalized Score",
            )

            plt.gcf().canvas.manager.set_window_title(
                "IQM Normalized Score - Sample Efficiency Curve"
            )
            plt.legend()
            plt.tight_layout()
            plt.show()


            if save:
                plt.savefig(f"plots/training/iqm_reward_plot_{self.timestamp}.png")

        except FileNotFoundError:
            print("No training data files found for the given timestamp.")
        
  
    def heatmap_multipleseeds(self,width,height,start = 0,end = 5000,save=False):
        try:
            heatmap = np.zeros((width-2, height-2), dtype=int)
            for path in self.heatmap_paths: 
                df = pd.read_csv(path)
                positions_x = df["x"].to_numpy()
                positions_y = df["y"].to_numpy()
                positions_x = positions_x[start:end]
                positions_y = positions_y[start:end]

                positions_x -= 1
                positions_y -= 1
            
                for i in range(len(positions_x)):
                    x = positions_x[i]
                    y = positions_y[i]
                    heatmap[x, y] += 1

            plt.figure(figsize=(9,7.5))             
            sns.heatmap(heatmap.T, annot=True, fmt="d",cmap="hot", cbar=True, vmax = 25e2, annot_kws={"size": 18})
            plt.title(f"NoisyNet, steps: 0-500, 30 runs", fontsize=24)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            cbar = plt.gca().collections[0].colorbar
            cbar.ax.tick_params(labelsize=18)  # Skalenwerte
            if save:
                plt.savefig(f"plots/heatmaps/heatmap_multiple_seeds{self.timestamp}.png")               
            plt.show()
        
        except FileNotFoundError:
            print("No heatmap data files found for the given timestamp.")

    def heatmap(self,width,height,start = 0,end = 5000,save=False):
        try:
            for path in self.heatmap_paths: 
                df = pd.read_csv(path)
                positions_x = df["x"].to_numpy()
                positions_y = df["y"].to_numpy()
                positions_x = positions_x[start:end]
                positions_y = positions_y[start:end]

                positions_x -= 1
                positions_y -= 1
            
                heatmap = np.zeros((width-2, height-2), dtype=int)

                for i in range(len(positions_x)):
                    x = positions_x[i]
                    y = positions_y[i]
                    heatmap[x, y] += 1

                        
            plt.figure(figsize=(9,7.5))             
            sns.heatmap(heatmap.T, annot=True, fmt="d",cmap="hot", cbar=True, vmax = 25e2, annot_kws={"size": 18})
            plt.title(f"NoisyNet, steps: 0-500", fontsize=24)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            cbar = plt.gca().collections[0].colorbar
            cbar.ax.tick_params(labelsize=18)  # Skalenwerte
            if save:
                plt.savefig(f"plots/heatmaps/heatmap_{self.timestamp}.png")               
            plt.show()
        
        except FileNotFoundError:
            print("No heatmap data files found for the given timestamp.")

if __name__ == "__main__":
    timestamp = "20250802_112343" #epsilon greedy, when using iqm
    second_timestamp = "20250802_104744" #noisy without noise reduction, set so None if not used   
    third_timestamp = "20250803_123825" #noisy with noise reduction k=4, set so None if not used
    print(convert_timestamp_to_time(timestamp))

    algorithm_timestamps = {"DQN-epsilon_greedy": timestamp,
                           "DQN-NoisyNet": second_timestamp,
                           "DQN-NoisyNet-noise_reduction": third_timestamp}

    auswertung = Auswerter(algorithm_timestamps) # Second timestamp has to be noisy net data
    auswertung.heatmap_multipleseeds(5,5,0,1000)
    auswertung.heatmap(5,5,0,1000)
    auswertung.simpleplot()

    auswertung.iqmplot()

