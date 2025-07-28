import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
import numpy as np
from datetime import datetime

def convert_timestamp_to_time(timestamp):
    dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
    
    # Formatieren: "17. Juli 2025 18:30"
    return dt.strftime("%-d. %B %Y %H:%M")

class Auswerter():
    def __init__(self, timestamp):
        self.timestamp = timestamp
        try:
            heatmap_dir = os.path.join("RAW_Data", timestamp, "heatmap_data")
            self.heatmap_paths = []
            for filename in os.listdir(heatmap_dir):
                self.heatmap_paths.append(os.path.join(heatmap_dir, filename))

            training_data_dir = os.path.join("RAW_Data", timestamp, "training_data")
            self.training_data_paths = []
            for filename in os.listdir(training_data_dir):
                self.training_data_paths.append(os.path.join(training_data_dir, filename))
        except FileNotFoundError:
            print("No files found for the given timestamp.")


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


    def heatmap(self,width,height,save=False):
        try:
            for path in self.heatmap_paths: 
                df = pd.read_csv(path)
                positions_x = df["x"].to_numpy()
                positions_y = df["y"].to_numpy()

                heatmap = np.zeros((width, height), dtype=int)

                for i in range(len(positions_x)):
                    x = positions_x[i]
                    y = positions_y[i]
                    heatmap[x, y] += 1

                        
                sns.heatmap(heatmap.T, annot=True, cmap="hot", cbar=True)
                plt.title(f"Heatmap ({convert_timestamp_to_time(timestamp)})")
                plt.gca().invert_yaxis()
                if save:
                    plt.savefig(f"plots/heatmaps/heatmap_{self.timestamp}.png")               
                plt.show()

        
        except FileNotFoundError:
            print(  "No heatmap data files found for the given timestamp.")

if __name__ == "__main__":
    timestamp = "20250728_133214"
    #print(convert_timestamp_to_time(timestamp))
    auswertung = Auswerter(timestamp)
    auswertung.simpleplot(True)
    auswertung.heatmap(8,8,True)