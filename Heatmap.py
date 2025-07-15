"""
Generate Heatmap
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns



width = 8
height = 8
seed = 0

df = pd.read_csv(f"RAW_data\heatmap_data\positions_MiniGrid_Empty_8x8_v0_noisy_True_seed_0_20250715_173825.csv")
positions_x = df["x"].to_numpy()
positions_y = df["y"].to_numpy()

heatmap = np.zeros((width, height), dtype=int)

for i in range(len(positions_x)):
    x = positions_x[i]
    y = positions_y[i]
    heatmap[x, y] += 1

        
sns.heatmap(heatmap.T, annot=True, cmap="hot", cbar=True)
plt.title("Besuchte Felder (Heatmap)")
plt.gca().invert_yaxis()
plt.show()