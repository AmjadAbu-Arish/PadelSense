import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from scipy.ndimage import gaussian_filter

class HeatmapGenerator:
    def __init__(self, mini_court_w=200, mini_court_h=400):
        self.w = mini_court_w
        self.h = mini_court_h
        # Initialize a 2D histogram
        self.heatmap_data = np.zeros((self.h, self.w), dtype=np.float32)

    def generate(self, mapped_positions, output_path="outputs/heatmap.png", sigma=5):
        """
        Generates a heatmap image based on the mapped positions and saves it.
        """
        if not mapped_positions:
            print("No positions provided to generate heatmap.")
            return

        for pos in mapped_positions:
            if pos is not None:
                x, y = int(pos[0]), int(pos[1])
                # Ensure coordinates are within bounds
                if 0 <= x < self.w and 0 <= y < self.h:
                    self.heatmap_data[y, x] += 1.0

        # Apply Gaussian filter for smoothing
        smoothed_heatmap = gaussian_filter(self.heatmap_data, sigma=sigma)

        # Normalize between 0 and 1
        max_val = smoothed_heatmap.max()
        if max_val > 0:
            smoothed_heatmap = smoothed_heatmap / max_val

        # Create visual heatmap using matplotlib
        plt.figure(figsize=(4, 8))

        # We can overlay it on a dark background resembling the mini-court
        ax = sns.heatmap(smoothed_heatmap, cmap="inferno", cbar=True, xticklabels=False, yticklabels=False)
        ax.invert_yaxis() # Match the image coordinate system if necessary, but usually index 0 is top. Let's leave as is for now.

        plt.title('Ball Position Heatmap')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Heatmap saved to {output_path}")
