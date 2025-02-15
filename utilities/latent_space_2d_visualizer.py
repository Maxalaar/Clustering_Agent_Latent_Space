import tkinter as tk
from pathlib import Path
import torch
import cupy as cp
from cuml import UMAP, KMeans
import numpy as np
from PIL import Image, ImageTk

from lightning_repertory.surrogate_policy import SurrogatePolicy
from utilities.latent_space_analysis.get_data import get_data
from utilities.latent_space_analysis.projection_clusterization_latent_space import \
    projection_clusterization_latent_space

class LatentSpace2DVisualizer:
    def __init__(
            self,
            surrogate_policy_path,
            dataset_path: Path,
            width=800,
            height=600,
            device=torch.device('cpu'),
            number_projection_points=30000, #30_000,  # New argument for number of points to calculate projection
            number_rendering_points=1000,     # New argument for number of points to display on rendering
    ):
        assert number_projection_points >= number_rendering_points, "number_projection_points must be >= number_rendering_points"

        self.device = device
        self.width = width
        self.height = height
        self.number_projection_points = number_projection_points
        self.number_rendering_points = number_rendering_points

        # Load surrogate policy
        self.surrogate_policy = self.load_surrogate_policy(surrogate_policy_path).to(device)

        # Initialize GPU-accelerated projector
        self.projector_2d = UMAP(
            n_components=2,
            n_epochs=100_000,
            # min_dist=0.0,
            n_neighbors=30,
        )

        # Setup Tkinter window
        self.setup_window()

        # Load data and compute embeddings
        observations = self.load_data(dataset_path)
        embeddings = self.compute_embeddings(observations)

        # GPU-accelerated clustering
        cluster_labels = self.clusterize_embeddings(embeddings)

        # GPU-accelerated projection
        projection_2d = self.compute_projection(embeddings)

        # Precompute visualization data
        self.projection_2d = projection_2d[:self.number_rendering_points]
        self.colors_rgb = self.precompute_colors(cluster_labels[:self.number_rendering_points])
        self.offsets = self.generate_circle_offsets(radius=3)

        # Initial render
        self.draw_points()

    def setup_window(self):
        """Initialize Tkinter window with canvas and image container."""
        self.window = tk.Toplevel()
        self.window.title("Latent Space 2D Visualizer")
        self.canvas = tk.Canvas(self.window, width=self.width, height=self.height, bg='white')
        self.canvas.pack()
        self.canvas_image = self.canvas.create_image(0, 0, anchor=tk.NW, image=None)
        self.photo_image = None

    def generate_circle_offsets(self, radius):
        """Generate circle offsets for point visualization."""
        offsets = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx**2 + dy**2 <= radius**2:
                    offsets.append((dx, dy))
        return np.array(offsets)

    def precompute_colors(self, cluster_labels):
        """Map cluster labels to RGB colors."""
        colors = np.array([
            [255, 0, 0],    # Red
            [0, 0, 255],    # Blue
            [0, 255, 0],    # Green
            [255, 165, 0],  # Orange
            [128, 0, 128],  # Purple
            [165, 42, 42],  # Brown
            [255, 192, 203],# Pink
            [128, 128, 128],# Gray
            [128, 128, 0],  # Olive
            [0, 255, 255]   # Cyan
        ], dtype=np.uint8)
        return colors[cluster_labels % len(colors)]

    def load_surrogate_policy(self, surrogate_policy_path):
        """Load surrogate policy with error handling."""
        try:
            return SurrogatePolicy.load_from_checkpoint(surrogate_policy_path)
        except Exception as e:
            raise RuntimeError(f"Error loading surrogate policy: {e}")

    def load_data(self, dataset_path):
        """Load observations data."""
        try:
            observations, = get_data(
                dataset_names=['observations'],
                data_number=self.number_projection_points,  # Use n_projection_points here
                trajectory_dataset_file_path=dataset_path / 'trajectory_dataset.h5',
                device=self.device,
            )
            return observations
        except Exception as e:
            raise RuntimeError(f"Error loading data: {e}")

    def compute_embeddings(self, observations):
        """Compute embeddings on GPU."""
        return projection_clusterization_latent_space(
            observations=observations,
            surrogate_policy=self.surrogate_policy,
        ).to(self.device)

    def clusterize_embeddings(self, embeddings):
        """GPU-accelerated KMeans clustering."""
        kmeans = KMeans(n_clusters=self.surrogate_policy.clusterization_function.number_cluster)
        return kmeans.fit_predict(cp.asarray(embeddings)).get()

    def compute_projection(self, embeddings):
        """GPU-accelerated UMAP projection."""
        return self.projector_2d.fit_transform(cp.asarray(embeddings)).get()

    def draw_points(self):
        """Vectorized point drawing using Numpy and PIL."""
        # Normalize coordinates
        xs, ys = self.projection_2d.T
        min_x, max_x = xs.min(), xs.max()
        min_y, max_y = ys.min(), ys.max()

        x_norm = (xs - min_x) / (max_x - min_x + 1e-8)
        y_norm = (ys - min_y) / (max_y - min_y + 1e-8)

        # Convert to canvas coordinates
        x_canvas = (x_norm * (self.width - 1)).astype(int)
        y_canvas = (y_norm * (self.height - 1)).astype(int)

        # Generate all point coordinates with offsets
        x_coords = x_canvas[:, None] + self.offsets[:, 0]
        y_coords = y_canvas[:, None] + self.offsets[:, 1]

        # Flatten and clip coordinates
        x_flat = np.clip(x_coords.ravel(), 0, self.width - 1)
        y_flat = np.clip(y_coords.ravel(), 0, self.height - 1)

        # Create image array
        image = np.full((self.height, self.width, 3), 255, dtype=np.uint8)
        image[y_flat, x_flat] = np.repeat(self.colors_rgb, len(self.offsets), axis=0)

        # Update canvas
        pil_image = Image.fromarray(image)
        self.photo_image = ImageTk.PhotoImage(pil_image)
        self.canvas.itemconfig(self.canvas_image, image=self.photo_image)

    def update(self):
        """Optimized update method."""
        self.draw_points()
        self.window.update_idletasks()
        self.window.update()

    def start(self):
        """Start main loop."""
        self.window.mainloop()
