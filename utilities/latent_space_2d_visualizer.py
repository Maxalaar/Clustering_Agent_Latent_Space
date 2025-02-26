import tkinter as tk
from pathlib import Path
import torch
import cupy as cp
from cuml import UMAP, KMeans, PCA, TruncatedSVD, GaussianRandomProjection, SparseRandomProjection
import numpy as np
from PIL import Image, ImageTk

from lightning_repertory.surrogate_policy import SurrogatePolicy
from utilities.latent_space_analysis.get_data import get_data
from utilities.latent_space_analysis.projection_clusterization_latent_space import projection_clusterization_latent_space


class LatentSpace2DVisualizer:
    def __init__(
        self,
        surrogate_policy_path,
        dataset_path: Path,
        width=800,
        height=600,
        device=torch.device('cpu'),
        number_projection_points=10_000,  # par exemple 5000
        number_rendering_points=1_000,
        number_point_trajectory=5,
    ):
        assert number_projection_points >= number_rendering_points, "number_projection_points must be >= number_rendering_points"

        self.device = device
        self.width = width
        self.height = height
        self.number_projection_points = number_projection_points
        self.number_rendering_points = number_rendering_points
        self.number_point_trajectory = number_point_trajectory
        self.trajectory = []

        # Charger la politique substitutive et configurer la fenêtre Tkinter
        self.surrogate_policy = self.load_surrogate_policy(surrogate_policy_path).to(device)
        self.setup_window()

        # Charger les données et calculer les embeddings
        observations = self.load_data(dataset_path)
        embeddings = self.compute_embeddings(observations)

        # Clustering accéléré par GPU
        cluster_labels = self.clusterize_embeddings(embeddings)
        # Projection accélérée par GPU
        projection_2d = self.compute_projection(embeddings)

        # Sélectionner uniquement une partie des points à rendre
        self.projection_2d = projection_2d[:self.number_rendering_points]
        self.colors_rgb = self.precompute_colors(cluster_labels[:self.number_rendering_points])
        self.offsets = self.generate_circle_offsets(radius=3)

        # --- Pré-calculer les données statiques de rendu ---
        xs, ys = self.projection_2d.T
        self.min_x, self.max_x = xs.min(), xs.max()
        self.min_y, self.max_y = ys.min(), ys.max()

        x_norm = (xs - self.min_x) / (self.max_x - self.min_x + 1e-8)
        y_norm = (ys - self.min_y) / (self.max_y - self.min_y + 1e-8)
        self.main_x_canvas = (x_norm * (self.width - 1)).astype(np.int32)
        self.main_y_canvas = (y_norm * (self.height - 1)).astype(np.int32)

        # Répéter les couleurs pour chaque point (pour l'ensemble des offsets)
        self.main_colors_repeated = np.repeat(self.colors_rgb, len(self.offsets), axis=0)

        # Calculer les coordonnées finales de chaque point après application des offsets
        main_x = self.main_x_canvas[:, None] + self.offsets[:, 0]
        main_y = self.main_y_canvas[:, None] + self.offsets[:, 1]
        self.main_x_final = np.clip(main_x.ravel(), 0, self.width - 1)
        self.main_y_final = np.clip(main_y.ravel(), 0, self.height - 1)

        # Calculer l'image de fond statique (ne changeant pas entre les updates)
        self.background_image = self.compute_background()

        # Rendu initial
        self.draw_points()

    def setup_window(self):
        """Initialise la fenêtre Tkinter avec un canvas."""
        self.window = tk.Toplevel()
        self.window.title("Visualisation de l'espace latent 2D")
        self.canvas = tk.Canvas(self.window, width=self.width, height=self.height, bg='white')
        self.canvas.pack()
        self.canvas_image = self.canvas.create_image(0, 0, anchor=tk.NW, image=None)
        self.photo_image = None

    def generate_circle_offsets(self, radius):
        """Génère les offsets pour dessiner un cercle autour d'un point."""
        offsets = [
            (dx, dy)
            for dx in range(-radius, radius + 1)
            for dy in range(-radius, radius + 1)
            if dx ** 2 + dy ** 2 <= radius ** 2
        ]
        return np.array(offsets)

    def precompute_colors(self, cluster_labels):
        """Mappe les étiquettes de clusters à des couleurs RGB."""
        colors = np.array([
            [255, 0, 0],    # Rouge
            [0, 0, 255],    # Bleu
            [0, 255, 0],    # Vert
            [255, 165, 0],  # Orange
            [128, 0, 128],  # Violet
            [165, 42, 42],  # Marron
            [255, 192, 203],# Rose
            [128, 128, 128],# Gris
            [128, 128, 0],  # Olive
            [0, 255, 255]   # Cyan
        ], dtype=np.uint8)
        return colors[cluster_labels % len(colors)]

    def load_surrogate_policy(self, surrogate_policy_path):
        """Charge la politique substitutive avec gestion des erreurs."""
        try:
            return SurrogatePolicy.load_from_checkpoint(surrogate_policy_path)
        except Exception as e:
            raise RuntimeError(f"Erreur lors du chargement de la politique substitutive: {e}")

    def load_data(self, dataset_path):
        """Charge les données d'observations."""
        try:
            observations, = get_data(
                dataset_names=['observations'],
                data_number=self.number_projection_points,
                trajectory_dataset_file_path=dataset_path / 'trajectory_dataset.h5',
                device=self.device,
            )
            return observations
        except Exception as e:
            raise RuntimeError(f"Erreur lors du chargement des données: {e}")

    def compute_embeddings(self, observations):
        """Calcule les embeddings sur GPU."""
        return projection_clusterization_latent_space(
            observations=observations,
            surrogate_policy=self.surrogate_policy,
        ).to(self.device)

    def clusterize_embeddings(self, embeddings):
        """Effectue un clustering (KMeans) accéléré par GPU."""
        kmeans = KMeans(n_clusters=self.surrogate_policy.clusterization_function.number_cluster)
        return kmeans.fit_predict(cp.asarray(embeddings)).get()

    def compute_projection(self, embeddings):
        """Effectue une projection UMAP accélérée par GPU."""
        # self.projector_2d = UMAP(
        #     n_components=2,
        #     n_epochs=100_000,
        #     n_neighbors=30,
        # )

        self.projector_2d = PCA(
            n_components=2,
        )

        # self.projector_2d = TruncatedSVD(
        #     n_components=2,
        # )

        # self.projector_2d = GaussianRandomProjection(
        #     n_components=2,
        # )

        # self.projector_2d = SparseRandomProjection(
        #     n_components=2,
        # )

        return self.projector_2d.fit_transform(cp.asarray(embeddings)).get()

    def compute_background(self):
        """Calcule l'image de fond statique à partir des points principaux."""
        image = np.full((self.height, self.width, 3), 255, dtype=np.uint8)
        image[self.main_y_final, self.main_x_final] = self.main_colors_repeated
        return image

    def overlay_trajectory(self, image):
        """Superpose tous les points de trajectoire (avec alpha) sur une copie de l'image de fond."""
        traj_points = np.array([pt['point'] for pt in self.trajectory])  # forme: (T, 2)
        alphas = np.array([max(0.0, min(1.0, pt['alpha'])) for pt in self.trajectory])  # forme: (T,)

        # Normalisation des points de trajectoire sur le canvas
        x_traj = ((traj_points[:, 0] - self.min_x) / (self.max_x - self.min_x + 1e-8) * (self.width - 1)).astype(np.int32)
        y_traj = ((traj_points[:, 1] - self.min_y) / (self.max_y - self.min_y + 1e-8) * (self.height - 1)).astype(np.int32)

        # Application des offsets (pour dessiner le cercle autour de chaque point)
        x_final = x_traj[:, None] + self.offsets[:, 0]
        y_final = y_traj[:, None] + self.offsets[:, 1]

        # Aplatissement et limitation aux dimensions du canvas
        x_flat = np.clip(x_final.ravel(), 0, self.width - 1)
        y_flat = np.clip(y_final.ravel(), 0, self.height - 1)

        # Répéter les valeurs alpha pour chaque offset
        alphas_flat = np.repeat(alphas, len(self.offsets))
        trajectory_color = np.array([0, 255, 0], dtype=np.uint8)

        # Fusion (blending) des couleurs avec alpha blending
        current_colors = image[y_flat, x_flat].astype(np.float32)
        blended = (current_colors * (1 - alphas_flat[:, None]) + trajectory_color * alphas_flat[:, None]).astype(np.uint8)
        image[y_flat, x_flat] = blended

        return image

    def draw_points(self):
        """Rend la scène en combinant l'image de fond statique et la trajectoire dynamique."""
        image = self.background_image.copy()
        if self.trajectory:
            image = self.overlay_trajectory(image)

        pil_image = Image.fromarray(image)
        self.photo_image = ImageTk.PhotoImage(pil_image)
        self.canvas.itemconfig(self.canvas_image, image=self.photo_image)

    def update(self, observation=None):
        """
        Méthode update appelée environ 30 fois par seconde.
        Si une observation est fournie, elle est transformée en embedding et projetée,
        puis ajoutée à la trajectoire avec une opacité initiale de 1.0.
        Ensuite, la trajectoire est mise à jour (opacité diminuée) et l'affichage est rafraîchi.
        """
        if observation is not None:
            if not isinstance(observation, torch.Tensor):
                observation = torch.tensor(observation, dtype=torch.float32, device=self.device)
            if observation.dim() == 1:
                observation = observation.unsqueeze(0)
            embedding = self.compute_embeddings(observation)
            embedding_cp = cp.asarray(embedding)
            # Utilisation de l'instance déjà ajustée pour transformer l'embedding
            projected_cp = self.projector_2d.transform(embedding_cp)
            projected_point = projected_cp.get().squeeze()
            self.trajectory.append({'point': projected_point, 'alpha': 1.0})

        # Mise à jour des valeurs alpha et suppression des points expirés
        self.trajectory = [
            {'point': p['point'], 'alpha': p['alpha'] - 1.0 / self.number_point_trajectory}
            for p in self.trajectory if p['alpha'] - 1.0 / self.number_point_trajectory > 0
        ]

        self.draw_points()
        self.window.update_idletasks()
        self.window.update()

    def start(self):
        """Démarre la boucle principale Tkinter."""
        self.window.mainloop()
