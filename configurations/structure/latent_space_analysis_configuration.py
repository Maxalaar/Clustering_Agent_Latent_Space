

class LatentSpaceAnalysisConfiguration:
    def __init__(self):
        self.number_data: int = 500_000
        self.number_data_with_rending: int = 5_000
        self.number_points_for_silhouette_score: int = 10_000
        self.number_data_projection_2d: int = 10_000

        self.tree_max_depth_observations_to_all_clusters: int = 3
        self.tree_max_depth_observations_to_cluster = 2

        self.tree_max_depth_observations_to_actions: int = 3
        self.tree_max_depth_cluster_observations_to_actions: int = 2
