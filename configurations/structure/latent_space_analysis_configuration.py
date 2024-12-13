

class LatentSpaceAnalysisConfiguration:
    def __init__(self):
        self.number_observations: int = 1_000_000
        self.number_observations_with_rending: int = 30_000
        self.number_points_for_silhouette_score: int = 10_000
        self.number_data_projection_2d = 10_000
