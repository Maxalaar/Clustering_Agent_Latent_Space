import torch
import numpy as np

from sklearn.metrics import rand_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score


def compare_clustering_between_surrogate_policies(surrogate_policies_cluster_labels):
    rand_score_values = []
    adjusted_rand_score_values = []
    normalized_mutual_info_values = []

    for surrogate_policy_cluster_labels_1 in surrogate_policies_cluster_labels:
        for surrogate_policy_cluster_labels_2 in surrogate_policies_cluster_labels:
            if surrogate_policy_cluster_labels_1 is not surrogate_policy_cluster_labels_2:
                rand_score_values.append(rand_score(surrogate_policy_cluster_labels_1.cpu(), surrogate_policy_cluster_labels_2.cpu()))
                adjusted_rand_score_values.append(adjusted_rand_score(surrogate_policy_cluster_labels_1.cpu(), surrogate_policy_cluster_labels_2.cpu()))
                normalized_mutual_info_values.append(normalized_mutual_info_score(surrogate_policy_cluster_labels_1.cpu(), surrogate_policy_cluster_labels_2.cpu()))

    information = 'Compare clustering between surrogate policies: \n'
    information += 'Rand score: ' + np.array(rand_score_values).mean() + '\n'
    information += 'Adjusted Rand score: ' + np.array(adjusted_rand_score_values).mean() + '\n'
    information += 'Normalized Mutual Information score: ' + np.array(normalized_mutual_info_values).mean() + '\n'
    print(information)