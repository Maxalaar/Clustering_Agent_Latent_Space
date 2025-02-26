import numpy as np
import torch
import os
import matplotlib
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from pathlib import Path
from typing import Optional
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, DecisionTreeRegressor
from sklearn.metrics import accuracy_score


def train_observations_clusters_decision_tree(
        observations: torch.Tensor,
        cluster_labels: torch.Tensor,
        save_path: Path,
        tree_max_depth_observations_to_all_clusters: int,
        tree_max_depth_observations_to_cluster: int,
        feature_names: Optional[list] = None,
):
    save_path = save_path / 'observations_clusters_decision_trees'
    os.makedirs(save_path, exist_ok=True)

    observations_cluster_accuracy_values = []

    observations = observations.cpu().numpy()
    cluster_labels = cluster_labels.cpu().numpy()

    class_names = []
    for label in np.unique(cluster_labels):
        class_names.append('cluster_' + str(label))

    x_train, x_test, y_train, y_test = train_test_split(observations, cluster_labels, test_size=0.2)

    decision_tree = DecisionTreeClassifier(max_depth=tree_max_depth_observations_to_all_clusters)
    decision_tree.fit(x_train, y_train)
    predict_y_test = decision_tree.predict(x_test)
    observations_cluster_accuracy_value = accuracy_score(y_test, predict_y_test)
    information = 'Decision tree (observations -> all clusters), max depth: ' + str(decision_tree.max_depth) + ', accuracy: ' + str(observations_cluster_accuracy_value) + '\n'
    print(information)
    with open(save_path / 'information.txt', 'a') as file:
        file.write(information)

    plt.figure(figsize=(12, 12))
    plot_tree(decision_tree, filled=True, feature_names=feature_names, class_names=class_names)
    plt.savefig(save_path / 'all_clusters_decision_tree.png', bbox_inches='tight', dpi=300)

    for label in np.unique(cluster_labels):
        y_binary = (cluster_labels == label).astype(int)
        class_names = ['other_cluster', 'cluster_' + str(label)]
        random_over_sampler = RandomOverSampler(sampling_strategy=1.0)
        x_balance, y_balance = random_over_sampler.fit_resample(observations, y_binary)
        x_train, x_test, y_train, y_test = train_test_split(x_balance, y_balance, test_size=0.2)

        decision_tree = DecisionTreeClassifier(max_depth=tree_max_depth_observations_to_cluster)
        decision_tree.fit(x_train, y_train)
        predict_y_test = decision_tree.predict(x_test)
        observations_cluster_accuracy_value = accuracy_score(y_test, predict_y_test)
        observations_cluster_accuracy_values.append(observations_cluster_accuracy_value)
        information = 'Decision tree (observations -> cluster ' + str(label) + '), max depth: ' + str(decision_tree.max_depth) + ', accuracy: ' + str(observations_cluster_accuracy_value) + '\n'
        print(information)
        with open(save_path / 'information.txt', 'a') as file:
            file.write(information)

        plt.figure(figsize=(12, 12))
        plot_tree(decision_tree, filled=True, feature_names=feature_names, class_names=class_names)
        plt.savefig(save_path / ('cluster_' + str(label) + '_decision_tree.png'), bbox_inches='tight', dpi=300)
        matplotlib.pyplot.close()

    return observations_cluster_accuracy_values