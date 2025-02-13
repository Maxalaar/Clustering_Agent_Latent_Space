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


def train_observations_actions_decision_tree(
        observations: torch.Tensor,
        actions: torch.Tensor,
        cluster_labels: torch.Tensor,
        save_path: Path,
        feature_names: Optional[list] = None,
        class_names: Optional[list] = None,
):
    save_path = save_path / 'observations_actions_decision_trees'
    os.makedirs(save_path, exist_ok=True)

    observations = observations.cpu().numpy()
    is_convertible_to_int = torch.all(actions == actions.to(torch.int))

    if not is_convertible_to_int:
        return

    actions = actions.cpu().numpy()
    cluster_labels = cluster_labels.cpu().numpy()

    x_train, x_test, y_train, y_test = train_test_split(observations, actions, test_size=0.2)

    decision_tree = DecisionTreeClassifier(max_depth=2)
    decision_tree.fit(x_train, y_train)
    predict_y_test = decision_tree.predict(x_test)
    accuracy_value = accuracy_score(y_test, predict_y_test)
    information = 'Decision tree (all clusters: observations -> actions), max depth: ' + str(
        decision_tree.max_depth) + ', accuracy: ' + str(accuracy_value) + '\n'
    print(information)
    with open(save_path / 'information.txt', 'a') as file:
        file.write(information)

    plt.figure(figsize=(12, 12))
    plot_tree(decision_tree, filled=True, feature_names=feature_names, class_names=class_names)
    plt.savefig(save_path / 'all_clusters_decision_tree.png', bbox_inches='tight', dpi=300)

    for label in np.unique(cluster_labels):
        indices = np.where(cluster_labels == label)[0]
        observations_current_cluster = observations[indices]
        actions_current_cluster = actions[indices]

        random_over_sampler = RandomOverSampler()
        x_balance, y_balance = random_over_sampler.fit_resample(observations_current_cluster, actions_current_cluster)
        x_train, x_test, y_train, y_test = train_test_split(x_balance, y_balance, test_size=0.2)
        decision_tree = DecisionTreeClassifier(max_depth=2)
        decision_tree.fit(x_train, y_train)

        predict_y_test = decision_tree.predict(x_test)
        accuracy_value = accuracy_score(y_test, predict_y_test)
        information = 'Decision tree cluster ' + str(label) + ' (observations -> actions), max depth: ' + str(decision_tree.max_depth) + ', accuracy: ' + str(accuracy_value) + '\n'
        print(information)
        with open(save_path / 'information.txt', 'a') as file:
            file.write(information)

        plt.figure(figsize=(12, 12))
        plot_tree(decision_tree, filled=True, feature_names=feature_names, class_names=class_names)
        plt.savefig(save_path / ('cluster_' + str(label) + '_observations_actions_decision_tree.png'), bbox_inches='tight', dpi=300)
        matplotlib.pyplot.close()

    #
    #     if is_convertible_to_int:
    #         decision_tree = DecisionTreeClassifier(max_depth=2)
    #         decision_tree.fit(x_train, y_train)
    #     else:
    #         return
    #         # decision_tree = DecisionTreeRegressor()
    #         # decision_tree.fit(x_train, y_train)
    #
    #     predict_y_test = decision_tree.predict(x_test)
    #     accuracy_value = accuracy_score(y_test, predict_y_test)
    #     information = 'Decision tree cluster ' + str(label) + ' (observations -> actions), max depth: ' + str(decision_tree.max_depth) + ', accuracy: ' + str(accuracy_value) + '\n'
    #     print(information)
    #     with open(save_path / 'information.txt', 'a') as file:
    #         file.write(information)
    #
    #     plt.figure(figsize=(12, 12))
    #     plot_tree(decision_tree, filled=True, feature_names=feature_names, class_names=class_names)
    #     plt.savefig(save_path / ('cluster_' + str(label) + '_observations_actions_decision_tree.png'), bbox_inches='tight', dpi=300)
    #     matplotlib.pyplot.close()