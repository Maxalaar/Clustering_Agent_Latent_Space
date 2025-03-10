import numpy as np
import torch
import os
import matplotlib
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from pathlib import Path
from typing import Optional
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score


def train_observations_actions_decision_tree(
        observations: torch.Tensor,
        actions: torch.Tensor,
        cluster_labels: torch.Tensor,
        save_path: Path,
        tree_max_depth_observations_to_actions,
        feature_names: Optional[list] = None,
        class_names: Optional[list] = None,
        use_random_oversampler: bool = True,  # Option pour activer le RandomOverSampler
):
    save_path = save_path / 'observations_actions_decision_trees'
    os.makedirs(save_path, exist_ok=True)

    observations = observations.cpu().numpy()
    is_convertible_to_int = torch.all(actions == actions.to(torch.int))

    if not is_convertible_to_int:
        return

    actions = actions.cpu().numpy()
    cluster_labels = cluster_labels.cpu().numpy()

    # Arbres de décision globaux par action
    unique_global_actions = np.unique(actions)
    for action in unique_global_actions:
        # Créer une cible binaire : 1 pour l'action courante, 0 pour toutes les autres
        y_binary_global = (actions == action).astype(int)
        if len(np.unique(y_binary_global)) < 2:
            print(f"Skipping global action {action} due to single class.")
            continue

        # Application optionnelle du RandomOverSampler si activé
        if use_random_oversampler:
            ros = RandomOverSampler()
            try:
                x_res_global, y_res_global = ros.fit_resample(observations, y_binary_global)
            except ValueError as e:
                print(f"Error processing global action {action}: {e}")
                continue
        else:
            x_res_global, y_res_global = observations, y_binary_global

        x_train, x_test, y_train, y_test = train_test_split(
            x_res_global, y_res_global, test_size=0.2, stratify=y_res_global
        )

        decision_tree_global = DecisionTreeClassifier(max_depth=tree_max_depth_observations_to_actions)
        decision_tree_global.fit(x_train, y_train)
        y_pred = decision_tree_global.predict(x_test)
        accuracy_value = accuracy_score(y_test, y_pred)

        if class_names is not None and action < len(class_names):
            action_name = class_names[action]
        else:
            action_name = str(action)
        class_names_binary = ['other', action_name]

        information = (
            f'Global decision tree for action {action_name}, max depth: {decision_tree_global.max_depth}, '
            f'accuracy: {accuracy_value}\n'
        )
        print(information)
        with open(save_path / 'information.txt', 'a') as file:
            file.write(information)

        plt.figure(figsize=(12, 12))
        plot_tree(
            decision_tree_global,
            filled=True,
            feature_names=feature_names,
            class_names=class_names_binary
        )
        plt.savefig(
            save_path / f'global_action_{action}_decision_tree.png',
            bbox_inches='tight',
            dpi=300
        )
        plt.close()

    # Pour chaque cluster, création d'arbres de décision binaires pour chaque action
    for label in np.unique(cluster_labels):
        indices = np.where(cluster_labels == label)[0]
        observations_current_cluster = observations[indices]
        actions_current_cluster = actions[indices]

        unique_actions = np.unique(actions_current_cluster)
        for action in unique_actions:
            # Création d'une cible binaire : 1 pour l'action courante, 0 pour toutes les autres
            y_binary = (actions_current_cluster == action).astype(int)

            if len(np.unique(y_binary)) < 2:
                print(f"Skipping action {action} in cluster {label} due to single class.")
                continue

            # Application optionnelle du RandomOverSampler si activé
            if use_random_oversampler:
                ros = RandomOverSampler()
                try:
                    x_res, y_res = ros.fit_resample(observations_current_cluster, y_binary)
                except ValueError as e:
                    print(f"Error processing cluster {label}, action {action}: {e}")
                    continue
            else:
                x_res, y_res = observations_current_cluster, y_binary

            x_train, x_test, y_train, y_test = train_test_split(
                x_res, y_res, test_size=0.2, stratify=y_res
            )

            decision_tree_cluster = DecisionTreeClassifier(
                max_depth=tree_max_depth_observations_to_actions
            )
            decision_tree_cluster.fit(x_train, y_train)
            y_pred = decision_tree_cluster.predict(x_test)
            accuracy_value = accuracy_score(y_test, y_pred)

            if class_names is not None and action < len(class_names):
                action_name = class_names[action]
            else:
                action_name = str(action)
            class_names_binary = ['other', action_name]

            information = (
                f'Decision tree cluster {label}, action {action_name} (observations -> action), '
                f'max depth: {decision_tree_cluster.max_depth}, accuracy: {accuracy_value}\n'
            )
            print(information)
            with open(save_path / 'information.txt', 'a') as file:
                file.write(information)

            plt.figure(figsize=(12, 12))
            plot_tree(
                decision_tree_cluster,
                filled=True,
                feature_names=feature_names,
                class_names=class_names_binary
            )
            plt.savefig(
                save_path / f'cluster_{label}_action_{action}_decision_tree.png',
                bbox_inches='tight',
                dpi=300
            )
            plt.close()
