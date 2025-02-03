import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def get_linear_modules(module, prefix=""):
    """
    Renvoie la liste des sous-modules Linear sous forme de tuples (nom_complet, module)
    dans l'ordre d'apparition.
    """
    linear_modules = []
    for name, child in module.named_children():
        child_name = f"{prefix}.{name}" if prefix else name
        if name == 'critic_layers':
            pass
        elif isinstance(child, nn.Linear):
            linear_modules.append((child_name, child))
        else:
            linear_modules.extend(get_linear_modules(child, prefix=child_name))
    return linear_modules


class GraphVisualizer:
    def __init__(self, model, n_edges=2):
        """
        model : le modèle à visualiser.
        n_edges : nombre de liaisons (pour les contributions positives et négatives)
                  à afficher par connexion entre couches.
        """
        self.model = model
        self.n_edges = n_edges
        # On attend que le modèle enregistre ses hooks, y compris celui pour l'entrée.
        self.model._register_hooks()
        # La liste des "couches" contient désormais la couche d'entrée suivie des couches linéaires.
        # Pour la couche d'entrée, on utilise le nom "input" et None pour le module.
        self.layers = [("input", None)] + get_linear_modules(self.model)
        # Calcul des positions de chaque neurone pour toutes les couches.
        self.node_order, self.node_positions = self._compute_node_positions()

        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        xs = [pos[0] for pos in self.node_positions.values()]
        ys = [pos[1] for pos in self.node_positions.values()]
        self.nodes = self.ax.scatter(xs, ys, s=500, c=[0.5] * len(xs),
                                     cmap=plt.cm.coolwarm, vmin=-1, vmax=1)

        self.ax.set_axis_off()
        self.fig.tight_layout()

        # Liste des lignes (arêtes) actuellement affichées
        self.edges = []
        self.cmap = plt.cm.coolwarm

    def _compute_node_positions(self):
        """
        Calcule la position de chaque neurone pour chaque couche.
        Pour la couche d'entrée, on prend le nombre de neurones défini dans model.input_dim.
        Retourne :
          - node_order : une liste ordonnée des clés (layer_index, neuron_index)
          - node_positions : un dictionnaire { (layer_index, neuron_index) : (x, y) }
        """
        node_positions = {}
        node_order = {}
        # Dictionnaire qui donne le nombre de neurones pour chaque couche.
        neurons_per_layer = {}
        for idx, (name, module) in enumerate(self.layers):
            if name == "input":
                # On suppose que le modèle possède un attribut input_dim
                neurons_per_layer[idx] = self.model.inpout_size
            else:
                neurons_per_layer[idx] = module.out_features

        max_neurons = max(neurons_per_layer.values()) if neurons_per_layer else 1

        for layer_idx, num_neurons in neurons_per_layer.items():
            for j in range(num_neurons):
                # Position horizontale en fonction de la couche
                x = layer_idx
                # Position verticale : centrée par rapport au maximum de neurones
                y = (max_neurons - num_neurons) / 2 + j
                node_positions[(layer_idx, j)] = (x, y)
                node_order.setdefault(layer_idx, []).append(j)
        # node_order peut être utilisé si vous souhaitez garder l'ordre par couche.
        return node_order, node_positions

    def update(self, frame):
        """
        Pour chaque frame, met à jour les couleurs des nœuds (en fonction de leur activation)
        et trace les arêtes correspondant aux contributions les plus importantes.
        """
        # Mise à jour des couleurs des nœuds.
        node_colors = []
        all_activations = []
        # Parcourt les couches et les neurones de chaque couche.
        for layer_idx, neuron_list in self.node_order.items():
            for j in neuron_list:
                if layer_idx == 0:
                    # Pour la couche d'entrée, l'activation se trouve dans self.model.activations["input"]
                    if hasattr(self.model, 'activations') and "input" in self.model.activations:
                        act = self.model.activations["input"][0, j].item()
                    else:
                        act = 0.0
                else:
                    layer_name, _ = self.layers[layer_idx]
                    if hasattr(self.model, 'activations') and layer_name in self.model.activations:
                        act = self.model.activations[layer_name][0, j].item()
                    else:
                        act = 0.0
                all_activations.append(act)
        max_abs_act = max(np.max(np.abs(all_activations)), 1e-8)

        # Re-parcourt pour attribuer une couleur normalisée à chaque nœud.
        for layer_idx, neuron_list in self.node_order.items():
            for j in neuron_list:
                if layer_idx == 0:
                    if hasattr(self.model, 'activations') and "input" in self.model.activations:
                        act = self.model.activations["input"][0, j].item()
                    else:
                        act = 0.0
                else:
                    layer_name, _ = self.layers[layer_idx]
                    if hasattr(self.model, 'activations') and layer_name in self.model.activations:
                        act = self.model.activations[layer_name][0, j].item()
                    else:
                        act = 0.0
                node_colors.append(act / max_abs_act)
        self.nodes.set_array(np.array(node_colors))

        # Supprime les arêtes affichées précédemment.
        for line in self.edges:
            line.remove()
        self.edges = []

        # Pour chaque connexion entre deux couches consécutives,
        # on calcule la contribution de chaque lien et on trace uniquement les plus marquantes.
        # La contribution est définie par : activation_source * poids
        for i in range(len(self.layers) - 1):
            # Récupération de l'activation de la couche source.
            if i == 0:
                # Couche d'entrée.
                if not (hasattr(self.model, 'activations') and "input" in self.model.activations):
                    continue
                activation_src = self.model.activations["input"][0].detach().cpu().numpy()  # Shape: (input_dim,)
            else:
                layer_name_src, _ = self.layers[i]
                if not (hasattr(self.model, 'activations') and layer_name_src in self.model.activations):
                    continue
                activation_src = self.model.activations[layer_name_src][
                    0].detach().cpu().numpy()  # Shape: (src_neurons,)

            # La couche cible est toujours un module linéaire.
            _, tgt_module = self.layers[i + 1]
            weight_matrix = tgt_module.weight.data.detach().cpu().numpy()
            # Pour la couche cible, la matrice de poids a pour forme (tgt_neurons, src_neurons)
            # Calcul des contributions : pour chaque connexion, contribution = activation_src[j] * weight_matrix[k, j]
            contributions = weight_matrix * activation_src  # Shape: (tgt_neurons, src_neurons)
            contrib_flat = contributions.flatten()
            num_total = contrib_flat.size
            n = self.n_edges
            if num_total == 0:
                continue

            # Sélection des indices des n contributions les plus positives et négatives.
            if n < num_total:
                pos_indices = np.argpartition(-contrib_flat, n - 1)[:n]
                neg_indices = np.argpartition(contrib_flat, n - 1)[:n]
            else:
                pos_indices = np.arange(num_total)
                neg_indices = np.arange(num_total)
            selected_indices = np.unique(np.concatenate([pos_indices, neg_indices]))

            max_abs_contrib = max(np.max(np.abs(contrib_flat)), 1e-8)
            src_neurons = weight_matrix.shape[1]
            for idx in selected_indices:
                k = idx // src_neurons  # indice de la cible
                j = idx % src_neurons  # indice de la source
                start = self.node_positions.get((i, j), None)
                end = self.node_positions.get((i + 1, k), None)
                if start is None or end is None:
                    continue
                contrib = contrib_flat[idx]
                normalized = contrib / max_abs_contrib
                color = self.cmap(normalized / 2 + 0.5)
                alpha = abs(normalized)
                line, = self.ax.plot([start[0], end[0]], [start[1], end[1]],
                                     color=color, alpha=alpha, linewidth=2)
                self.edges.append(line)

        return [self.nodes] + self.edges