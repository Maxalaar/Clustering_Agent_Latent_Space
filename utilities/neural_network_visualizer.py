import torch
import torch.nn as nn
import tkinter as tk


class NeuralNetworkVisualizer:
    def __init__(self, model, width=800, height=600, neuron_radius=20):
        """
        model : instance de torch.nn.Module (ici supposé être nn.Sequential avec des Linear)
        width, height : dimensions de la fenêtre d'affichage
        neuron_radius : rayon des cercles représentant les neurones
        """
        self.model = model
        self.width = width
        self.height = height
        self.neuron_radius = neuron_radius

        # Création de la fenêtre Tkinter
        self.window = tk.Tk()
        self.window.title("Neural Network Visualizer")
        self.canvas = tk.Canvas(self.window, width=self.width, height=self.height, bg='white')
        self.canvas.pack()

        # Extraction des tailles de couches (la première valeur correspond à l'entrée)
        self.layers = self.extract_layers(model)

        # Initialisation de la liste des modules linéaires
        self.linear_layers = []
        for module in model:
            if isinstance(module, nn.Linear):
                self.linear_layers.append(module)

        # Calcul des positions des neurones pour chaque couche
        self.positions = self.compute_positions()

        # Dessin initial du réseau (neurones et connexions)
        self.draw_network()
        self.window.update()

        # Dictionnaire pour stocker les activations issues des hooks.
        # La clé 0 correspond aux activations de la couche d'entrée,
        # et pour chaque module Linear, la clé correspond à (indice du module + 1)
        self._activations = {}

        # Enregistrement du hook sur le premier module pour capter l'entrée
        if len(self.linear_layers) > 0:
            handle_in = self.linear_layers[0].register_forward_pre_hook(self._save_input)
        else:
            handle_in = None

        # Enregistrement des hooks sur les modules linéaires (forward hooks)
        self.hook_handles = []
        if handle_in is not None:
            self.hook_handles.append(handle_in)
        for i, layer in enumerate(self.linear_layers):
            # Le hook forward permettra de récupérer la sortie de chaque couche linéaire.
            # Pour la couche i, on sauvegarde dans la clé i+1 (car la clé 0 correspond à l'entrée)
            handle = layer.register_forward_hook(
                lambda module, input, output, idx=i: self._save_activation(idx, output))
            self.hook_handles.append(handle)

    def extract_layers(self, model):
        """
        Extrait le nombre de neurones pour chaque couche.
        La première valeur correspond à in_features du premier module Linear,
        et chaque module Linear ajoute son out_features.
        """
        sizes = []
        for module in model:
            if isinstance(module, nn.Linear):
                if not sizes:
                    sizes.append(module.in_features)
                sizes.append(module.out_features)
        return sizes

    def compute_positions(self):
        """
        Calcule les positions (x, y) de chaque neurone dans la fenêtre.
        Les couches sont réparties horizontalement, et dans chaque couche, les neurones sont espacés verticalement.
        """
        positions = []
        n_layers = len(self.layers)
        x_spacing = self.width / (n_layers - 1) if n_layers > 1 else self.width / 2
        for i, n_neurons in enumerate(self.layers):
            x = i * x_spacing
            y_spacing = self.height / (n_neurons + 1)
            layer_positions = []
            for j in range(n_neurons):
                y = (j + 1) * y_spacing
                layer_positions.append((x, y))
            positions.append(layer_positions)
        return positions

    def draw_network(self):
        """
        Dessine initialement le réseau sur le canvas :
         - Chaque neurone est dessiné sous forme de cercle.
         - Les connexions entre les couches consécutives sont représentées par des lignes,
           dont la couleur est déterminée par les poids.
        """
        self.neuron_items = []  # Liste de listes : neurones par couche
        self.connection_items = []  # Liste de listes : connexions entre couches

        # Dessin des neurones pour chaque couche
        for layer in self.positions:
            neuron_ids = []
            for (x, y) in layer:
                neuron_id = self.canvas.create_oval(
                    x - self.neuron_radius, y - self.neuron_radius,
                    x + self.neuron_radius, y + self.neuron_radius,
                    fill='grey', outline='black'
                )
                neuron_ids.append(neuron_id)
            self.neuron_items.append(neuron_ids)

        # Dessin des connexions entre chaque paire de couches consécutives
        for i in range(len(self.positions) - 1):
            connection_layer = []
            if i < len(self.linear_layers):
                layer_module = self.linear_layers[i]
                weights = layer_module.weight.data  # Taille : (neurones_out, neurones_in)
            else:
                weights = None

            for j, (x1, y1) in enumerate(self.positions[i]):
                for k, (x2, y2) in enumerate(self.positions[i + 1]):
                    color = 'grey'
                    if weights is not None:
                        # Les poids sont indexés [k, j] pour la connexion du neurone j de la couche i
                        # au neurone k de la couche i+1
                        weight = weights[k, j].item()
                        color = self.weight_to_color(weight)
                    line_id = self.canvas.create_line(
                        x1 + self.neuron_radius, y1,
                        x2 - self.neuron_radius, y2,
                        fill=color
                    )
                    connection_layer.append(line_id)
            self.connection_items.append(connection_layer)

    def _save_input(self, module, input):
        """
        Hook pré-forward pour capturer l'entrée du réseau.
        input est un tuple ; on considère ici le premier élément (le tenseur d'entrée).
        """
        inp = input[0]
        if inp.dim() > 1:
            act = inp.detach().cpu()[0]
        else:
            act = inp.detach().cpu()
        self._activations[0] = act.tolist()

    def _save_activation(self, idx, output):
        """
        Hook forward pour capturer la sortie d'un module linéaire.
        On sauvegarde la sortie (premier échantillon du batch) dans self._activations
        sous la clé idx + 1 (car 0 correspond à l'entrée).
        """
        if output.dim() > 1:
            act = output.detach().cpu()[0]
        else:
            act = output.detach().cpu()
        self._activations[idx + 1] = act.tolist()

    def activation_to_color(self, activation):
        """
        Convertit une valeur d'activation en couleur.
        Rouge pour positif, bleu pour négatif, gris pour zéro.
        L'intensité est proportionnelle à la valeur absolue (clippée à 1).
        """
        try:
            act = float(activation)
        except Exception:
            act = 0.0
        norm = max(min(abs(act), 1), 0)
        intensity = int(255 * norm)
        if act > 0:
            color = f'#{intensity:02x}0000'
        elif act < 0:
            color = f'#0000{intensity:02x}'
        else:
            color = '#cccccc'
        return color

    def weight_to_color(self, weight):
        """
        Convertit la valeur d'un poids en couleur (même logique que pour les activations).
        """
        try:
            w = float(weight)
        except Exception:
            w = 0.0
        norm = max(min(abs(w), 1), 0)
        intensity = int(255 * norm)
        if w > 0:
            color = f'#{intensity:02x}0000'
        elif w < 0:
            color = f'#0000{intensity:02x}'
        else:
            color = '#cccccc'
        return color

    def update(self):
        """
        Met à jour l'affichage en se basant sur les activations sauvegardées.
        Cette méthode parcourt toutes les couches (y compris l'entrée, indice 0)
        et met à jour la couleur des neurones.
        """
        for i in range(len(self.neuron_items)):
            if i in self._activations:
                activations = self._activations[i]
                for j, act in enumerate(activations):
                    color = self.activation_to_color(act)
                    self.canvas.itemconfig(self.neuron_items[i][j], fill=color)
        self.window.update_idletasks()
        self.window.update()

    def start(self):
        """Lance la boucle principale Tkinter."""
        self.window.mainloop()


# --- Exemple d'utilisation ---
if __name__ == '__main__':
    # Exemple de modèle simple
    model = nn.Sequential(
        nn.Linear(4, 5),
        nn.ReLU(),
        nn.Linear(5, 3)
    )

    vis = NeuralNetworkVisualizer(model)


    # Simule des passes forward du modèle en générant des entrées aléatoires.
    # Les hooks se chargent de récupérer les activations et l'entrée.
    def update_loop():
        inp = torch.randn(1, model[0].in_features)
        _ = model(inp)
        vis.update()
        vis.window.after(100, update_loop)


    update_loop()
    vis.start()
