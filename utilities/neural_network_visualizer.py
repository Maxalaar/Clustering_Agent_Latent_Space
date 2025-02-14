import torch
import torch.nn as nn
import tkinter as tk


class NeuralNetworkVisualizer:
    def __init__(self, model, width=800, height=600, neuron_radius=20, max_connections_per_layer=None):
        self.model = model
        self.width = width
        self.height = height
        self.neuron_radius = neuron_radius
        self.max_connections_per_layer = max_connections_per_layer

        self.window = tk.Tk()
        self.window.title("Neural Network Visualizer")
        self.canvas = tk.Canvas(self.window, width=self.width, height=self.height, bg='white')
        self.canvas.pack()

        self.layers = self.extract_layers(model)
        self.linear_layers = [module for module in model if isinstance(module, nn.Linear)]
        self.positions = self.compute_positions()

        self.draw_network()
        self.window.update()

        self._activations = {}
        self.hook_handles = []

        if not self.linear_layers:
            return

        # Register hooks
        handle = self.linear_layers[0].register_forward_pre_hook(
            lambda module, input: self._save_activation(0, input[0]))
        self.hook_handles.append(handle)

        for i in range(1, len(self.linear_layers)):
            layer = self.linear_layers[i]
            handle = layer.register_forward_pre_hook(
                lambda module, input, idx=i: self._save_activation(idx, input[0]))
            self.hook_handles.append(handle)

        last_layer_idx = len(self.linear_layers) - 1
        handle = self.linear_layers[last_layer_idx].register_forward_hook(
            lambda module, input, output: self._save_activation(last_layer_idx + 1, output))
        self.hook_handles.append(handle)

    def extract_layers(self, model):
        sizes = []
        for module in model:
            if isinstance(module, nn.Linear):
                if not sizes:
                    sizes.append(module.in_features)
                sizes.append(module.out_features)
        return sizes

    def compute_positions(self):
        positions = []
        n_layers = len(self.layers)
        x_spacing = self.width / (n_layers - 1) if n_layers > 1 else self.width / 2
        for i, n_neurons in enumerate(self.layers):
            x = i * x_spacing
            y_spacing = self.height / (n_neurons + 1)
            layer_positions = [(x, (j + 1) * y_spacing) for j in range(n_neurons)]
            positions.append(layer_positions)
        return positions

    def draw_network(self):
        self.neuron_items = []
        self.connection_items = []

        for layer in self.positions:
            neuron_ids = [self.create_neuron(x, y) for (x, y) in layer]
            self.neuron_items.append(neuron_ids)

        for i in range(len(self.positions) - 1):
            connection_layer = []
            weights = self.linear_layers[i].weight.data if i < len(self.linear_layers) else None
            for j, (x1, y1) in enumerate(self.positions[i]):
                for k, (x2, y2) in enumerate(self.positions[i + 1]):
                    color = self.weight_to_color(weights[k, j].item()) if weights is not None else 'grey'
                    line_id = self.create_connection(x1, y1, x2, y2, color)
                    connection_layer.append(line_id)
            self.connection_items.append(connection_layer)

    def create_neuron(self, x, y):
        return self.canvas.create_oval(
            x - self.neuron_radius, y - self.neuron_radius,
            x + self.neuron_radius, y + self.neuron_radius,
            fill='grey', outline='black'
        )

    def create_connection(self, x1, y1, x2, y2, color):
        return self.canvas.create_line(
            x1 + self.neuron_radius, y1,
            x2 - self.neuron_radius, y2,
            fill=color
        )

    def _save_activation(self, idx, tensor):
        act = tensor.detach().cpu()[0] if tensor.dim() > 1 else tensor.detach().cpu()
        self._activations[idx] = act.tolist()

    def activation_to_color(self, activation):
        max_val = 1.0
        try:
            act = float(activation)
        except:
            act = 0.0

        clamped = max(min(act, max_val), -max_val)
        if clamped < 0:
            t = (clamped + max_val) / max_val
            red = green = int(255 * t)
            blue = 255
        else:
            t = clamped / max_val
            red = 255
            green = blue = int(255 * (1 - t))

        red = max(0, min(255, red))
        green = max(0, min(255, green))
        blue = max(0, min(255, blue))
        return f'#{red:02x}{green:02x}{blue:02x}'

    def weight_to_color(self, weight):
        return self.activation_to_color(weight)

    def update(self):
        # Update neurons
        for i in range(len(self.neuron_items)):
            if i in self._activations:
                for j, neuron_id in enumerate(self.neuron_items[i]):
                    color = self.activation_to_color(self._activations[i][j])
                    self.canvas.itemconfig(neuron_id, fill=color)

        # Update connections
        for i in range(len(self.connection_items)):
            if i >= len(self.linear_layers):
                continue
            weights = self.linear_layers[i].weight.data
            activations = self._activations.get(i, [])
            if not activations:
                continue

            next_size = len(self.positions[i + 1])
            strengths = []
            for j in range(len(activations)):
                for k in range(next_size):
                    strengths.append(activations[j] * weights[k, j].item())

                if not strengths:
                    continue

            max_abs = max(abs(s) for s in strengths) or 1.0
            top_connections = set()
            if self.max_connections_per_layer is not None:
                sorted_conn = sorted(enumerate(strengths), key=lambda x: -abs(x[1]))
                top_connections = set(idx for idx, _ in sorted_conn[:self.max_connections_per_layer])

            for c, line_id in enumerate(self.connection_items[i]):
                if self.max_connections_per_layer is not None and c not in top_connections:
                    self.canvas.itemconfig(line_id, fill='#cccccc', width=1)
                else:
                    j, k = c // next_size, c % next_size
                    strength = activations[j] * weights[k, j].item()
                    norm_strength = strength / max_abs
                    color = self.activation_to_color(norm_strength)
                    thickness = 1 + 3 * abs(norm_strength)
                    self.canvas.itemconfig(line_id, fill=color, width=thickness)

            self.window.update_idletasks()
            self.window.update()

    def start(self):
        self.window.mainloop()


# Example usage
if __name__ == '__main__':
    model = nn.Sequential(
        nn.Linear(4, 5),
        nn.ReLU(),
        nn.Linear(5, 3)
    )
    vis = NeuralNetworkVisualizer(model, max_connections_per_layer=10)


    def update_loop():
        inp = torch.randn(1, 4)
        _ = model(inp)
        vis.update()
        vis.window.after(100, update_loop)


    update_loop()
    vis.start()