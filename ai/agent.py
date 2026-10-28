"""
Agente inteligente que aprende a jugar al Snake mediante DQN (Deep Q-Network).
"""

import os
import random
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ai.model import SnakeNet
from ai.replay_memory import ReplayMemory


class DQNAgent:
    """Agente que decide movimientos y aprende mediante DQN."""

    def __init__(self):
        # Hiperparámetros de aprendizaje
        self.gamma = 0.9              # Factor de descuento para recompensas futuras
        self.epsilon = 1.0            # Probabilidad inicial de exploración aleatoria
        self.epsilon_min = 0.01       # Mínimo valor de exploración permitido
        self.epsilon_decay = 0.995    # Tasa de reducción de epsilon por episodio
        self.learning_rate = 0.001    # Tasa de aprendizaje de Adam
        self.batch_size = 100         # Tamaño de minilote para entrenar la red
        self.memory_size = 100_000    # Capacidad máxima del buffer de experiencias

        # Red neuronal principal (Policy Network)
        self.model = SnakeNet(input_size=11, output_size=3)

        # Red objetivo (Target Network) para estabilizar los valores futuros
        self.target_model = SnakeNet(input_size=11, output_size=3)
        self.update_target_network()
        self.target_model.eval()

        # Optimizador y función de pérdida (MSE entre Q predicho y Q objetivo)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        # Memoria de repetición de experiencias
        self.memory = ReplayMemory(capacity=self.memory_size)

    def update_target_network(self) -> None:
        """Copia los pesos de la red principal a la red objetivo."""
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, state: List[int]) -> int:
        """
        Devuelve una acción:
        0 = girar a la izquierda (relativo a la cabeza)
        1 = seguir recto
        2 = girar a la derecha (relativo a la cabeza)
        """
        # Exploración (epsilon-greedy): probar una acción al azar
        if random.random() < self.epsilon:
            return random.randint(0, 2)

        # Explotación: elegir la acción con mayor valor Q estimado
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            predictions = self.model(state_tensor)

        return int(torch.argmax(predictions, dim=1).item())

    def remember(
        self,
        state: List[int],
        action: int,
        reward: float,
        next_state: List[int],
        done: bool,
    ) -> None:
        """Guarda una experiencia en la memoria de repetición."""
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self) -> float:
        """
        Entrena la red principal utilizando un lote aleatorio de la memoria.
        Calcula la ecuación de Bellman: Q_target = r + gamma * max(Q_target(s')) * (1 - done).
        """
        if len(self.memory) < self.batch_size:
            return 0.0

        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_tensor = torch.tensor(np.array(states), dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones_tensor = torch.tensor(dones, dtype=torch.float32)

        # Valores Q actuales que predice la red para la acción que se tomó:
        current_q = self.model(states_tensor).gather(
            1, actions_tensor.unsqueeze(1)
        ).squeeze(1)

        # Objetivo de Bellman usando la red objetivo:
        with torch.no_grad():
            next_q = self.target_model(next_states_tensor).max(dim=1)[0]
            target_q = rewards_tensor + self.gamma * next_q * (1 - dones_tensor)

        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def decay_epsilon(self) -> None:
        """Reduce la tasa de exploración tras cada episodio."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str) -> None:
        """Guarda los pesos del modelo en disco asegurando que exista el directorio."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.model.state_dict(), filepath)

    def load(self, filepath: str) -> None:
        """Carga los pesos del modelo desde disco."""
        self.model.load_state_dict(torch.load(filepath, weights_only=True))
        self.update_target_network()
