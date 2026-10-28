"""
Memoria de repetición de experiencias (Replay Buffer) para DQN.

Almacena transiciones (estado, acción, recompensa, siguiente_estado, fin)
para permitir el muestreo aleatorio en minilotes, rompiendo la correlación
temporal entre pasos sucesivos.
"""

import random
from collections import deque
from typing import Any, List, Tuple


class ReplayMemory:
    """Buffer circular con capacidad máxima fija."""

    def __init__(self, capacity: int = 100_000):
        self.memory = deque(maxlen=capacity)

    def push(
        self,
        state: List[int],
        action: int,
        reward: float,
        next_state: List[int],
        done: bool,
    ) -> None:
        """Guarda una experiencia en la memoria."""
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Tuple]:
        """Devuelve un lote aleatorio de experiencias."""
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        """Cantidad actual de experiencias almacenadas."""
        return len(self.memory)

