"""
Bucle principal de entrenamiento del agente DQN en el juego Snake.
"""

import os
import sys
from pathlib import Path
from typing import List

# Asegurar que el directorio raíz del proyecto esté en sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from ai.agent import DQNAgent
from game.settings import Direction
from game.snake_game import SnakeGame


def get_new_direction(action: int, current_direction: Direction) -> Direction:
    """
    Convierte la acción relativa en una dirección absoluta del juego:
    - Acción 0: Girar a la izquierda (relativo a la cabeza)
    - Acción 1: Seguir recto
    - Acción 2: Girar a la derecha (relativo a la cabeza)
    """
    left_turn = {
        Direction.RIGHT: Direction.UP,
        Direction.UP: Direction.LEFT,
        Direction.LEFT: Direction.DOWN,
        Direction.DOWN: Direction.RIGHT,
    }

    right_turn = {
        Direction.RIGHT: Direction.DOWN,
        Direction.DOWN: Direction.LEFT,
        Direction.LEFT: Direction.UP,
        Direction.UP: Direction.RIGHT,
    }

    if action == 0:
        return left_turn[current_direction]
    elif action == 1:
        return current_direction
    else:
        return right_turn[current_direction]


def train(episodes: int = 500) -> DQNAgent:
    """
    Ejecuta el bucle de entrenamiento por refuerzo:
    1. Obtiene el estado s_t (vector de 11 características).
    2. El agente elige una acción a_t (0: izq, 1: recto, 2: der) con epsilon-greedy.
    3. El entorno ejecuta la acción y devuelve la recompensa r_t y el nuevo estado s_{t+1}.
    4. Guarda la experiencia en la memoria de repetición.
    5. Entrena la red con un lote aleatorio (DQN replay).
    6. Actualiza periódicamente la red objetivo (Target Network).
    """
    agent = DQNAgent()

    target_update_frequency = 10
    record_score = 0
    scores_history: List[int] = []

    print("=" * 65)
    print("INICIANDO ENTRENAMIENTO DQN - SNAKE")
    print(f"Episodios programados: {episodes}")
    print(f"Batch size: {agent.batch_size} | Memoria máxima: {agent.memory_size}")
    print("=" * 65)

    for episode in range(episodes):
        game = SnakeGame()
        state = game.get_state()
        total_reward = 0.0
        done = False
        episode_losses: List[float] = []

        while not done:
            # 1. El agente elige una acción (0: izquierda, 1: recto, 2: derecha)
            action = agent.get_action(state)

            # 2. Traducir acción relativa a la dirección absoluta
            direction = get_new_direction(action, game.direction)

            # 3. Avanzar el juego un paso
            old_score = game.score
            done, score = game.step(direction)

            # 4. Asignar recompensa según las especificaciones:
            # +10 por comer, -10 por morir, -0.1 por movimiento normal
            if done:
                reward = -10.0
            elif score > old_score:
                reward = 10.0
            else:
                reward = -0.1

            # 5. Obtener el siguiente estado
            next_state = game.get_state()

            # 6. Guardar la transición en la memoria de experiencias
            agent.remember(state, action, reward, next_state, done)

            # 7. Entrenar la red neuronal con un lote aleatorio
            loss = agent.train_step()
            if loss > 0.0:
                episode_losses.append(loss)

            state = next_state
            total_reward += reward

        # Reducción de exploración (epsilon decay) al finalizar cada episodio
        agent.decay_epsilon()

        # Actualizar la red objetivo cada 'target_update_frequency' episodios
        if (episode + 1) % target_update_frequency == 0:
            agent.update_target_network()

        # Guardar récord
        scores_history.append(score)
        if score > record_score:
            record_score = score
            agent.save("models/best_snake_model.pth")

        # Métricas de los últimos 50 episodios
        avg_score = sum(scores_history[-50:]) / len(scores_history[-50:])
        avg_loss = (sum(episode_losses) / len(episode_losses)) if episode_losses else 0.0

        print(
            f"Episodio: {episode + 1:4d}/{episodes} | "
            f"Score: {score:2d} | "
            f"Récord: {record_score:2d} | "
            f"Media(50): {avg_score:4.1f} | "
            f"Recompensa: {total_reward:6.1f} | "
            f"Epsilon: {agent.epsilon:.3f} | "
            f"Loss: {avg_loss:.4f}"
        )

    # Guardar modelo final entrenado
    agent.save("models/snake_model.pth")
    print("=" * 65)
    print("Entrenamiento completado.")
    print("Modelos guardados en 'models/snake_model.pth' y 'models/best_snake_model.pth'.")
    print("=" * 65)

    return agent


if __name__ == "__main__":
    train(episodes=500)
