"""
Bucle principal de entrenamiento del agente DQN en el juego Snake.

Reanuda el entrenamiento donde se dejó:
- Carga los pesos del modelo guardado.
- Recupera epsilon, récord y nº de episodios acumulados (training_state.json).
- Guarda checkpoints periódicos y también al interrumpir con Ctrl+C.

Uso:
    python train.py                 # reanuda (o empieza si no hay modelo)
    python train.py --episodes 1000
    python train.py --fresh         # empieza desde cero ignorando lo guardado
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Asegurar que el directorio raíz del proyecto esté en sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402,F401
from ai.agent import DQNAgent  # noqa: E402
from game.settings import Direction  # noqa: E402
from game.snake_game import SnakeGame  # noqa: E402

# Rutas absolutas: no dependen de desde dónde se ejecute el script
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "snake_model.pth"
BEST_MODEL_PATH = MODELS_DIR / "best_snake_model.pth"
STATE_PATH = MODELS_DIR / "training_state.json"

CHECKPOINT_EVERY = 25  # episodios entre checkpoints


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


# ----------------------------------------------------------------------
# Persistencia del estado de entrenamiento
# ----------------------------------------------------------------------
def default_state() -> Dict[str, Any]:
    return {"epsilon": None, "record_score": 0, "episodes_done": 0, "scores_history": []}


def load_state() -> Dict[str, Any]:
    state = default_state()
    if STATE_PATH.exists():
        try:
            state.update(json.loads(STATE_PATH.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError) as e:
            print(f"No se pudo leer {STATE_PATH.name}: {e}. Se usan valores por defecto.")
    return state


def save_checkpoint(agent: DQNAgent, state: Dict[str, Any]) -> None:
    """Guarda pesos + metadatos del entrenamiento."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    agent.save(str(MODEL_PATH))
    state["epsilon"] = agent.epsilon
    state["scores_history"] = state["scores_history"][-50:]
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


# ----------------------------------------------------------------------
# Entrenamiento
# ----------------------------------------------------------------------
def train(episodes: int = 500, fresh: bool = False) -> DQNAgent:
    agent = DQNAgent()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    state_meta = default_state()

    if fresh:
        print("Modo --fresh: se empieza desde cero (se ignora el modelo guardado).")
    elif MODEL_PATH.exists():
        try:
            agent.load(str(MODEL_PATH))
            state_meta = load_state()
            if state_meta["epsilon"] is not None:
                agent.epsilon = state_meta["epsilon"]
            agent.update_target_network()  # sincroniza la red objetivo con la cargada
            print(f"Modelo anterior cargado: {MODEL_PATH}")
            print(
                f"Reanudando: {state_meta['episodes_done']} episodios previos | "
                f"récord {state_meta['record_score']} | epsilon {agent.epsilon:.3f}"
            )
        except RuntimeError as e:
            state_meta = default_state()
            print(f"Modelo anterior incompatible (cambió el tamaño de la red): {e}")
            print("Se empieza desde cero con la nueva arquitectura.")
    else:
        print("No se encontró un modelo anterior. Se empieza desde cero.")

    target_update_frequency = 10
    record_score: int = state_meta["record_score"]
    scores_history: List[int] = list(state_meta["scores_history"])
    state_meta["scores_history"] = scores_history

    print("=" * 65)
    print("INICIANDO ENTRENAMIENTO DQN - SNAKE")
    print(f"Episodios programados en esta sesión: {episodes}")
    print(f"Batch size: {agent.batch_size} | Memoria máxima: {agent.memory_size}")
    print("=" * 65)

    try:
        for episode in range(episodes):
            game = SnakeGame()
            state = game.get_state()
            total_reward = 0.0
            done = False
            score = 0
            episode_losses: List[float] = []

            while not done:
                # 1. El agente elige una acción (0: izquierda, 1: recto, 2: derecha)
                action = agent.get_action(state)

                # 2. Traducir acción relativa a la dirección absoluta
                direction = get_new_direction(action, game.direction)

                # Distancia Manhattan a la comida ANTES de mover
                old_food_distance = (
                    abs(game.head.x - game.food.x) + abs(game.head.y - game.food.y)
                )

                # 3. Avanzar el juego un paso
                old_score = game.score
                done, score = game.step(direction)

                # 4. Recompensa:
                # +10 por comer, -10 por morir,
                # +0.15 si se acerca, -0.15 si se aleja, -0.05 si mantiene distancia
                if done:
                    reward = -10.0
                elif score > old_score:
                    reward = 10.0
                else:
                    new_food_distance = (
                        abs(game.head.x - game.food.x) + abs(game.head.y - game.food.y)
                    )
                    if new_food_distance < old_food_distance:
                        reward = 0.15
                    elif new_food_distance > old_food_distance:
                        reward = -0.15
                    else:
                        reward = -0.05

                # 5. Siguiente estado
                next_state = game.get_state()

                # 6. Guardar la transición en la memoria de experiencias
                agent.remember(state, action, reward, next_state, done)

                # 7. Entrenar la red neuronal con un lote aleatorio
                loss = agent.train_step()
                if loss > 0.0:
                    episode_losses.append(loss)

                state = next_state
                total_reward += reward

            # Reducción de exploración al finalizar cada episodio
            agent.decay_epsilon()

            # Actualizar la red objetivo cada 'target_update_frequency' episodios
            if (episode + 1) % target_update_frequency == 0:
                agent.update_target_network()

            # Contadores acumulados
            state_meta["episodes_done"] += 1
            scores_history.append(score)

            # Guardar récord (solo si de verdad supera el récord histórico)
            if score > record_score:
                record_score = score
                state_meta["record_score"] = record_score
                agent.save(str(BEST_MODEL_PATH))

            # Checkpoint periódico
            if (episode + 1) % CHECKPOINT_EVERY == 0:
                save_checkpoint(agent, state_meta)

            # Métricas de los últimos 50 episodios
            recent = scores_history[-50:]
            avg_score = sum(recent) / len(recent)
            avg_loss = (sum(episode_losses) / len(episode_losses)) if episode_losses else 0.0

            print(
                f"Episodio: {state_meta['episodes_done']:5d} "
                f"({episode + 1:4d}/{episodes}) | "
                f"Score: {score:2d} | "
                f"Récord: {record_score:2d} | "
                f"Media(50): {avg_score:4.1f} | "
                f"Recompensa: {total_reward:6.1f} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Loss: {avg_loss:.4f}"
            )

    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido por el usuario. Guardando progreso...")

    finally:
        # Se guarda siempre: fin normal, Ctrl+C o error
        save_checkpoint(agent, state_meta)

    print("=" * 65)
    print("Entrenamiento finalizado.")
    print(f"Modelo guardado en: {MODEL_PATH}")
    print(f"Mejor modelo en:    {BEST_MODEL_PATH}")
    print(f"Estado guardado en: {STATE_PATH}")
    print("=" * 65)

    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento DQN para Snake")
    parser.add_argument("--episodes", type=int, default=500, help="Episodios de esta sesión")
    parser.add_argument("--fresh", action="store_true", help="Empezar desde cero")
    args = parser.parse_args()

    train(episodes=args.episodes, fresh=args.fresh)