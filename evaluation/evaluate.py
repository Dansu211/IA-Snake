"""
Módulo de evaluación visual del agente entrenado.

Permite cargar los pesos del modelo guardado y observar a la IA jugando
en tiempo real con la interfaz gráfica de Pygame.
"""

import sys
from pathlib import Path

# Asegurar acceso a los módulos del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from ai.agent import DQNAgent
from game.settings import FPS
from game.snake_game import SnakeGame
from training.train import get_new_direction
from ui.renderer import PygameRenderer


def evaluate(
    model_path: str = "models/best_snake_model.pth",
    games: int = 10,
    fps: int = 15,
) -> None:
    """
    Ejecuta partidas visuales del agente ya entrenado:
    - Epsilon = 0.0 (modo pura explotación: la IA siempre elige su mejor opción).
    - Renderiza cada paso con Pygame a velocidad observable.
    """
    agent = DQNAgent()

    # Si no existe el modelo 'best', intentar cargar el general
    if not Path(model_path).exists():
        fallback_path = "models/snake_model.pth"
        if Path(fallback_path).exists():
            model_path = fallback_path
        else:
            print(f"Error: No se encontró ningún modelo guardado en '{model_path}'.")
            print("Entrena primero al agente ejecutando 'python training/train.py'.")
            return

    agent.load(model_path)
    agent.epsilon = 0.0  # Sin exploración aleatoria
    print(f"Modelo cargado con éxito desde '{model_path}'.")
    print(f"Iniciando {games} partidas de demostración visual (FPS: {fps})...\n")

    renderer = PygameRenderer()
    scores = []

    try:
        for game_num in range(1, games + 1):
            game = SnakeGame()
            done = False

            while not done:
                # 1. Procesar eventos (permitir salir con ESC / cerrar ventana)
                should_quit, _, _ = renderer.process_input()
                if should_quit:
                    print("Demostración interrumpida por el usuario.")
                    return

                # 2. La IA observa el estado y elige la mejor acción
                state = game.get_state()
                action = agent.get_action(state)

                # 3. Mover la serpiente
                direction = get_new_direction(action, game.direction)
                done, score = game.step(direction)

                # 4. Dibujar el fotograma
                renderer.draw(game)
                renderer.tick(fps)

            scores.append(game.score)
            print(f"Partida {game_num:2d}/{games} | Puntuación final: {game.score} | Pasos: {game.steps}")

    finally:
        renderer.close()

    if scores:
        print("\n" + "=" * 50)
        print(f"RESUMEN DE EVALUACIÓN ({len(scores)} partidas):")
        print(f"  Puntuación máxima: {max(scores)}")
        print(f"  Puntuación media:  {sum(scores) / len(scores):.2f}")
        print("=" * 50)


if __name__ == "__main__":
    evaluate()

