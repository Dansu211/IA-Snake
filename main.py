"""
Punto de entrada principal del proyecto Snake AI.

Permite:
1. Jugar manualmente como humano (probar controles y físicas).
2. Entrenar a la IA a máxima velocidad en modo headless.
3. Observar a la IA jugar en modo visual con Pygame.
"""

import argparse
import sys
from game.settings import FPS
from game.snake_game import SnakeGame
from ui.renderer import PygameRenderer


def run_manual_game() -> None:
    """Bucle principal para jugar manualmente con teclado."""
    print("=" * 60)
    print("SNAKE - MODO MANUAL (HUMANO)")
    print("=" * 60)
    print("Controles:")
    print("  - Flechas / WASD : Cambiar dirección de la serpiente")
    print("  - R              : Reiniciar partida tras colisión")
    print("  - Q / ESC        : Salir del juego")
    print("=" * 60)

    game = SnakeGame()
    renderer = PygameRenderer()

    running = True
    next_direction = None

    try:
        while running:
            # 1. Procesar eventos del teclado
            should_quit, should_restart, requested_direction = renderer.process_input()

            if should_quit:
                running = False
                break

            if should_restart:
                game.reset()
                next_direction = None

            if requested_direction is not None:
                next_direction = requested_direction

            # 2. Si la partida está activa, avanzar paso
            if not game.is_game_over:
                game.step(next_direction)
                next_direction = None

            # 3. Dibujar fotograma
            renderer.draw(game)

            # 4. Controlar velocidad de refresco
            renderer.tick(FPS)

    except KeyboardInterrupt:
        print("\nJuego interrumpido por el usuario.")
    finally:
        renderer.close()
        print("Ventana cerrada.")


def main():
    parser = argparse.ArgumentParser(description="Snake AI - Reinforcement Learning")
    parser.add_argument(
        "--mode",
        choices=["play", "train", "eval"],
        default="play",
        help="Modo de ejecución: 'play' (manual), 'train' (entrenar IA), 'eval' (ver IA jugar)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help="Número de episodios para el modo entrenamiento (por defecto: 500)",
    )

    args = parser.parse_args()

    if args.mode == "play":
        run_manual_game()
    elif args.mode == "train":
        from training.train import train
        train(episodes=args.episodes)
    elif args.mode == "eval":
        from evaluation.evaluate import evaluate
        evaluate()


if __name__ == "__main__":
    main()
