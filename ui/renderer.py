"""
Módulo de renderizado gráfico del juego utilizando Pygame.

Se encarga exclusivamente de:
- Dibujar el tablero de 20x20 casillas y las líneas de la cuadrícula.
- Dibujar la serpiente con diferenciación clara de la cabeza y el cuerpo.
- Mostrar el marcador superior (pasos, longitud, estado).
- Capturar eventos del teclado (flechas / WASD) para pruebas manuales.
"""

from typing import Optional, Tuple
import pygame
from game.settings import (
    CELL_SIZE,
    COLOR_BG,
    COLOR_FOOD,
    COLOR_GAME_OVER,
    COLOR_GRID,
    COLOR_HEADER,
    COLOR_SNAKE_BODY,
    COLOR_SNAKE_EYE,
    COLOR_SNAKE_HEAD,
    COLOR_TEXT,
    COLOR_TEXT_MUTED,
    Direction,
    GRID_SIZE,
    HEADER_HEIGHT,
    Point,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)
from game.snake_game import SnakeGame


class PygameRenderer:
    """Gestiona la ventana y la representación gráfica del juego."""

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Snake AI - Entorno de Refuerzo")
        self.clock = pygame.time.Clock()

        # Fuentes tipográficas
        self.font = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 18, bold=True)
        self.font_small = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 14)
        self.font_large = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 28, bold=True)

    def draw(self, game: SnakeGame) -> None:
        """Dibuja el fotograma completo: fondo, cuadrícula, encabezado y serpiente."""
        # 1. Fondo de la zona de juego
        self.screen.fill(COLOR_BG)

        # 2. Dibujar encabezado superior (barra de estado)
        self._draw_header(game)

        # 3. Dibujar líneas de la cuadrícula del tablero
        self._draw_grid()

        # 4. Dibujar serpiente
        self._draw_snake(game)

        self._draw_food(game)

        # 5. Si ha terminado el juego, mostrar banner de Game Over
        if game.is_game_over:
            self._draw_game_over()

        pygame.display.flip()

    def _draw_header(self, game: SnakeGame) -> None:
        """Dibuja la barra superior con información del juego."""
        header_rect = pygame.Rect(0, 0, WINDOW_WIDTH, HEADER_HEIGHT)
        pygame.draw.rect(self.screen, COLOR_HEADER, header_rect)
        pygame.draw.line(
            self.screen, COLOR_GRID, (0, HEADER_HEIGHT), (WINDOW_WIDTH, HEADER_HEIGHT), 2
        )

        title_surf = self.font.render("SNAKE AI", True, COLOR_TEXT)
        self.screen.blit(title_surf, (20, 18))

        steps_surf = self.font_small.render(f"Pasos: {game.steps}", True, COLOR_TEXT_MUTED)
        self.screen.blit(steps_surf, (220, 22))

        length_surf = self.font_small.render(f"Longitud: {len(game.snake)}", True, COLOR_TEXT_MUTED)
        self.screen.blit(length_surf, (340, 22))

        controls_surf = self.font_small.render("Flechas: Mover | R: Reiniciar", True, COLOR_TEXT_MUTED)
        self.screen.blit(controls_surf, (WINDOW_WIDTH - controls_surf.get_width() - 20, 22))

    def _draw_grid(self) -> None:
        """Dibuja la cuadrícula de 20x20 casillas."""
        for x in range(0, WINDOW_WIDTH, CELL_SIZE):
            pygame.draw.line(
                self.screen,
                COLOR_GRID,
                (x, HEADER_HEIGHT),
                (x, WINDOW_HEIGHT),
                1,
            )
        for y in range(HEADER_HEIGHT, WINDOW_HEIGHT, CELL_SIZE):
            pygame.draw.line(
                self.screen,
                COLOR_GRID,
                (0, y),
                (WINDOW_WIDTH, y),
                1,
            )

    def _draw_food(self, game: SnakeGame) -> None:
        """Dibuja la comida en el tablero."""
        food = game.food

        rect = pygame.Rect(
            food.x * CELL_SIZE + 3,
            HEADER_HEIGHT + food.y * CELL_SIZE + 3,
            CELL_SIZE - 6,
            CELL_SIZE - 6,
        )

        pygame.draw.rect(
            self.screen,
            COLOR_FOOD,
            rect,
            border_radius=6,
        )

    def _draw_snake(self, game: SnakeGame) -> None:
        """Dibuja los segmentos de la serpiente."""
        # 1. Dibujar segmentos del cuerpo
        for segment in game.body:
            rect = pygame.Rect(
                segment.x * CELL_SIZE + 2,
                HEADER_HEIGHT + segment.y * CELL_SIZE + 2,
                CELL_SIZE - 4,
                CELL_SIZE - 4,
            )
            pygame.draw.rect(self.screen, COLOR_SNAKE_BODY, rect, border_radius=4)

        # 2. Dibujar cabeza con color diferenciado
        head = game.head
        head_rect = pygame.Rect(
            head.x * CELL_SIZE + 2,
            HEADER_HEIGHT + head.y * CELL_SIZE + 2,
            CELL_SIZE - 4,
            CELL_SIZE - 4,
        )
        pygame.draw.rect(self.screen, COLOR_SNAKE_HEAD, head_rect, border_radius=6)

        # 3. Dibujar pequeños ojos según la dirección para orientar visualmente
        self._draw_eyes(head, game.direction)

    def _draw_eyes(self, head: Point, direction: Direction) -> None:
        """Dibuja dos ojos en la cabeza de la serpiente según hacia dónde mire."""
        base_x = head.x * CELL_SIZE
        base_y = HEADER_HEIGHT + head.y * CELL_SIZE
        eye_radius = 2

        offsets = {
            Direction.RIGHT: [(20, 9), (20, 21)],
            Direction.LEFT: [(9, 9), (9, 21)],
            Direction.UP: [(9, 9), (21, 9)],
            Direction.DOWN: [(9, 20), (21, 20)],
        }

        for ox, oy in offsets.get(direction, []):
            pygame.draw.circle(
                self.screen,
                COLOR_SNAKE_EYE,
                (base_x + ox, base_y + oy),
                eye_radius,
            )

    def _draw_game_over(self) -> None:
        """Muestra un mensaje visual centrado cuando la serpiente colisiona."""
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT - HEADER_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))  # Fondo negro translúcido
        self.screen.blit(overlay, (0, HEADER_HEIGHT))

        center_y = HEADER_HEIGHT + (WINDOW_HEIGHT - HEADER_HEIGHT) // 2

        text_game_over = self.font_large.render("¡COLISIÓN!", True, COLOR_GAME_OVER)
        rect_go = text_game_over.get_rect(center=(WINDOW_WIDTH // 2, center_y - 20))
        self.screen.blit(text_game_over, rect_go)

        text_restart = self.font.render(
            "Pulsa [R] para reiniciar o [Q] para salir", True, COLOR_TEXT
        )
        rect_re = text_restart.get_rect(center=(WINDOW_WIDTH // 2, center_y + 25))
        self.screen.blit(text_restart, rect_re)

    def process_input(self) -> Tuple[bool, bool, Optional[Direction]]:
        """
        Lee los eventos de Pygame.
        Devuelve:
            (should_quit, should_restart, new_direction)
        """
        should_quit = False
        should_restart = False
        new_direction: Optional[Direction] = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                should_quit = True

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    should_quit = True
                elif event.key == pygame.K_r:
                    should_restart = True
                elif event.key in (pygame.K_UP, pygame.K_w):
                    new_direction = Direction.UP
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    new_direction = Direction.DOWN
                elif event.key in (pygame.K_LEFT, pygame.K_a):
                    new_direction = Direction.LEFT
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    new_direction = Direction.RIGHT

        return should_quit, should_restart, new_direction

    def tick(self, fps: int) -> None:
        """Controla la tasa de refresco por segundo."""
        self.clock.tick(fps)

    def close(self) -> None:
        """Cierra Pygame de forma segura."""
        pygame.quit()

