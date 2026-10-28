"""
Configuraciones y constantes globales del juego Snake y la interfaz visual.
"""

from enum import Enum
from typing import NamedTuple


class Point(NamedTuple):
    """Representa una coordenada (x, y) en la cuadrícula del tablero."""
    x: int
    y: int


class Direction(Enum):
    """Direcciones de movimiento de la serpiente."""
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

    @property
    def opposite(self) -> "Direction":
        """Devuelve la dirección opuesta para evitar giros de 180 grados inmediatos."""
        opposites = {
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP,
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT,
        }
        return opposites[self]


# --- Dimensiones del tablero y la ventana ---
GRID_SIZE = 20           # Tablero de 20x20 casillas
CELL_SIZE = 30           # Cada casilla mide 30x30 píxeles
HEADER_HEIGHT = 60       # Altura reservada en la parte superior para marcadores y estado

WINDOW_WIDTH = GRID_SIZE * CELL_SIZE                    # 600 px
WINDOW_HEIGHT = GRID_SIZE * CELL_SIZE + HEADER_HEIGHT   # 660 px

# --- Parámetros de juego ---
FPS = 10                 # Fotogramas por segundo (velocidad inicial adecuada para humanos)
INITIAL_SNAKE_LENGTH = 3 # Longitud inicial de la serpiente (3 segmentos)

# --- Paleta de colores (tema oscuro moderno) ---
COLOR_BG = (18, 22, 28)            # Fondo del tablero (gris azulado oscuro)
COLOR_GRID = (28, 34, 44)          # Líneas de la cuadrícula
COLOR_HEADER = (13, 16, 20)        # Fondo del encabezado superior

COLOR_SNAKE_HEAD = (46, 204, 113)  # Cabeza de la serpiente (verde esmeralda brillante)
COLOR_SNAKE_BODY = (39, 174, 96)   # Cuerpo de la serpiente (verde medio)
COLOR_SNAKE_EYE = (15, 30, 20)     # Ojos de la serpiente (detalle visual)

COLOR_FOOD = (231, 76, 60)         # Comida (rojo carmesí - para próximas fases)
COLOR_TEXT = (236, 240, 241)        # Texto principal (blanco suave)
COLOR_TEXT_MUTED = (149, 165, 166)  # Texto secundario (gris claro)
COLOR_GAME_OVER = (231, 76, 60)    # Alerta de Game Over

