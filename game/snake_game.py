"""
Lógica central del juego Snake.

Este módulo encapsula el estado del juego (la posición de la serpiente,
comida, colisiones y avance) sin depender de Pygame ni de ninguna biblioteca visual.
Esto permite entrenar la IA a máxima velocidad en modo "headless" (sin gráficos)
o conectarlo a un renderizador visual.
"""

import random
from typing import List, Optional, Tuple
from game.settings import Direction, Point, GRID_SIZE, INITIAL_SNAKE_LENGTH


class SnakeGame:
    """Gestiona el estado y las reglas del juego Snake."""

    def __init__(self, grid_size: int = GRID_SIZE):
        self.grid_size = grid_size
        self.snake: List[Point] = []
        self.direction: Direction = Direction.RIGHT
        self.score: int = 0
        self.steps: int = 0
        self.steps_without_food: int = 0
        self.is_game_over: bool = False
        self.food: Point = Point(0, 0)

        self.reset()

    def reset(self) -> None:
        """
        Reinicia el estado del juego:
        - Serpiente colocada en el centro del tablero con 3 segmentos horizontales.
        - Dirección inicial hacia la derecha.
        - Generación aleatoria de comida.
        - Marcadores en 0.
        """
        center_x = self.grid_size // 2
        center_y = self.grid_size // 2

        self.direction = Direction.RIGHT
        self.snake = [
            Point(center_x - i, center_y) for i in range(INITIAL_SNAKE_LENGTH)
        ]

        self.score = 0
        self.steps = 0
        self.steps_without_food = 0
        self.is_game_over = False
        self.food = self._generate_food()

    def _generate_food(self) -> Point:
        """Genera comida en una casilla libre del tablero que no ocupe la serpiente."""
        while True:
            x = random.randrange(self.grid_size)
            y = random.randrange(self.grid_size)
            food = Point(x, y)

            if food not in self.snake:
                return food

    @property
    def head(self) -> Point:
        """Devuelve las coordenadas de la cabeza de la serpiente."""
        return self.snake[0]

    @property
    def body(self) -> List[Point]:
        """Devuelve la lista de segmentos del cuerpo (excluyendo la cabeza)."""
        return self.snake[1:]

    def change_direction(self, new_direction: Direction) -> bool:
        """
        Cambia la dirección de avance si no es contraria a la actual.
        No se permite girar 180° directamente (ej. de DERECHA a IZQUIERDA).
        """
        if new_direction != self.direction.opposite:
            self.direction = new_direction
            return True
        return False

    def is_collision(self, point: Optional[Point] = None) -> bool:
        """
        Comprueba si un punto dado colisiona con los muros o con el propio cuerpo.
        Si point es None, comprueba la cabeza actual de la serpiente.
        """
        if point is None:
            point = self.head

        # 1. Colisión con los límites del tablero (muros)
        if (point.x < 0 or point.x >= self.grid_size or
                point.y < 0 or point.y >= self.grid_size):
            return True

        # 2. Colisión con el cuerpo:
        # Si comprobamos la cabeza actual, se evalúa contra el resto del cuerpo (self.snake[1:]).
        # Si comprobamos un punto futuro, se evalúa contra todos menos el último segmento que se moverá.
        if point == self.head:
            return point in self.snake[1:]
        return point in self.snake[:-1]

    def step(self, new_direction: Optional[Direction] = None) -> Tuple[bool, int]:
        """
        Avanza el juego un paso temporal:
        1. Cambia de dirección si se especifica una nueva válida.
        2. Calcula la nueva posición de la cabeza.
        3. Comprueba colisiones con muros o consigo misma.
        4. Si no hay colisión, desplaza la serpiente y comprueba comida.
        5. Aplica límite de pasos sin comer para evitar bucles infinitos.

        Devuelve:
            Tuple[bool, int]: (is_game_over, score)
        """
        if self.is_game_over:
            return self.is_game_over, self.score

        if new_direction is not None:
            self.change_direction(new_direction)

        self.steps += 1

        # 1. Calcular nueva posición de la cabeza
        dx, dy = self.direction.value
        new_head = Point(self.head.x + dx, self.head.y + dy)

        # 2. Comprobar colisión
        if self.is_collision(new_head):
            self.is_game_over = True
            return self.is_game_over, self.score

        # 3. Mover la serpiente: insertar nueva cabeza
        self.snake.insert(0, new_head)

        # 4. Comprobar si la serpiente ha comido
        if new_head == self.food:
            self.score += 1
            self.steps_without_food = 0
            self.food = self._generate_food()
        else:
            self.snake.pop()
            self.steps_without_food += 1

        # 5. Control de bucle infinito (inanición):
        # Si da demasiados pasos sin comer (100 * longitud), finaliza la partida
        if self.steps_without_food > 100 * len(self.snake):
            self.is_game_over = True

        return self.is_game_over, self.score

    def get_state(self) -> List[int]:
        """
        Convierte la situación actual en un vector de 11 características para la IA:
        - [0..2]  Peligro inmediato si toma: [Giro Izquierda, Seguir Recto, Giro Derecha]
        - [3..6]  Dirección actual de movimiento: [Derecha, Abajo, Izquierda, Arriba]
        - [7..10] Posición relativa de la comida: [Comida Izq, Comida Der, Comida Arriba, Comida Abajo]
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

        def point_in_direction(direction: Direction) -> Point:
            dx, dy = direction.value
            return Point(self.head.x + dx, self.head.y + dy)

        # 1. Peligros alineados exactamente con las acciones relativas:
        # Acción 0 (Girar Izquierda) -> danger_left
        # Acción 1 (Seguir Recto)     -> danger_straight
        # Acción 2 (Girar Derecha)   -> danger_right
        danger_left = self.is_collision(point_in_direction(left_turn[self.direction]))
        danger_straight = self.is_collision(point_in_direction(self.direction))
        danger_right = self.is_collision(point_in_direction(right_turn[self.direction]))

        # 2. Dirección actual
        dir_right = self.direction == Direction.RIGHT
        dir_down = self.direction == Direction.DOWN
        dir_left = self.direction == Direction.LEFT
        dir_up = self.direction == Direction.UP

        # 3. Posición relativa de la comida respecto a la cabeza
        food_left = self.food.x < self.head.x
        food_right = self.food.x > self.head.x
        food_up = self.food.y < self.head.y
        food_down = self.food.y > self.head.y

        return [
            int(danger_left),
            int(danger_straight),
            int(danger_right),
            int(dir_right),
            int(dir_down),
            int(dir_left),
            int(dir_up),
            int(food_left),
            int(food_right),
            int(food_up),
            int(food_down),
        ]
