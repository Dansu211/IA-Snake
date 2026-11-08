# Snake AI con Aprendizaje por Refuerzo (DQN)

Proyecto paso a paso para desarrollar un agente inteligente que aprenda a jugar al Snake utilizando una red neuronal profunda (*Deep Q-Network*) en PyTorch.

---

## 📁 Estructura del Proyecto

```text
snake-ai/
├── main.py                 # Punto de entrada para probar y ejecutar el entorno
├── requirements.txt        # Dependencias del proyecto
├── README.md               # Documentación y guía de desarrollo
├── game/                   # Lógica central del juego (sin dependencias gráficas)
│   ├── __init__.py
│   ├── snake_game.py       # Estado del juego, movimientos y colisiones
│   └── settings.py         # Constantes de configuración, dimensiones y colores
├── ui/                     # Interfaz gráfica desacoplada (Pygame)
│   ├── __init__.py
│   └── renderer.py         # Renderizado del tablero, cuadrícula y serpiente
├── ai/                     # Componentes del Agente Inteligente (Próxima etapa)
│   ├── __init__.py
│   ├── agent.py            # Toma de decisiones y política epsilon-greedy
│   ├── model.py            # Red Neuronal (Q-Network) en PyTorch
│   └── replay_memory.py    # Buffer de repetición de experiencias
├── training/               # Lógica de entrenamiento
│   ├── __init__.py
│   └── train.py            # Bucle de entrenamiento y registro de métricas
├── evaluation/             # Evaluación y pruebas del modelo
│   ├── __init__.py
│   └── evaluate.py         # Test visual del agente ya entrenado
├── models/                 # Almacenamiento de pesos del modelo (.pth)
└── results/                # Gráficas de aprendizaje y estadísticas
```

---

## 🛠️ Tecnologías

- **Python 3.10+**
- **Pygame (pygame-ce)**: Renderizado gráfico e interactivo.
- **PyTorch**: Construcción y entrenamiento de la red neuronal.
- **NumPy**: Manipulación de vectores de estado y tensores.
- **Matplotlib**: Visualización del progreso del aprendizaje.

---

## 🚀 Estado Actual: Fase 1 (Entorno Base)

- [x] Arquitectura y separación limpia de módulos.
- [x] Lógica de tablero de 20 × 20 casillas.
- [x] Serpiente inicial de 3 segmentos con movimiento discreto.
- [x] Detección de colisiones contra los muros y contra sí misma.
- [x] Renderizador con Pygame (cuadrícula, serpiente orientada y HUD de estado).
- [ ] *Próximo paso:* Generación de comida y vector de recompensas (+10, -10, -0.1).

---

## 🎮 Cómo Ejecutar

1. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

2. Ejecutar la prueba interactiva del tablero y la serpiente:
   ```bash
   python main.py
   ```

**Controles:**
- **Flechas / WASD:** Cambiar de dirección.
- **R:** Reiniciar tras una colisión.
- **Q / ESC:** Salir del juego.

1. Entrenar a la IA (Modo rápido sin gráficos)
powershell
python.exe main.py --mode train --episodes 300

.venv\Scripts\python.exe main.py --mode train --episodes 500

2. Observar a la IA jugar (Modo visual con Pygame)

.venv\Scripts\python.exe main.py --mode eval