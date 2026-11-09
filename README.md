# 🧠 Snake AI con Aprendizaje por Refuerzo (DQN)

Proyecto de desarrollo de un agente inteligente capaz de aprender a jugar al Snake mediante **Deep Q-Network (DQN)**, utilizando redes neuronales profundas con PyTorch.

El objetivo es desarrollar una IA desde cero, comprender el aprendizaje por refuerzo y mejorar progresivamente su capacidad para tomar decisiones, evitar obstáculos y conseguir comida.

---

## 📁 Estructura del proyecto

```text
IAsnake/
├── main.py                 # Punto de entrada principal
├── requirements.txt        # Dependencias del proyecto
├── README.md               # Documentación
│
├── game/                   # Entorno y lógica del juego
│   ├── __init__.py
│   ├── snake_game.py       # Movimientos, estado y colisiones
│   └── settings.py         # Configuración del tablero
│
├── ui/                     # Interfaz gráfica
│   ├── __init__.py
│   └── renderer.py         # Renderizado mediante Pygame
│
├── ai/                     # Agente inteligente
│   ├── __init__.py
│   ├── agent.py            # Política de decisión y aprendizaje DQN
│   ├── model.py            # Red neuronal Q-Network
│   └── replay_memory.py    # Memoria de experiencias (si se utiliza)
│
├── training/               # Entrenamiento
│   ├── __init__.py
│   └── train.py            # Bucle de entrenamiento
│
├── evaluation/             # Evaluación del agente
│   ├── __init__.py
│   └── evaluate.py         # Pruebas visuales del modelo
│
├── models/                 # Modelos entrenados (.pth)
│
└── results/                # Estadísticas y gráficas
```

---

## 🛠️ Tecnologías utilizadas

- **Python 3.10+**: Lenguaje principal.
- **PyTorch**: Construcción y entrenamiento de redes neuronales.
- **Pygame (pygame-ce)**: Entorno gráfico del juego.
- **NumPy**: Manipulación de vectores y datos numéricos.
- **Matplotlib**: Visualización de resultados y evolución del aprendizaje.

---

## 🚀 Estado actual del proyecto

El proyecto ya dispone de un entorno funcional y un agente DQN capaz de aprender mediante entrenamiento.

### Estado de percepción de la serpiente

Actualmente, la red neuronal recibe un vector de **20 características** en cada movimiento.

| Posiciones | Característica | Información |
|---|---|---|
| 0-2 | Peligro izquierda, recto y derecha | Detecta posibles colisiones |
| 3-5 | Distancia a obstáculos | Mide el espacio libre en cada dirección |
| 6-9 | Dirección actual | Indica hacia dónde se mueve |
| 10-13 | Dirección de la comida | Indica la posición relativa de la comida |
| 14-15 | `food_dx`, `food_dy` | Distancia relativa horizontal y vertical a la comida |
| 16-19 | Distancia a paredes | Información sobre la proximidad a los cuatro muros |

Esta ampliación permite que la serpiente disponga de más información espacial para anticipar obstáculos y planificar sus movimientos.

### Sistema de recompensas

El agente aprende mediante un sistema de recompensas que incentiva los comportamientos adecuados:

| Situación | Recompensa |
|---|---:|
| Comer comida | +10 |
| Morir por colisión | -10 |
| Acercarse a la comida | +0.15 |
| Alejarse de la comida | -0.15 |
| Movimiento sin progreso | -0.05 |

Las recompensas por distancia ayudan a orientar al agente hacia la comida, mientras que las penalizaciones desincentivan las colisiones y los movimientos poco útiles.

---

## 🧠 Cómo funciona el aprendizaje DQN

El agente utiliza Deep Q-Learning para aprender qué acción ofrece mejores resultados en cada estado.

Dispone de tres acciones:

- 0: girar a la izquierda.
- 1: continuar recto.
- 2: girar a la derecha.

El proceso de aprendizaje es:

1. El juego proporciona el estado actual de 20 características.
2. La red neuronal estima el valor Q de las tres acciones.
3. El agente selecciona una acción mediante epsilon-greedy.
4. El juego ejecuta el movimiento y devuelve una recompensa.
5. La experiencia se almacena en la memoria.
6. La red neuronal se entrena utilizando experiencias anteriores.
7. La red objetivo se actualiza periódicamente.

Con el entrenamiento, el agente debería aprender progresivamente a conseguir más comida y sobrevivir durante más tiempo.

---

## 🎮 Cómo ejecutar el proyecto

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Ejecutar el juego manualmente

```bash
.venv\Scripts\python.exe main.py
```

**Controles:**

- Flechas / WASD: cambiar de dirección.
- R: reiniciar después de una colisión.
- Q / ESC: salir del juego.

### 3. Entrenar a la IA

El entrenamiento se ejecuta sin gráficos para acelerar el aprendizaje.

```bash
.venv\Scripts\python.exe main.py --mode train --episodes 300
```

Ejemplo con más episodios:

```bash
.venv\Scripts\python.exe main.py --mode train --episodes 1000
```

### 4. Observar a la IA jugar

Para cargar el modelo entrenado y observar su comportamiento:

```bash
.venv\Scripts\python.exe main.py --mode eval
```

---

## 💾 Modelos y resultados

Los modelos entrenados se almacenan en formato `.pth`, que permite guardar los parámetros de la red neuronal.

La carpeta `models/` está destinada a almacenar los modelos generados durante el entrenamiento.

La carpeta `results/` está destinada a guardar estadísticas y gráficas para analizar la evolución del agente.

**Importante:** al cambiar la arquitectura de la red neuronal de 11 a 20 entradas, los modelos anteriores dejan de ser compatibles con la nueva arquitectura.

---

## 🎯 Próximas mejoras

- [ ] Guardar y recuperar el estado completo del entrenamiento.
- [ ] Conservar automáticamente el mejor modelo.
- [ ] Añadir gráficas de puntuación media y recompensas.
- [ ] Reducir las colisiones contra el propio cuerpo.
- [ ] Mejorar la estrategia de supervivencia.
- [ ] Experimentar con diferentes sistemas de recompensas.
- [ ] Implementar Double DQN para mejorar la estimación de valores Q.
- [ ] Comparar el rendimiento con diferentes configuraciones de entrenamiento.

---

## 📌 Objetivo final

Conseguir una serpiente autónoma capaz de aprender por sí misma, tomar decisiones eficientes, evitar colisiones y maximizar la puntuación mediante aprendizaje por refuerzo.

El proyecto se desarrolla de forma incremental, priorizando la comprensión de los algoritmos y la implementación de cada componente.
