# Autonomous Flappy Bird AI with Neuroevolution & Genetic Algorithms

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Pygame](https://img.shields.io/badge/Pygame-2.x-brightgreen.svg)](https://www.pygame.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.19%2B-orange.svg)](https://numpy.org/)
[![Algorithm](https://img.shields.io/badge/Algorithm-Neuroevolution%20%2B%20GA-blueviolet.svg)](genetic_algorithm.py)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL%203.0-lightgrey.svg)](LICENSE)

An artificial life simulation where autonomous agents evolve neural network brain topologies from scratch using **Genetic Algorithms** in **Pygame**. Without relying on external deep learning frameworks, the agents autonomously learn flight mechanics, pipe gap navigation, and obstacle avoidance through natural selection, single-point crossover, and continuous adaptive mutation.

---

## 📸 Simulation & Gameplay Overview

```
┌────────────────────────────────────────────────────────┐
│  Generation: 14       Distance(best): 18420            │
│  Best gen: 12         Distance(current): 4230          │
│  Deaths: 182 / 200    FPS: 180 (Visual Mode)           │
│                                                        │
│           │   │                                        │
│           │   │                                        │
│           │   │                                        │
│                                                        │
│       (•> (•>                                          │
│     (•> (•>                                            │
│                                                        │
│           │   │                                        │
│           │   │                                        │
│           └───┘                                        │
└────────────────────────────────────────────────────────┘
```

### Key Highlights
* **Zero Machine Learning Dependencies:** The neural network feedforward engine, genetic chromosome encoding, selection, crossover, and mutation operators are written from first principles in pure Python and NumPy.
* **Compact Chromosomal Representation:** Each agent's brain is parameterized by an elegant 25-dimensional genome (18 synaptic weights and 7 biases) controlling a 2-6-1 Multi-Layer Perceptron (MLP).
* **Elitism & Adaptive Mutation:** Top-tier performers (top 10%) are automatically preserved as elite survivors, while subsequent offspring are synthesized via single-point crossover and stochastic scalar mutation.
* **Dynamic Time Dilation & Headless Acceleration:** Seamlessly throttle simulation execution from slow-motion frame-by-frame analysis (1 FPS) up to unthrottled headless computation (`F2`) for rapid evolutionary convergence.

---

## 🧠 System Architecture & Evolutionary Pipeline

The simulation loop couples a 2D physics simulation with generational genetic optimization:

```
[ Initialize Population: 200 Birds with Random 25-Gene Genotypes ]
                                │
                                ▼
 ┌──────────────────────────────────────────────────────────────┐
 │                    Active Game Physics Loop                  │
 │                                                              │
 │ 1. Measure Spatial Delta: Δx (horizontal) & Δy (vertical)    │
 │ 2. Feedforward Activation: Multi-Layer Perceptron (2 -> 6 -> 1)│
 │ 3. Decision Boundary: Jump if σ(output) > 0.5                │
 │ 4. Collision Check: Bounds detection & pipe obstacle collision│
 │ 5. Fitness Assignment: F = Total Distance + Δx               │
 └──────────────────────────────────────────────────────────────┘
                                │
                  All 200 Agents Dead in Generation
                                │
                                ▼
 ┌──────────────────────────────────────────────────────────────┐
 │                     Generational Evolution                   │
 │                                                              │
 │ 1. Selection & Elitism: Sort by fitness, retain top 20 units  │
 │ 2. Single-Point Crossover: Recombine parent weights (cut 0-17)│
 │ 3. Adaptive Mutation: Perturb weights by factor (P_mut = 0.1) │
 │ 4. Offspring Initialization: Spawn generation N + 1           │
 └──────────────────────────────────────────────────────────────┘
                                │
                                ▼
                [ Loop to Next Generation ]
```

---

## 📐 Mathematical Formulation

### 1. Perceptual State & Normalization
At each simulation step, an active bird evaluates its spatial offset relative to the upcoming pipe obstacle:
$$\Delta x = x_{\text{pipe}} - x_{\text{bird}}, \quad \Delta y = y_{\text{gap\_center}} - y_{\text{bird}}$$

These continuous spatial offsets are bounded to $[-350, 350]$ and scaled by gain factor $S = 200$:
$$\mathbf{x}_{\text{input}} = \begin{bmatrix} \text{clip}\left(\frac{\Delta x}{350}, -1, 1\right) \times 200 \\[6pt] \text{clip}\left(\frac{\Delta y}{350}, -1, 1\right) \times 200 \end{bmatrix}$$

### 2. Neural Network Forward Propagation
The agent's decision engine is structured as a 2-6-1 feedforward network:

* **Hidden Layer (6 Neurons, Sigmoid Activation):**
  $$h_k = \sigma\Big(\mathbf{x}_{\text{input}}[0] \cdot w_{0,k} + \mathbf{x}_{\text{input}}[1] \cdot w_{1,k} + b_k\Big), \quad k \in \{1, 2, \dots, 6\}$$
  where $\sigma(z) = \frac{1}{1 + e^{-z}}$.

* **Output Layer (1 Neuron, Sigmoid Activation):**
  $$j = \sigma\left(\sum_{k=1}^6 h_k \cdot w_{k,\text{out}} + b_{\text{out}}\right)$$

* **Action Policy:**
  $$\text{Action} = \begin{cases} \text{Flap/Jump}, & \text{if } j > 0.5 \\ \text{Glide/Fall}, & \text{otherwise} \end{cases}$$

### 3. Chromosome Representation & Genetics
Each agent's genome consists of a flat array of 25 scalar parameters:
$$\mathbf{G} = [\underbrace{w_0, w_1, \dots, w_{11}}_{\text{Input } \to \text{ Hidden Weights (12)}}, \quad \underbrace{w_{12}, \dots, w_{17}}_{\text{Hidden } \to \text{ Output Weights (6)}}, \quad \underbrace{b_{18}, \dots, b_{23}}_{\text{Hidden Biases (6)}}, \quad \underbrace{b_{24}}_{\text{Output Bias (1)}}]$$

* **Single-Point Crossover:** A random cut point $c \sim \mathcal{U}(0, 17)$ is chosen along the weight vector. Weights $[0, c)$ are inherited from Parent A, $[c, 18)$ from Parent B, and bias values are inherited from a randomly selected parent.
* **Continuous Stochastic Mutation:** Each gene has an independent mutation probability $P_{\text{mut}} = 0.10$. When triggered, the parameter is modulated by an adaptive perturbation factor:
  $$w'_i = w_i \times \Big(1 + \delta\Big), \quad \delta \in [-2.0, 2.0]$$

---

## 🎮 Real-Time Simulation Controls

The environment features interactive runtime controls allowing instant time dilation and headless simulation acceleration:

| Key | Mode / Function | Description |
| :---: | :--- | :--- |
| <kbd>F1</kbd> | **Visual Rendering On** | Enables standard Pygame graphical display (default). |
| <kbd>F2</kbd> | **Headless Accelerated Mode** | Disables display blitting, eliminating GPU/rendering sync for maximum CPU training speed. |
| <kbd>1</kbd> | **1 FPS (Slow Motion)** | Step-by-step frame examination of bird decision boundaries and flap timings. |
| <kbd>2</kbd> | **90 FPS** | Smooth half-speed playback. |
| <kbd>3</kbd> | **180 FPS** | Standard fluid simulation speed. |
| <kbd>4</kbd> | **900 FPS (Fast-Forward)** | High-speed training with visual feedback. |
| <kbd>5</kbd> | **Unlimited FPS** | Unthrottled CPU computation (clocks as many generations as hardware permits). |

---

## 📁 Repository Structure

```
flappy-bird-ai/
├── assets/                   # Simulation sprites and graphical textures
│   ├── bg.png                # Fixed parallax sky background
│   ├── bg2.png               # Scrolling ground surface texture
│   ├── bird.png              # Animated yellow bird sprite
│   ├── pause.png             # Simulation pause overlay
│   └── pipe.png              # Obstacle pipe texture
├── genetic_algorithm.py      # Core simulation: physics, neural net, & genetic algorithm
├── requirements.txt          # Python dependencies
└── LICENSE                   # GNU General Public License v3.0
```

---

## ⚡ Quickstart & Installation

### 1. Environment Setup

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/FortunateSpy5/flappy-bird-ai.git
cd flappy-bird-ai

# Create and activate virtual environment
python -m venv venv

# On Windows:
.\venv\Scripts\activate

# On Linux / macOS:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Simulation

Launch the evolutionary simulation:

```bash
python genetic_algorithm.py
```

* The simulation will spawn Generation 1 with 200 randomly parameterized birds.
* Watch the on-screen HUD to track generational progress, current best distance, and active death counts.
* Press <kbd>F2</kbd> to run in headless mode for 10–20 generations to rapidly evolve high-performing agents, then press <kbd>F1</kbd> to observe the resulting obstacle navigation behavior!

---

## 📄 License

This repository is distributed under the **GNU General Public License v3.0**. See the [LICENSE](LICENSE) file for complete details.
