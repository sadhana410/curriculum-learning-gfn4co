# GFlowNets for Combinatorial Optimization

Studying how curriculum learning can enhance the performance of GFlowNets for combinatorial optimization problems.

## Project Structure

```
curriculum-learning-gfn4co/
├── envs/                        # Shared base classes
│   └── base_env.py              # Base environment class
├── losses/
│   ├── trajectorybalance.py     # Trajectory Balance (TB) loss
│   ├── detailedbalance.py       # Detailed Balance (DB) loss
│   └── subtrajectorybalance.py  # Sub-Trajectory Balance (SubTB) loss
├── visualization/
│   └── plot_graph.py            # Graph visualization
├── problems/                    # Problem-specific implementations
│   ├── graph_coloring/
│   │   ├── data/                # Myciel graph instances (.col files)
│   │   ├── checkpoints/         # Saved model checkpoints
│   │   ├── logs/                # Training logs (JSONL)
│   │   ├── env.py               # Graph coloring environment
│   │   ├── policy.py            # GNN policy network
│   │   ├── sampler.py           # Trajectory sampler
│   │   ├── trainer.py           # Training loop with early stopping
│   │   ├── utils.py             # Graph loading utilities
│   │   ├── main.py              # Training script
│   │   └── evaluate.py          # Evaluation script
│   └── knapsack/
│       ├── data/                # Knapsack instances (p01/, p02/, ...)
│       ├── checkpoints/         # Saved model checkpoints
│       ├── logs/                # Training logs (JSONL)
│       ├── env.py               # Knapsack environment
│       ├── policy.py            # MLP policy network
│       ├── sampler.py           # Trajectory sampler
│       ├── trainer.py           # Training loop with early stopping
│       ├── utils.py             # Instance loading utilities
│       ├── main.py              # Training script
│       └── evaluate.py          # Evaluation script
```

## Installation

```bash
conda create -n gfn python=3.9
conda activate gfn
pip install -r requirements.txt
```

## Usage

### Graph Coloring

Train GFlowNet on Myciel graphs:

```bash
# Train on myciel2 (chromatic number = 3)
python problems/graph_coloring/main.py --chromatic 3 --steps 5000

# Train on myciel3 (chromatic number = 4)
python problems/graph_coloring/main.py --chromatic 4 --steps 10000

# Train on myciel4 (chromatic number = 5) with max 7 colors
python problems/graph_coloring/main.py --chromatic 5 --max-colors 7 --steps 10000

# Train with custom batch size, exploration, and loss function
python problems/graph_coloring/main.py --chromatic 4 --batch-size 64 --epsilon 0.2 --loss SubTB --steps 5000
```

**Arguments:**
- `--chromatic`: Target chromatic number (selects corresponding myciel graph)
- `--max-colors`: Maximum number of colors available (default: number of nodes)
- `--steps`: Number of training steps (default: 10000)
- `--batch-size`: Number of trajectories per training step (default: 128)
- `--epsilon`: Initial epsilon for exploration, decays to 0.01 (default: 0.1)
- `--temperature`: Sampling temperature, higher = more exploration (default: 1.0)
- `--top-p`: Top-P (Nucleus) sampling threshold (default: 1.0)
- `--patience`: Early stopping patience in steps (default: 5000)
- `--loss`: Loss function: TB, DB, or SubTB (default: TB)
- `--hidden-dim`: Hidden dimension for policy networks (default: 64)
- `--save-every`: Save checkpoint every N steps (default: 1000)

Training logs are saved to `problems/graph_coloring/logs/` in JSON Lines format.

#### Evaluation

Evaluate trained models on graph coloring:

```bash
# List available graphs
python problems/graph_coloring/evaluate.py --list

# Evaluate by chromatic number
python problems/graph_coloring/evaluate.py --chromatic 4 --samples 100

# Evaluate specific graph with specific checkpoint
python problems/graph_coloring/evaluate.py --graph myciel3.col --checkpoint path/to/checkpoint.pt
```

**Evaluation Arguments:**
- `--graph`: Graph file to evaluate (e.g., myciel3.col)
- `--chromatic`: Select graph by chromatic number
- `--checkpoint`: Path to checkpoint (default: latest in checkpoints/)
- `--samples`: Number of stochastic samples (default: 100)
- `--hidden-dim`: Hidden dimension (must match training, default: 64)
- `--list`: List available graphs

### Knapsack

Train GFlowNet on knapsack instances:

```bash
# List available problems
python problems/knapsack/main.py --list

# Train on a specific problem
python problems/knapsack/main.py --problem p01 --steps 5000
python problems/knapsack/main.py --problem p02 --steps 5000

# Train with custom batch size, exploration, and loss function
python problems/knapsack/main.py --problem p01 --batch-size 32 --epsilon 0.2 --loss SubTB --steps 3000
```

**Arguments:**
- `--problem`: Problem name (e.g., p01, p02). Required.
- `--list`: List all available problems and exit
- `--steps`: Number of training steps (default: 5000)
- `--batch-size`: Number of trajectories per training step (default: 16)
- `--epsilon`: Initial epsilon for exploration, decays to 0.01 (default: 0.1)
- `--temperature`: Sampling temperature, higher = more exploration (default: 1.0)
- `--top-p`: Top-P (Nucleus) sampling threshold (default: 1.0)
- `--patience`: Early stopping patience in steps (default: 5000)
- `--loss`: Loss function: TB, DB, or SubTB (default: TB)
- `--hidden-dim`: Hidden dimension for policy networks (default: 128)
- `--save-every`: Save checkpoint every N steps (default: 1000)

Training logs are saved to `problems/knapsack/logs/` in JSON Lines format.

#### Evaluation

Evaluate trained models on knapsack problems:

```bash
# List available problems
python problems/knapsack/evaluate.py

# Evaluate on a specific problem
python problems/knapsack/evaluate.py --problem p01 --samples 100

# Evaluate with specific checkpoint
python problems/knapsack/evaluate.py --problem p01 --checkpoint path/to/checkpoint.pt
```

**Evaluation Arguments:**
- `--problem`: Problem name to evaluate
- `--checkpoint`: Path to checkpoint (default: latest in checkpoints/)
- `--samples`: Number of stochastic samples (default: 100)
- `--hidden-dim`: Hidden dimension (must match training, default: 128)
- `--all`: Evaluate all problems with matching checkpoints

## Available Datasets

### Graph Coloring (Myciel Graphs)
| File | Nodes | Chromatic Number |
|------|-------|------------------|
| myciel2.col | 5 | 3 |
| myciel3.col | 11 | 4 |
| myciel4.col | 23 | 5 |
| myciel5.col | 47 | 6 |
| myciel6.col | 95 | 7 |
| myciel7.col | 191 | 8 |

### Knapsack
| Problem | Items | Capacity | Optimal Profit |
|---------|-------|----------|----------------|
| p01 | 10 | 165 | 309 |
| p02 | 5 | 26 | 51 |
| p03 | 6 | 190 | 150 |
| p04 | 7 | 50 | 107 |
| p05 | 8 | 104 | 900 |
| p06 | 7 | 170 | 1735 |
| p07 | 15 | 750 | 1458 |
| p08 | 24 | 6404180 | 13549094 |

## Method

This project uses **Generative Flow Networks (GFlowNets)** to learn policies that sample diverse, high-quality solutions to combinatorial optimization problems.

### Loss Functions

Three loss functions are available:
- **TB (Trajectory Balance)**: Standard trajectory balance loss
- **DB (Detailed Balance)**: Detailed balance loss for finer-grained credit assignment
- **SubTB (Sub-Trajectory Balance)**: Sub-trajectory balance with configurable lambda

### Key Features
- Multiple loss functions (TB, DB, SubTB)
- Epsilon-greedy exploration with decay during training
- Temperature and Top-P (Nucleus) sampling for exploration control
- Early stopping with configurable patience
- Checkpoint saving at configurable intervals (default: every 1000 steps)
- Reward shaping to encourage optimal solutions
- Greedy and stochastic evaluation modes

## Checkpoint Naming

Checkpoints are saved with problem-specific names:
- **Graph Coloring**: `{graph}_K{colors}_step_{step}.pt` (e.g., `myciel3_K5_step_1000.pt`)
- **Knapsack**: `{problem}_step_{step}.pt` (e.g., `p01_step_1000.pt`)

The evaluation scripts automatically find the latest checkpoint for each problem.

## Adding New Problems

To add a new combinatorial optimization problem:

1. Create a new folder under `problems/` (e.g., `problems/tsp/`)
2. Implement the required files:
   - `env.py` - Environment class inheriting from `BaseEnv`
   - `policy.py` - Neural network policy
   - `utils.py` - Data loading utilities
   - `main.py` - Training script
   - `evaluate.py` - Evaluation script
3. Add data files under `data/`

## License

See [LICENSE](LICENSE) for details.
