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
│   │   ├── distribution/        # Sample distribution plots (updated during training)
│   │   ├── env.py               # Graph coloring environment
│   │   ├── policy.py            # GNN policy network
│   │   ├── sampler.py           # Trajectory sampler
│   │   ├── trainer.py           # Training loop with early stopping
│   │   ├── utils.py             # Graph loading utilities
│   │   ├── main.py              # Training script
│   │   ├── train_curriculum.py  # Curriculum learning script
│   │   ├── evaluate.py          # Evaluation script
│   │   └── plot_distribution.py # Plot sample distribution
│   ├── knapsack/
│   │   ├── data/                # Knapsack instances (p01/, p02/, ...)
│   │   ├── checkpoints/         # Saved model checkpoints
│   │   ├── logs/                # Training logs (JSONL)
│   │   ├── env.py               # Knapsack environment
│   │   ├── policy.py            # MLP policy network
│   │   ├── sampler.py           # Trajectory sampler
│   │   ├── trainer.py           # Training loop with early stopping
│   │   ├── utils.py             # Instance loading utilities
│   │   ├── main.py              # Training script
│   │   └── evaluate.py          # Evaluation script
│   └── tsp/
│       ├── data/                # TSP instances (.tsp files in TSPLIB format)
│       ├── checkpoints/         # Saved model checkpoints
│       ├── logs/                # Training logs (JSONL)
│       ├── env.py               # TSP environment
│       ├── policy.py            # Attention-based policy network
│       ├── sampler.py           # Trajectory sampler
│       ├── trainer.py           # Training loop with early stopping
│       ├── utils.py             # TSPLIB loading utilities
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

#### Single-Instance Training

Train GFlowNet on a single Myciel graph:

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

#### Conditional GFlowNet Training

Train a single policy that generalizes across multiple graph instances:

```bash
# Train on specific graphs
python problems/graph_coloring/main.py --conditional --graphs myciel3 myciel4 myciel5 --steps 10000

# Train on all available graphs
python problems/graph_coloring/main.py --conditional --all-graphs --steps 20000

# Train on randomly generated graphs
python problems/graph_coloring/main.py --conditional --random-graphs 50 --min-nodes 10 --max-nodes 100 --steps 20000

# Resume training from checkpoint
python problems/graph_coloring/main.py --conditional --graphs myciel3 myciel4 --resume latest
python problems/graph_coloring/main.py --conditional --graphs myciel3 myciel4 --resume checkpoints/conditional_K47_step_5000.pt
```

#### Curriculum Learning

Train with curriculum learning, gradually adding harder graphs (myciel2 → myciel3 → ... → myciel7):

```bash
# Full curriculum with config file (recommended)
python problems/graph_coloring/train_curriculum.py --config problems/graph_coloring/curriculum_config.json

# Partial curriculum (up to myciel5)
python problems/graph_coloring/train_curriculum.py --stages myciel2 myciel3 myciel4 myciel5 --steps-per-stage 5000

# Custom training parameters
python problems/graph_coloring/train_curriculum.py --steps-per-stage 10000 --batch-size 64 --lr 5e-4 --loss SubTB
```

**How it works:**
1. **Stage 1**: Train on myciel2 only
2. **Stage 2**: Add myciel3, resume from Stage 1 best checkpoint
3. **Stage 3**: Add myciel4, resume from Stage 2 best checkpoint
4. ... and so on until all stages are complete

Each stage trains until completion, saves the best checkpoint (based on distribution mean), and moves to the next stage. The policy learns easier graphs first, then progressively harder ones.

**Live Plotting:** During training, distribution plots show the initial distribution vs current distribution. For instances from previous stages, the initial distribution is taken from the best checkpoint of that stage.

**Curriculum Config File (`curriculum_config.json`):**
```json
{
    "stages": ["myciel2", "myciel3", "myciel4", "myciel5"],
    "default": {
        "steps": 5000,
        "lr": 0.001,
        "batch_size": 128,
        "epsilon": 0.1,
        "alpha": 1.0,
        "beta": 0.5,
        "gamma": 0.2
    },
    "stage_params": {
        "myciel2": {
            "steps": 1000,
            "lr": 0.01,
            "epsilon": 0.1,
            "beta": 2.0
        }
    }
}
```

Stage-specific parameters override defaults. Reward parameters (α, β, γ) can be tuned per stage.

**Curriculum Arguments:**
- `--config`: Path to curriculum config JSON file (recommended)
- `--stages`: Specific stages to train (default: myciel2 through myciel7)
- `--steps-per-stage`: Training steps per stage (default: 5000)

**Common Arguments:**
- `--steps`: Number of training steps (default: 10000)
- `--batch-size`: Number of trajectories per training step (default: 128)
- `--epsilon`: Initial epsilon for exploration, decays to 0.01 (default: 0.1)
- `--temperature`: Sampling temperature, higher = more exploration (default: 1.0)
- `--top-p`: Top-P (Nucleus) sampling threshold (default: 1.0)
- `--patience`: Early stopping patience in steps (default: 5000)
- `--loss`: Loss function: TB, DB, or SubTB (default: TB)
- `--hidden-dim`: Hidden dimension for policy networks (default: 64)
- `--save-every`: Save checkpoint every N steps (default: 1000)

**Single-Instance Arguments:**
- `--chromatic`: Target chromatic number (selects corresponding myciel graph)
- `--max-colors`: Maximum number of colors available (default: number of nodes)

**Conditional Arguments:**
- `--conditional`: Enable conditional GFlowNet mode
- `--graphs`: Specific graph names to train on (e.g., myciel3 myciel4)
- `--all-graphs`: Train on all graphs in data directory
- `--random-graphs N`: Generate N random Erdos-Renyi graphs
- `--min-nodes`, `--max-nodes`: Node range for random graphs
- `--edge-prob`: Edge probability for random graphs (default: 0.3)
- `--num-layers`: Number of GNN layers (default: 3)
- `--resume`: Resume from checkpoint (path or 'latest')

Training logs are saved to `problems/graph_coloring/logs/` in JSON Lines format.

#### Checkpoints

The training saves three types of checkpoints:
- **Periodic**: `{name}_step_{N}.pt` - Saved every `--save-every` steps
- **Best**: `{name}_best.pt` - Saved when average colors improves
- **Final**: `{name}_final.pt` - Saved at end of training

#### Evaluation

Evaluate trained models on graph coloring:

```bash
# Single-instance evaluation
python problems/graph_coloring/evaluate.py --list                    # List available graphs
python problems/graph_coloring/evaluate.py --chromatic 4 --samples 100
python problems/graph_coloring/evaluate.py --graph myciel3.col --checkpoint path/to/checkpoint.pt

# Conditional evaluation (evaluate on multiple/unseen graphs)
python problems/graph_coloring/evaluate.py --conditional --checkpoint checkpoints/conditional_K47_best.pt --graphs myciel5 myciel6
python problems/graph_coloring/evaluate.py --conditional --checkpoint checkpoints/conditional_K47_best.pt --all-graphs
```

**Single-Instance Evaluat
romatic number
- `--checkpoint`: Path to checkpoint (default: latest in checkpoints/)
- `--samples`: Number of stochastic samples (default: 100)
- `--hidden-dim`: Hidden dimension (must match training, default: 64)
- `--list`: List available graphs

**Conditional Evaluation Arguments:**
- `--conditional`: Enable conditional evaluation mode
- `--checkpoint`: Path to conditional checkpoint
- `--graphs`: Specific graphs to evaluate
- `--all-graphs`: Evaluate on all graphs
- `--samples`: Number of stochastic samples (default: 100)

**Output:** Evaluation outputs the best solution found from stochastic sampling, including:
- Status (OPTIMAL/VALID/INVALID)
- Colors used and conflicts
- Full coloring vector (node → color mapping)

### Knapsack

#### Single-Instance Training

Train GFlowNet on a single knapsack instance:

```bash
# List available problems
python problems/knapsack/main.py --list

# Train on a specific problem
python problems/knapsack/main.py --problem p01 --steps 5000
python problems/knapsack/main.py --problem p02 --steps 5000

# Train with custom batch size, exploration, and loss function
python problems/knapsack/main.py --problem p01 --batch-size 32 --epsilon 0.2 --loss SubTB --steps 3000
```

#### Conditional GFlowNet Training

Train a single policy that generalizes across multiple knapsack instances:

```bash
# Train on specific problems
python problems/knapsack/main.py --conditional --problems p01 p02 p03 --steps 10000

# Train on all available problems
python problems/knapsack/main.py --conditional --all-problems --steps 20000

# Resume training from checkpoint
python problems/knapsack/main.py --conditional --problems p01 p02 --resume latest
python problems/knapsack/main.py --conditional --problems p01 p02 --resume checkpoints/conditional_p01-p02_step_5000.pt
```

**Common Arguments:**
- `--steps`: Number of training steps (default: 5000)
- `--batch-size`: Number of trajectories per training step (default: 16)
- `--epsilon`: Initial epsilon for exploration, decays to 0.01 (default: 0.1)
- `--temperature`: Sampling temperature, higher = more exploration (default: 1.0)
- `--top-p`: Top-P (Nucleus) sampling threshold (default: 1.0)
- `--patience`: Early stopping patience in steps (default: 5000)
- `--loss`: Loss function: TB, DB, or SubTB (default: TB)
- `--hidden-dim`: Hidden dimension for policy networks (default: 128)
- `--save-every`: Save checkpoint every N steps (default: 1000)

**Single-Instance Arguments:**
- `--problem`: Problem name (e.g., p01, p02). Required for single-instance mode.
- `--list`: List all available problems and exit

**Conditional Arguments:**
- `--conditional`: Enable conditional GFlowNet mode
- `--problems`: Specific problem names to train on (e.g., p01 p02 p03)
- `--all-problems`: Train on all problems in data directory
- `--num-layers`: Number of attention layers in policy (default: 3)
- `--resume`: Resume from checkpoint (path or 'latest')

Training logs are saved to `problems/knapsack/logs/` in JSON Lines format.

#### Checkpoints

The training saves three types of checkpoints:
- **Periodic**: `{name}_step_{N}.pt` - Saved every `--save-every` steps
- **Best**: `{name}_best.pt` - Saved when average gap improves
- **Final**: `{name}_final.pt` - Saved at end of training

#### Evaluation

Evaluate trained models on knapsack problems:

```bash
# Single-instance evaluation
python problems/knapsack/evaluate.py                                    # List available problems
python problems/knapsack/evaluate.py --problem p01 --samples 100
python problems/knapsack/evaluate.py --problem p01 --checkpoint path/to/checkpoint.pt

# Conditional evaluation (evaluate on multiple/unseen problems)
python problems/knapsack/evaluate.py --conditional --checkpoint checkpoints/conditional_p01-p02-p03_best.pt --problems p01 p02 p03
python problems/knapsack/evaluate.py --conditional --checkpoint checkpoints/conditional_all_best.pt --all
```

**Single-Instance Evaluation Arguments:**
- `--problem`: Problem name to evaluate
- `--checkpoint`: Path to checkpoint (default: latest in checkpoints/)
- `--samples`: Number of stochastic samples (default: 100)
- `--hidden-dim`: Hidden dimension (must match training, default: 128)
- `--all`: Evaluate all problems with matching checkpoints

**Conditional Evaluation Arguments:**
- `--conditional`: Enable conditional evaluation mode
- `--checkpoint`: Path to conditional checkpoint
- `--problems`: Specific problems to evaluate
- `--all`: Evaluate on all problems
- `--samples`: Number of stochastic samples (default: 100)

**Output:** Evaluation outputs the best solution found from stochastic sampling, including:
- Status (OPTIMAL/VALID/INVALID)
- Profit and weight
- Items selected (indices and selection vector)

### Traveling Salesman Problem (TSP)

#### Single-Instance Training

Train GFlowNet on a TSP instance:

```bash
# List available problems
python problems/tsp/main.py --list

# Train on a TSPLIB file
python problems/tsp/main.py --problem p_all --steps 5000

# Train on a randomly generated instance
python problems/tsp/main.py --random --num-cities 20 --steps 5000
python problems/tsp/main.py --random --num-cities 50 --seed 42 --steps 10000

# Train with custom batch size, exploration, and loss function
python problems/tsp/main.py --random --num-cities 30 --batch-size 32 --epsilon 0.2 --loss SubTB --steps 5000
```

#### Conditional GFlowNet Training

Train a single policy that generalizes across multiple TSP instances:

```bash
# Train on randomly generated instances
python problems/tsp/main.py --conditional --num-instances 10 --min-cities 10 --max-cities 30 --steps 10000

# Train with more instances and larger cities
python problems/tsp/main.py --conditional --num-instances 50 --min-cities 20 --max-cities 50 --steps 20000

# Resume training from checkpoint
python problems/tsp/main.py --conditional --num-instances 10 --resume latest
```

**Common Arguments:**
- `--steps`: Number of training steps (default: 5000)
- `--batch-size`: Number of trajectories per training step (default: 16)
- `--epsilon`: Initial epsilon for exploration, decays to 0.01 (default: 0.1)
- `--temperature`: Sampling temperature, higher = more exploration (default: 1.0)
- `--top-p`: Top-P (Nucleus) sampling threshold (default: 1.0)
- `--patience`: Early stopping patience in steps (default: 5000)
- `--loss`: Loss function: TB, DB, or SubTB (default: TB)
- `--hidden-dim`: Hidden dimension for policy networks (default: 128)
- `--save-every`: Save checkpoint every N steps (default: 1000)

**Single-Instance Arguments:**
- `--problem`: Problem name (TSPLIB file in data directory)
- `--random`: Generate a random TSP instance
- `--num-cities`: Number of cities for random instance (default: 20)
- `--seed`: Random seed for instance generation
- `--list`: List all available problems and exit

**Conditional Arguments:**
- `--conditional`: Enable conditional GFlowNet mode
- `--num-instances`: Number of random instances to generate (default: 10)
- `--min-cities`: Minimum cities per instance (default: 10)
- `--max-cities`: Maximum cities per instance (default: 30)
- `--num-layers`: Number of attention layers in policy (default: 3)
- `--resume`: Resume from checkpoint (path or 'latest')

Training logs are saved to `problems/tsp/logs/` in JSON Lines format.

#### Checkpoints

The training saves three types of checkpoints:
- **Periodic**: `{name}_step_{N}.pt` - Saved every `--save-every` steps
- **Best**: `{name}_best.pt` - Saved when average gap improves
- **Final**: `{name}_final.pt` - Saved at end of training

#### Evaluation

Evaluate trained models on TSP problems:

```bash
# Single-instance evaluation
python problems/tsp/evaluate.py --list                                    # List available problems
python problems/tsp/evaluate.py --problem p_all --samples 100
python problems/tsp/evaluate.py --random --num-cities 20 --checkpoint path/to/checkpoint.pt

# Conditional evaluation (evaluate on random instances)
python problems/tsp/evaluate.py --conditional --checkpoint checkpoints/conditional_best.pt --num-instances 10
```

**Single-Instance Evaluation Arguments:**
- `--problem`: Problem name to evaluate
- `--random`: Evaluate on a random instance
- `--num-cities`: Number of cities for random instance
- `--checkpoint`: Path to checkpoint (default: latest in checkpoints/)
- `--samples`: Number of stochastic samples (default: 100)
- `--hidden-dim`: Hidden dimension (must match training, default: 128)

**Conditional Evaluation Arguments:**
- `--conditional`: Enable conditional evaluation mode
- `--checkpoint`: Path to conditional checkpoint
- `--num-instances`: Number of random instances to evaluate (default: 10)
- `--min-cities`, `--max-cities`: City range for random instances
- `--samples`: Number of stochastic samples (default: 100)

**Output:** Evaluation outputs the best solution found from stochastic sampling, including:
- Tour length (and gap to optimal if known)
- Tour order (sequence of cities)

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

### TSP (Traveling Salesman Problem)
| Problem | Cities | Description |
|---------|--------|-------------|
| p_all (uy734) | 734 | 734 locations in Uruguay |

*Note: TSP also supports randomly generated instances with `--random --num-cities N`*

## Method

This project uses **Generative Flow Networks (GFlowNets)** to learn policies that sample diverse, high-quality solutions to combinatorial optimization problems.

### Loss Functions

Three loss functions are available:
- **TB (Trajectory Balance)**: Standard trajectory balance loss
- **DB (Detailed Balance)**: Detailed balance loss for finer-grained credit assignment
- **SubTB (Sub-Trajectory Balance)**: Sub-trajectory balance with configurable lambda

### Key Features
- **Conditional GFlowNets**: Train a single GNN policy that generalizes across multiple graph instances
- **Multiple loss functions**: TB, DB, SubTB
- **Exploration strategies**: Epsilon-greedy with decay, temperature scaling, Top-P (Nucleus) sampling
- **Training utilities**: Early stopping, checkpoint saving (periodic/best/final), resume from checkpoint
- **Evaluation modes**: Greedy and stochastic sampling on seen and unseen instances

## Checkpoint Naming

Checkpoints are saved with problem-specific names:
- **Graph Coloring (Single)**: `{graph}_K{colors}_step_{step}.pt` (e.g., `myciel3_K11_step_1000.pt`)
- **Graph Coloring (Conditional)**: `conditional_K{colors}_{type}.pt` where type is `step_N`, `best`, or `final`
- **Knapsack (Single)**: `{problem}_step_{step}.pt` (e.g., `p01_step_1000.pt`)
- **Knapsack (Conditional)**: `conditional_{problems}_{type}.pt` (e.g., `conditional_p01-p02-p03_best.pt`)
- **TSP (Single)**: `{problem}_step_{step}.pt` or `random_{N}_step_{step}.pt`
- **TSP (Conditional)**: `conditional_{N}inst_{min}-{max}cities_{type}.pt`

The evaluation scripts automatically find the latest checkpoint for each problem.

## Adding New Problems

To add a new combinatorial optimization problem:

1. Create a new folder under `problems/` (e.g., `problems/new_problem/`)
2. Implement the required files:
   - `env.py` - Environment class inheriting from `BaseEnv`
   - `policy.py` - Neural network policy
   - `utils.py` - Data loading utilities
   - `main.py` - Training script
   - `evaluate.py` - Evaluation script
3. Add data files under `data/`

## License

See [LICENSE](LICENSE) for details.
