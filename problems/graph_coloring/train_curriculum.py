# problems/graph_coloring/train_curriculum.py
"""
Curriculum learning for Graph Coloring GFlowNet.

Gradually adds harder graphs to the training set:
  myciel2 -> myciel3 -> myciel4 -> myciel5 -> myciel6 -> myciel7

Each stage trains for a specified number of steps, then adds the next
harder graph and resumes training from the previous checkpoint.
"""

import argparse
import os
import sys
import json
import time
import random
from datetime import datetime
import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from problems.graph_coloring.env import ConditionalGraphColoringEnv
from problems.graph_coloring.policy import ConditionalGNNPolicy, ConditionalGNNPolicyWrapper
from problems.graph_coloring.utils import load_col_file
from problems.graph_coloring.trainer import train_conditional
from losses.trajectorybalance import TrajectoryBalance
from losses.detailedbalance import DetailedBalance
from losses.subtrajectorybalance import SubTrajectoryBalance


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For full determinism (may reduce performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")


# Data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Curriculum order (easy to hard)
CURRICULUM_ORDER = ["myciel2", "myciel3", "myciel4", "myciel5", "myciel6", "myciel7"]

# Chromatic numbers
CHROMATIC_NUMBERS = {
    "myciel2.col": 3,
    "myciel3.col": 4,
    "myciel4.col": 5,
    "myciel5.col": 6,
    "myciel6.col": 7,
    "myciel7.col": 8,
}


class CurriculumScheduler:
    """
    Scheduler for curriculum learning with per-stage hyperparameters.
    
    Allows customizing learning rate, batch size, epsilon, temperature, etc.
    for each stage of curriculum learning.
    
    Example usage:
        scheduler = CurriculumScheduler(
            stages=["myciel2", "myciel3", "myciel4"],
            default_params={
                "steps": 5000,
                "lr": 1e-3,
                "batch_size": 128,
                "epsilon": 0.1,
                "temperature": 1.0,
            },
            stage_params={
                "myciel2": {"steps": 2000, "lr": 1e-3},  # Quick warmup
                "myciel3": {"steps": 5000, "lr": 5e-4},  # Standard training
                "myciel4": {"steps": 10000, "lr": 1e-4, "epsilon": 0.05},  # Fine-tuning
            }
        )
        
        for stage in scheduler:
            params = scheduler.get_params(stage)
            # Train with params...
    """
    
    def __init__(self, stages, default_params=None, stage_params=None, cumulative=True):
        """
        Args:
            stages: List of stage names (e.g., ["myciel2", "myciel3", "myciel4"])
            default_params: Default hyperparameters for all stages
            stage_params: Dict mapping stage name to stage-specific params (overrides defaults)
            cumulative: If True, each stage includes all previous graphs. If False, each stage only has its own graph.
        """
        self.stages = stages
        self.cumulative = cumulative
        self.default_params = default_params or {
            "steps": 5000,
            "lr": 1e-3,
            "batch_size": 128,
            "epsilon": 0.1,
            "temperature": 1.0,
            "top_p": 1.0,
            "patience": -1,
            "save_every": 1000,
            "alpha": 1.0,   # Reward: conflict penalty
            "beta": 0.5,    # Reward: color usage penalty
            "gamma": 0.2,   # Reward: uncolored penalty
        }
        self.stage_params = stage_params or {}
        self.current_stage_idx = 0
    
    def __iter__(self):
        self.current_stage_idx = 0
        return self
    
    def __next__(self):
        if self.current_stage_idx >= len(self.stages):
            raise StopIteration
        stage = self.stages[self.current_stage_idx]
        self.current_stage_idx += 1
        return stage
    
    def __len__(self):
        return len(self.stages)
    
    def get_params(self, stage):
        """Get hyperparameters for a specific stage."""
        params = self.default_params.copy()
        if stage in self.stage_params:
            params.update(self.stage_params[stage])
        return params
    
    def get_stage_idx(self, stage):
        """Get the index of a stage."""
        return self.stages.index(stage)
    
    def get_graphs_for_stage(self, stage):
        """Get graphs for this stage based on cumulative setting."""
        if self.cumulative:
            # Include all graphs up to and including this stage
            idx = self.get_stage_idx(stage)
            return self.stages[:idx + 1]
        else:
            # Only include this stage's graph
            return [stage]
    
    @classmethod
    def from_args(cls, args):
        """Create scheduler from command-line arguments."""
        stages = args.stages if args.stages else CURRICULUM_ORDER
        stages = [s for s in stages if s in CURRICULUM_ORDER]
        
        default_params = {
            "steps": args.steps_per_stage,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epsilon": args.epsilon,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "patience": args.patience,
            "save_every": args.save_every,
        }
        
        # Load stage-specific params from config file if provided
        stage_params = {}
        cumulative = True  # Default to cumulative
        if hasattr(args, 'config') and args.config:
            config = cls.load_config(args.config)
            if 'stages' in config and not args.stages:
                stages = config['stages']
                stages = [s for s in stages if s in CURRICULUM_ORDER]
            if 'default' in config:
                default_params.update(config['default'])
            if 'stage_params' in config:
                stage_params = config['stage_params']
            if 'cumulative' in config:
                cumulative = config['cumulative']
        
        return cls(stages, default_params, stage_params, cumulative)
    
    @classmethod
    def load_config(cls, config_path):
        """
        Load curriculum config from a JSON file.
        
        Example config file (curriculum_config.json):
        {
            "stages": ["myciel2", "myciel3", "myciel4", "myciel5"],
            "default": {
                "steps": 5000,
                "lr": 0.001,
                "batch_size": 128,
                "epsilon": 0.1
            },
            "stage_params": {
                "myciel2": {"steps": 2000, "lr": 0.001},
                "myciel3": {"steps": 5000, "lr": 0.0005},
                "myciel4": {"steps": 10000, "lr": 0.0001, "epsilon": 0.05},
                "myciel5": {"steps": 20000, "lr": 0.00005, "epsilon": 0.02}
            }
        }
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return config
    
    def save_config(self, config_path):
        """Save current curriculum config to a JSON file."""
        config = {
            "stages": self.stages,
            "cumulative": self.cumulative,
            "default": self.default_params,
            "stage_params": self.stage_params,
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Saved curriculum config to: {config_path}")
    
    def summary(self):
        """Print a summary of the curriculum schedule."""
        print("Curriculum Schedule:")
        print(f"Mode: {'Cumulative' if self.cumulative else 'Non-cumulative (single graph per stage)'}")
        print("-" * 60)
        for i, stage in enumerate(self.stages):
            params = self.get_params(stage)
            graphs = self.get_graphs_for_stage(stage)
            print(f"  Stage {i+1}: {stage}")
            print(f"    Graphs: {', '.join(graphs)}")
            print(f"    Steps: {params['steps']}, LR: {params['lr']}, "
                  f"Batch: {params.get('batch_size', 128)}, ε: {params.get('epsilon', 0.1)}")
            print(f"    Reward: α={params.get('alpha', 1.0)}, β={params.get('beta', 0.5)}, γ={params.get('gamma', 0.2)}")
        print("-" * 60)


def load_graph(name):
    """Load a graph by name (e.g., 'myciel3')."""
    if not name.endswith('.col'):
        name = name + '.col'
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Graph not found: {path}")
    adj = load_col_file(path)
    return {
        'adj': adj,
        'name': name.replace('.col', ''),
        'chromatic_number': CHROMATIC_NUMBERS.get(name, None)
    }


def find_latest_checkpoint(save_dir, prefix):
    """Find the latest checkpoint with given prefix."""
    if not os.path.exists(save_dir):
        return None
    
    checkpoints = []
    for f in os.listdir(save_dir):
        if f.startswith(prefix) and f.endswith('.pt'):
            checkpoints.append(os.path.join(save_dir, f))
    
    if not checkpoints:
        return None
    
    # Prefer _best.pt, then _final.pt, then latest step
    for suffix in ['_best.pt', '_final.pt']:
        for ckpt in checkpoints:
            if ckpt.endswith(suffix):
                return ckpt
    
    # Sort by step number
    def get_step(path):
        name = os.path.basename(path)
        if 'step_' in name:
            try:
                return int(name.split('step_')[1].split('.')[0])
            except:
                pass
        return 0
    
    checkpoints.sort(key=get_step, reverse=True)
    return checkpoints[0]


def train_curriculum(args):
    """
    Train with curriculum learning: gradually add harder graphs.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Create curriculum scheduler
    scheduler = CurriculumScheduler.from_args(args)
    stages = scheduler.stages
    
    if not stages:
        print("Error: No valid stages specified")
        return
    
    print("=" * 60)
    print("CURRICULUM LEARNING FOR GRAPH COLORING")
    print("=" * 60)
    print(f"Stages: {' -> '.join(stages)}")
    print(f"Total stages: {len(stages)}")
    
    # Show schedule if config file provided or stage params exist
    if scheduler.stage_params:
        print()
        scheduler.summary()
    else:
        print(f"Default steps per stage: {args.steps_per_stage}")
    print()
    
    # Directories
    save_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Curriculum log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    curriculum_log_path = os.path.join(log_dir, f"curriculum_{timestamp}.jsonl")
    
    # Log curriculum config
    curriculum_config = {
        "type": "curriculum_config",
        "stages": stages,
        "default_params": scheduler.default_params,
        "stage_params": scheduler.stage_params,
        "max_colors": args.max_colors,
        "loss": args.loss,
        "timestamp": timestamp,
    }
    with open(curriculum_log_path, "w") as f:
        f.write(json.dumps(curriculum_config) + "\n")
    
    print(f"Curriculum log: {curriculum_log_path}")
    print()
    
    # Initialize policy (will be reused across stages)
    K = args.max_colors
    shared_policy = None
    forward = None
    backward = None
    loss_fn = None
    optimizer = None
    current_lr = args.lr  # Track current LR for optimizer updates
    
    # Track results across stages
    all_results = {}
    
    # Track best distributions from previous stages for plotting
    prev_stage_distributions = {}  # instance_name -> {'distribution': {...}, 'step': int}
    
    # Train each stage
    for stage_idx, stage_name in enumerate(stages):
        # Get stage-specific parameters
        stage_params = scheduler.get_params(stage_name)
        
        print()
        print("=" * 60)
        print(f"STAGE {stage_idx + 1}/{len(stages)}: Training up to {stage_name}")
        print("=" * 60)
        
        # Build instance list for this stage (all graphs up to current)
        current_graphs = stages[:stage_idx + 1]
        instances = []
        for graph_name in current_graphs:
            inst = load_graph(graph_name)
            instances.append(inst)
            print(f"  Loaded: {graph_name} ({inst['adj'].shape[0]} nodes, chromatic={inst['chromatic_number']})")
        
        # Determine K if not specified
        if K is None:
            max_nodes = max(inst['adj'].shape[0] for inst in instances)
            K = max_nodes
        
        # Create environment
        env = ConditionalGraphColoringEnv(instances, num_colors=K)
        
        # Set reward parameters for this stage
        alpha = stage_params.get('alpha', 1.0)
        beta = stage_params.get('beta', 0.5)
        gamma = stage_params.get('gamma', 0.2)
        env.set_reward_params(alpha=alpha, beta=beta, gamma=gamma)
        
        # Initialize or update policy
        if shared_policy is None:
            # First stage: create new policy
            shared_policy = ConditionalGNNPolicy(
                num_colors=K,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers
            ).to(device)
            forward = ConditionalGNNPolicyWrapper(shared_policy, mode='forward')
            backward = ConditionalGNNPolicyWrapper(shared_policy, mode='backward')
            
            # Loss function
            if args.loss == "TB":
                loss_fn = TrajectoryBalance(forward, backward).to(device)
            elif args.loss == "DB":
                loss_fn = DetailedBalance(forward, backward).to(device)
            elif args.loss == "SubTB":
                loss_fn = SubTrajectoryBalance(forward, backward, lambda_=1.0).to(device)
            
            # Optimizer with stage-specific learning rate
            current_lr = stage_params['lr']
            optimizer = torch.optim.Adam(
                list(shared_policy.parameters()) + list(loss_fn.parameters()),
                lr=current_lr
            )
            
            resume_from = None
        else:
            # Subsequent stages: resume from previous checkpoint
            prev_stage = stages[stage_idx - 1]
            prev_problem_name = f"curriculum_stage{stage_idx}_{'-'.join(stages[:stage_idx])}_K{K}"
            resume_from = find_latest_checkpoint(save_dir, prev_problem_name)
            
            if resume_from:
                print(f"  Resuming from: {os.path.basename(resume_from)}")
            else:
                print(f"  Warning: No checkpoint found for previous stage, continuing with current weights")
                resume_from = None
            
            # Update learning rate if changed for this stage
            new_lr = stage_params['lr']
            if new_lr != current_lr:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                current_lr = new_lr
                print(f"  Updated learning rate to: {new_lr}")
        
        # Problem name for this stage
        problem_name = f"curriculum_stage{stage_idx + 1}_{'-'.join(current_graphs)}_K{K}"
        
        # Print stage parameters
        print(f"\n  Problem name: {problem_name}")
        print(f"  Steps: {stage_params['steps']}, LR: {stage_params['lr']}, "
              f"Batch: {stage_params['batch_size']}, ε: {stage_params['epsilon']}")
        print(f"  Reward: α={alpha}, β={beta}, γ={gamma}")
        print()
        
        # Train this stage
        stage_start_time = time.time()
        
        best_per_instance, init_dist_info = train_conditional(
            env=env,
            forward_policy=forward,
            backward_policy=backward,
            loss_fn=loss_fn,
            optimizer=optimizer,
            steps=stage_params['steps'],
            device=device,
            save_dir=save_dir,
            problem_name=problem_name,
            batch_size=stage_params['batch_size'],
            epsilon_start=stage_params['epsilon'],
            log_dir=log_dir,
            save_every=stage_params['save_every'],
            temperature=stage_params['temperature'],
            top_p=stage_params['top_p'],
            early_stop_patience=stage_params['patience'],
            same_instance_per_batch=args.same_instance_per_batch,
            resume_from=resume_from,
            prev_stage_distributions=prev_stage_distributions if stage_idx > 0 else None,
            current_stage_instance=stage_name  # Use current stage graph for best checkpoint criterion
        )
        
        stage_time = time.time() - stage_start_time
        
        # Load best distributions from this stage's best checkpoint for next stage
        best_ckpt_path = os.path.join(save_dir, f'{problem_name}_best.pt')
        if os.path.exists(best_ckpt_path):
            best_ckpt = torch.load(best_ckpt_path, map_location='cpu', weights_only=False)
            if 'best_distributions' in best_ckpt:
                # Update prev_stage_distributions with this stage's best distributions
                prev_stage_distributions.update(best_ckpt['best_distributions'])
        
        # Log stage results with initial distribution
        stage_results = {
            "type": "stage_complete",
            "stage": stage_idx + 1,
            "stage_name": stage_name,
            "graphs": current_graphs,
            "initial_distributions": init_dist_info,
            "best_per_instance": {str(k): v['colors'] for k, v in best_per_instance.items()},
            "stage_time_seconds": stage_time,
        }
        with open(curriculum_log_path, "a") as f:
            f.write(json.dumps(stage_results) + "\n")
        
        # Store results
        all_results[stage_name] = best_per_instance
        
        print()
        print(f"Stage {stage_idx + 1} complete in {stage_time/60:.1f} minutes")
        print("Results:")
        for idx, info in best_per_instance.items():
            inst = env.get_instance(idx)
            chromatic = inst.get('chromatic_number', '?')
            gap = info['colors'] - chromatic if chromatic else 0
            print(f"  {info['name']}: {info['colors']} colors (chromatic={chromatic}, gap={gap})")
    
    # Final summary
    print()
    print("=" * 60)
    print("CURRICULUM TRAINING COMPLETE")
    print("=" * 60)
    print()
    print("Final results per stage:")
    for stage_name, results in all_results.items():
        print(f"\n  After adding {stage_name}:")
        for idx, info in results.items():
            print(f"    {info['name']}: {info['colors']} colors")
    
    # Log final summary
    final_summary = {
        "type": "curriculum_complete",
        "all_results": {
            stage: {str(k): v['colors'] for k, v in results.items()}
            for stage, results in all_results.items()
        }
    }
    with open(curriculum_log_path, "a") as f:
        f.write(json.dumps(final_summary) + "\n")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Curriculum Learning for Graph Coloring GFlowNet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default curriculum (myciel2 -> myciel7)
  python train_curriculum.py

  # Train with specific stages
  python train_curriculum.py --stages myciel2 myciel3 myciel4

  # Custom steps per stage
  python train_curriculum.py --steps-per-stage 5000

  # Load config from file
  python train_curriculum.py --config curriculum_config.json
        """
    )
    
    # Curriculum arguments
    parser.add_argument("--stages", nargs="+", default=None,
                        help="Curriculum stages (default: myciel2 through myciel7)")
    parser.add_argument("--steps-per-stage", type=int, default=5000,
                        help="Default training steps per stage (default: 5000)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to curriculum config JSON file with per-stage hyperparameters")
    
    # Training arguments
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--max-colors", type=int, default=191,
                        help="Maximum number of colors K (default: 191)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size (default: 128)")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="Initial exploration epsilon (default: 0.1)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (default: 1.0)")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="Nucleus sampling threshold (default: 1.0)")
    parser.add_argument("--patience", type=int, default=-1,
                        help="Early stopping patience per stage (-1 to disable)")
    parser.add_argument("--save-every", type=int, default=1000,
                        help="Checkpoint frequency (default: 1000)")
    parser.add_argument("--loss", type=str, default="TB", choices=["TB", "DB", "SubTB"],
                        help="Loss function (default: TB)")
    parser.add_argument("--hidden-dim", type=int, default=64,
                        help="Hidden dimension (default: 64)")
    parser.add_argument("--num-layers", type=int, default=3,
                        help="Number of GNN layers (default: 3)")
    parser.add_argument("--same-instance-per-batch", action="store_true", default=True,
                        help="Use same instance per batch (default: True)")
    parser.add_argument("--mixed-batch", dest="same_instance_per_batch", action="store_false",
                        help="Mix instances in batch")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    # Set deterministic random seeds
    set_seed(args.seed)
    
    train_curriculum(args)


if __name__ == "__main__":
    main()
