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
    
    # Parse curriculum stages
    if args.stages:
        stages = args.stages
    else:
        # Default: all myciel graphs
        stages = CURRICULUM_ORDER
    
    # Validate stages
    for stage in stages:
        if stage not in CURRICULUM_ORDER:
            print(f"Warning: Unknown graph '{stage}', skipping")
    stages = [s for s in stages if s in CURRICULUM_ORDER]
    
    if not stages:
        print("Error: No valid stages specified")
        return
    
    print("=" * 60)
    print("CURRICULUM LEARNING FOR GRAPH COLORING")
    print("=" * 60)
    print(f"Stages: {' -> '.join(stages)}")
    print(f"Steps per stage: {args.steps_per_stage}")
    print(f"Total stages: {len(stages)}")
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
        "steps_per_stage": args.steps_per_stage,
        "max_colors": args.max_colors,
        "batch_size": args.batch_size,
        "lr": args.lr,
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
    
    # Track results across stages
    all_results = {}
    
    # Train each stage
    for stage_idx, stage_name in enumerate(stages):
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
            
            # Optimizer
            optimizer = torch.optim.Adam(
                list(shared_policy.parameters()) + list(loss_fn.parameters()),
                lr=args.lr
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
        
        # Problem name for this stage
        problem_name = f"curriculum_stage{stage_idx + 1}_{'-'.join(current_graphs)}_K{K}"
        
        print(f"\n  Problem name: {problem_name}")
        print(f"  Training for {args.steps_per_stage} steps...")
        print()
        
        # Train this stage
        stage_start_time = time.time()
        
        best_per_instance = train_conditional(
            env=env,
            forward_policy=forward,
            backward_policy=backward,
            loss_fn=loss_fn,
            optimizer=optimizer,
            steps=args.steps_per_stage,
            device=device,
            save_dir=save_dir,
            problem_name=problem_name,
            batch_size=args.batch_size,
            epsilon_start=args.epsilon,
            log_dir=log_dir,
            save_every=args.save_every,
            temperature=args.temperature,
            top_p=args.top_p,
            early_stop_patience=args.patience,
            same_instance_per_batch=args.same_instance_per_batch,
            resume_from=resume_from
        )
        
        stage_time = time.time() - stage_start_time
        
        # Log stage results
        stage_results = {
            "type": "stage_complete",
            "stage": stage_idx + 1,
            "stage_name": stage_name,
            "graphs": current_graphs,
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
        """
    )
    
    # Curriculum arguments
    parser.add_argument("--stages", nargs="+", default=None,
                        help="Curriculum stages (default: myciel2 through myciel7)")
    parser.add_argument("--steps-per-stage", type=int, default=5000,
                        help="Training steps per curriculum stage (default: 5000)")
    
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
    parser.add_argument("--patience", type=int, default=2000,
                        help="Early stopping patience per stage (default: 2000)")
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
    
    args = parser.parse_args()
    train_curriculum(args)


if __name__ == "__main__":
    main()
