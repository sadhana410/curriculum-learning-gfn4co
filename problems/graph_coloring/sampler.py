# problems/graph_coloring/sampler.py

import torch
import numpy as np


def sample_trajectory(env, forward_policy, device, epsilon=0.1):
    """Sample a single trajectory with epsilon-greedy exploration."""
    state = env.reset()
    traj_states, traj_actions = [], []

    done = False
    while not done:
        traj_states.append(state.copy())

        logits = forward_policy(state, env.adj, device=device)
        mask = torch.tensor(env.allowed_actions(state), dtype=torch.float32).to(device)
        
        if mask.sum() == 0:
            done = True
            reward = env.reward(state)
            break
        
        if torch.rand(1).item() < epsilon:
            valid_actions = torch.where(mask > 0)[0]
            action = valid_actions[torch.randint(len(valid_actions), (1,))].item()
        else:
            masked_logits = logits.clone()
            masked_logits[mask == 0] = float('-inf')
            probs = torch.softmax(masked_logits, dim=0)
            action = torch.multinomial(probs, 1).item()

        next_state, reward, done = env.step(state, action)
        traj_actions.append(action)
        state = next_state

    traj_states.append(state.copy())
    return traj_states, traj_actions, reward


def get_batched_mask(states, env, device):
    """
    Compute allowed actions mask for a batch of states.
    states: (B, N) numpy array or tensor
    Returns: (B, N*K) tensor boolean mask
    """
    if isinstance(states, np.ndarray):
        states_t = torch.from_numpy(states).to(device=device, dtype=torch.long) # (B, N)
    else:
        states_t = states.to(dtype=torch.long)
        
    B, N = states_t.shape
    K = env.K
    
    # Get adjacency tensor
    if hasattr(env, 'adj_t') and env.adj_t.device == device:
        adj = env.adj_t
    else:
        adj = torch.from_numpy(env._adj_np).to(device=device, dtype=torch.float32)
        env.adj_t = adj
        
    # 1. Identify uncolored nodes: (B, N)
    uncolored_mask = (states_t == -1)
    
    # 2. Compute neighbors' colors for each node
    # We want a tensor Forbidden of shape (B, N, K)
    # Forbidden[b, u, c] = 1 if any neighbor v of u has color c
    
    # Expand states to one-hot colors: (B, N, K)
    # Treat -1 as no color (zero vector)
    colors_onehot = torch.zeros(B, N, K, device=device)
    
    # Only look at colored nodes
    colored_mask = (states_t != -1)
    batch_idx = torch.arange(B, device=device).view(B, 1).expand(B, N)
    node_idx = torch.arange(N, device=device).view(1, N).expand(B, N)
    
    valid_indices = colored_mask
    if valid_indices.any():
        b_idx = batch_idx[valid_indices]
        n_idx = node_idx[valid_indices]
        c_idx = states_t[valid_indices]
        colors_onehot[b_idx, n_idx, c_idx] = 1.0
        
    # Propagate to neighbors: Forbidden = Adj @ Colors
    # Adj: (N, N), Colors: (B, N, K) -> we want (B, N, K)
    # Can do (N, N) x (N, B*K) -> (N, B*K) -> reshape?
    # Or batch matmul: (B, N, N) @ (B, N, K)
    
    adj_batch = adj.unsqueeze(0).expand(B, -1, -1) # (B, N, N)
    forbidden = torch.bmm(adj_batch, colors_onehot) # (B, N, K)
    
    # Valid if not forbidden AND node is uncolored
    # But wait, we want allowed actions (node, color)
    # Action is allowed if node is uncolored AND color is not forbidden
    
    # Mask: (B, N, K)
    # forbidden > 0 means conflict
    valid_colors = (forbidden == 0)
    
    # Expand uncolored mask to (B, N, K)
    node_valid = uncolored_mask.unsqueeze(-1).expand(-1, -1, K)
    
    final_mask = node_valid & valid_colors # (B, N, K)
    
    return final_mask.reshape(B, -1) # (B, N*K)


def sample_trajectories_batched(env, forward_policy, device, batch_size, epsilon=0.01, temperature=1.0, top_p=1.0):
    """
    Sample multiple trajectories in parallel using batched forward passes.
    
    Args:
        epsilon: Float or tensor (B,). Probability of uniform random exploration.
        temperature: Float. Temperature for softmax sampling (higher = more exploration).
        top_p: Float (0.0 to 1.0). Nucleus sampling threshold.
    """
    N, K = env.N, env.K
    
    # Handle epsilon as tensor if scalar
    if isinstance(epsilon, float):
        epsilon = torch.full((batch_size,), epsilon, device=device)
    elif isinstance(epsilon, torch.Tensor) and epsilon.dim() == 0:
        epsilon = epsilon.expand(batch_size)
    
    # Initialize batch of states
    states = np.stack([env.reset() for _ in range(batch_size)])  # (B, N)
    trajectories = [{'states': [s.copy()], 'actions': [], 'reward': 0.0, 'done': False} 
                    for s in states]
    
    active_mask = np.ones(batch_size, dtype=bool)
    
    while active_mask.any():
        active_indices = np.where(active_mask)[0]
        active_states = states[active_indices] # (A, N)
        A = len(active_indices)
        
        # Batch forward pass
        with torch.no_grad():
            active_states_t = torch.from_numpy(active_states).to(device=device, dtype=torch.long)
            logits_batch = forward_policy(active_states_t, env.adj, device=device) # (A, N*K)
            masks_batch = get_batched_mask(active_states, env, device).bool()      # (A, N*K)
        
        # Check stuck
        valid_counts = masks_batch.sum(dim=1) # (A,)
        
        # Identify trajectories that are done (stuck)
        stuck_rel = (valid_counts == 0).cpu().numpy()
        
        if stuck_rel.any():
            stuck_indices = active_indices[stuck_rel]
            for idx in stuck_indices:
                trajectories[idx]['done'] = True
                trajectories[idx]['reward'] = env.reward(states[idx])
                active_mask[idx] = False
            
            # Filter active for next steps
            keep_rel = ~stuck_rel
            if not keep_rel.any():
                break
                
            active_indices = active_indices[keep_rel]
            active_states = active_states[keep_rel]
            logits_batch = logits_batch[torch.from_numpy(keep_rel).to(device)]
            masks_batch = masks_batch[torch.from_numpy(keep_rel).to(device)]
            A = len(active_indices)

        # Sampling Logic
        # 1. Mask invalid actions
        logits_batch[~masks_batch] = float('-inf')
        
        # 2. Apply temperature
        probs = torch.softmax(logits_batch / temperature, dim=1) # (A, N*K)
        
        # 3. Top-P (Nucleus) Sampling
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Scatter sorted indices to original indices
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            probs[indices_to_remove] = 0.0
            probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)
        
        # 4. Sample from policy distribution
        policy_actions = torch.multinomial(probs, 1).squeeze(-1) # (A,)
        
        # 5. Sample uniformly from valid actions (for epsilon exploration)
        # Create uniform logits where mask is True
        uniform_logits = torch.zeros_like(logits_batch)
        uniform_logits[~masks_batch] = float('-inf')
        uniform_probs = torch.softmax(uniform_logits, dim=1)
        random_actions = torch.multinomial(uniform_probs, 1).squeeze(-1) # (A,)
        
        # 6. Choose based on epsilon
        # Get epsilon for active trajectories
        active_eps = epsilon[active_indices] # (A,)
        rand_draw = torch.rand(A, device=device)
        explore_mask = (rand_draw < active_eps) # (A,)
        
        final_actions = torch.where(explore_mask, random_actions, policy_actions) # (A,)
        
        # Convert to numpy for env step
        actions_np = final_actions.cpu().numpy()
        
        # Batched Step
        next_states, rewards, dones = env.step_batch(states[active_indices], actions_np)
        
        # Update global states
        states[active_indices] = next_states
        
        # Update trajectories
        for i, idx in enumerate(active_indices):
            trajectories[idx]['actions'].append(int(actions_np[i]))
            trajectories[idx]['states'].append(next_states[i].copy())
            
            if dones[i]:
                trajectories[idx]['done'] = True
                trajectories[idx]['reward'] = float(rewards[i])
                active_mask[idx] = False
    
    # Convert to list of tuples
    results = [(t['states'], t['actions'], t['reward']) for t in trajectories]
    return results

def collate_trajectories(trajectories, env, device):
    """
    trajectories: list of (states_list, actions_list, reward)
      - states_list: list of np.array shape (N,) length L+1
      - actions_list: list of ints length L
    Returns:
      states:    (T_max+1, B, N) long
      actions:   (T_max,   B)    long
      step_mask: (T_max,   B)    bool (True where step is real)
      rewards:   (B,)            float32
    """
    B = len(trajectories)
    N = env.N

    lengths = [len(actions) for (_, actions, _) in trajectories]  # steps per traj
    T_max = max(lengths)

    states = torch.full(
        (T_max + 1, B, N),
        fill_value=-1,
        dtype=torch.long,
        device=device,
    )
    actions = torch.zeros(
        (T_max, B),
        dtype=torch.long,
        device=device,
    )
    step_mask = torch.zeros(
        (T_max, B),
        dtype=torch.bool,
        device=device,
    )
    rewards = torch.zeros(B, dtype=torch.float32, device=device)

    for b, (states_list, actions_list, reward) in enumerate(trajectories):
        L = len(actions_list)
        rewards[b] = float(reward)

        # states_list is length L+1, each (N,)
        s_np = np.stack(states_list)  # (L+1, N)
        s_t = torch.from_numpy(s_np).to(device=device, dtype=torch.long)
        states[:L+1, b, :] = s_t

        if L > 0:
            a_t = torch.tensor(actions_list, dtype=torch.long, device=device)  # (L,)
            actions[:L, b] = a_t
            step_mask[:L, b] = True

    return states, actions, step_mask, rewards


# ============================================================================
# Conditional GFlowNet Sampling (for training on multiple graph instances)
# ============================================================================

def sample_trajectory_conditional(env, forward_policy, adj, device, epsilon=0.1, temperature=1.0):
    """
    Sample a single trajectory for conditional GFlowNet.
    
    Args:
        env: ConditionalGraphColoringEnv with current instance set
        forward_policy: ConditionalGNNPolicy
        adj: Adjacency matrix for current instance (numpy or tensor)
        device: torch device
        epsilon: exploration probability
        temperature: sampling temperature
        
    Returns:
        (states_list, actions_list, reward, adj)
    """
    state = env.reset()
    traj_states, traj_actions = [], []
    
    done = False
    while not done:
        traj_states.append(state.copy())
        
        with torch.no_grad():
            logits = forward_policy(state, adj, device=device)
        mask = torch.tensor(env.allowed_actions(state), dtype=torch.float32, device=device)
        
        if mask.sum() == 0:
            done = True
            reward = env.reward(state)
            break
        
        if torch.rand(1).item() < epsilon:
            valid_actions = torch.where(mask > 0)[0]
            action = valid_actions[torch.randint(len(valid_actions), (1,))].item()
        else:
            masked_logits = logits.clone()
            masked_logits[mask == 0] = float('-inf')
            probs = torch.softmax(masked_logits / temperature, dim=0)
            action = torch.multinomial(probs, 1).item()
        
        next_state, reward, done = env.step(state, action)
        traj_actions.append(action)
        state = next_state
    
    traj_states.append(state.copy())
    return traj_states, traj_actions, reward, adj


def sample_trajectories_conditional(env, forward_policy, device, batch_size, 
                                    epsilon=0.1, temperature=1.0, top_p=1.0):
    """
    Sample trajectories from multiple graph instances for conditional GFlowNet.
    
    Each trajectory is sampled from a randomly selected graph instance.
    This is the main sampling function for conditional GFlowNet training.
    
    Args:
        env: ConditionalGraphColoringEnv
        forward_policy: ConditionalGNNPolicy
        device: torch device
        batch_size: number of trajectories to sample
        epsilon: exploration probability
        temperature: sampling temperature
        top_p: nucleus sampling threshold
        
    Returns:
        List of (states_list, actions_list, reward, instance_idx, adj)
    """
    trajectories = []
    
    for _ in range(batch_size):
        # Sample a random instance
        instance_idx = env.sample_instance()
        adj = env._adj_np
        
        # Sample trajectory for this instance
        states, actions, reward, _ = sample_trajectory_conditional(
            env, forward_policy, adj, device, epsilon, temperature
        )
        
        trajectories.append((states, actions, reward, instance_idx, adj))
    
    return trajectories


def sample_trajectories_conditional_batched(env, forward_policy, device, batch_size,
                                            epsilon=0.1, temperature=1.0, top_p=1.0,
                                            same_instance=False, instance_idx=None):
    """
    Sample trajectories with batched forward passes when possible.
    
    For conditional GFlowNet, we can batch trajectories that share the same graph.
    If same_instance=True, all trajectories use the same randomly sampled instance.
    
    Args:
        env: ConditionalGraphColoringEnv
        forward_policy: ConditionalGNNPolicy
        device: torch device
        batch_size: number of trajectories
        epsilon: exploration probability
        temperature: sampling temperature
        top_p: nucleus sampling threshold
        same_instance: if True, use same instance for all trajectories in batch
        instance_idx: if provided, use this specific instance (requires same_instance=True)
        
    Returns:
        List of (states_list, actions_list, reward, instance_idx, adj)
    """
    if same_instance:
        # All trajectories use the same instance - can batch efficiently
        if instance_idx is None:
            instance_idx = env.sample_instance()
        else:
            env.set_instance(instance_idx)
        adj = env._adj_np
        N, K = env.N, env.K
        
        # Initialize batch of states
        states = np.stack([env.reset() for _ in range(batch_size)])  # (B, N)
        trajectories = [{'states': [s.copy()], 'actions': [], 'reward': 0.0, 'done': False} 
                        for s in states]
        
        active_mask = np.ones(batch_size, dtype=bool)
        
        # Handle epsilon as tensor
        if isinstance(epsilon, float):
            epsilon_t = torch.full((batch_size,), epsilon, device=device)
        else:
            epsilon_t = epsilon
        
        while active_mask.any():
            active_indices = np.where(active_mask)[0]
            active_states = states[active_indices]
            A = len(active_indices)
            
            # Batch forward pass
            with torch.no_grad():
                active_states_t = torch.from_numpy(active_states).to(device=device, dtype=torch.long)
                logits_batch = forward_policy(active_states_t, adj, device=device)
                masks_batch = get_batched_mask(active_states, env, device).bool()
            
            # Check stuck
            valid_counts = masks_batch.sum(dim=1)
            stuck_rel = (valid_counts == 0).cpu().numpy()
            
            if stuck_rel.any():
                stuck_indices = active_indices[stuck_rel]
                for idx in stuck_indices:
                    trajectories[idx]['done'] = True
                    trajectories[idx]['reward'] = env.reward(states[idx])
                    active_mask[idx] = False
                
                keep_rel = ~stuck_rel
                if not keep_rel.any():
                    break
                
                active_indices = active_indices[keep_rel]
                active_states = active_states[keep_rel]
                logits_batch = logits_batch[torch.from_numpy(keep_rel).to(device)]
                masks_batch = masks_batch[torch.from_numpy(keep_rel).to(device)]
                A = len(active_indices)
            
            # Sampling
            logits_batch[~masks_batch] = float('-inf')
            probs = torch.softmax(logits_batch / temperature, dim=1)
            
            # Top-P sampling
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                probs[indices_to_remove] = 0.0
                probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)
            
            policy_actions = torch.multinomial(probs, 1).squeeze(-1)
            
            # Uniform sampling for exploration
            uniform_logits = torch.zeros_like(logits_batch)
            uniform_logits[~masks_batch] = float('-inf')
            uniform_probs = torch.softmax(uniform_logits, dim=1)
            random_actions = torch.multinomial(uniform_probs, 1).squeeze(-1)
            
            # Epsilon-greedy
            active_eps = epsilon_t[active_indices]
            rand_draw = torch.rand(A, device=device)
            explore_mask = (rand_draw < active_eps)
            final_actions = torch.where(explore_mask, random_actions, policy_actions)
            
            actions_np = final_actions.cpu().numpy()
            
            # Step
            next_states, rewards, dones = env.step_batch(states[active_indices], actions_np)
            states[active_indices] = next_states
            
            for i, idx in enumerate(active_indices):
                trajectories[idx]['actions'].append(int(actions_np[i]))
                trajectories[idx]['states'].append(next_states[i].copy())
                
                if dones[i]:
                    trajectories[idx]['done'] = True
                    trajectories[idx]['reward'] = float(rewards[i])
                    active_mask[idx] = False
        
        # Convert to output format
        results = [(t['states'], t['actions'], t['reward'], instance_idx, adj) 
                   for t in trajectories]
        return results
    
    else:
        # Different instances - sample sequentially
        return sample_trajectories_conditional(
            env, forward_policy, device, batch_size, epsilon, temperature, top_p
        )


def collate_trajectories_conditional(trajectories, num_colors, device):
    """
    Collate trajectories from conditional GFlowNet sampling.
    
    Since trajectories may come from different-sized graphs, we need to handle
    variable sizes. This function groups trajectories by graph size for efficient
    batched processing.
    
    Args:
        trajectories: List of (states_list, actions_list, reward, instance_idx, adj)
        num_colors: K, number of colors
        device: torch device
        
    Returns:
        Dictionary with collated data grouped by graph size:
        {
            N: {
                'states': (T_max+1, B_N, N),
                'actions': (T_max, B_N),
                'step_mask': (T_max, B_N),
                'rewards': (B_N,),
                'adj': (N, N) or list of adj matrices,
                'indices': original trajectory indices
            }
        }
    """
    # Group by graph size
    by_size = {}
    for i, (states_list, actions_list, reward, instance_idx, adj) in enumerate(trajectories):
        N = len(states_list[0])
        if N not in by_size:
            by_size[N] = []
        by_size[N].append((i, states_list, actions_list, reward, adj))
    
    result = {}
    for N, group in by_size.items():
        B = len(group)
        lengths = [len(actions) for (_, _, actions, _, _) in group]
        T_max = max(lengths) if lengths else 1
        
        states = torch.full((T_max + 1, B, N), fill_value=-1, dtype=torch.long, device=device)
        actions = torch.zeros((T_max, B), dtype=torch.long, device=device)
        step_mask = torch.zeros((T_max, B), dtype=torch.bool, device=device)
        rewards = torch.zeros(B, dtype=torch.float32, device=device)
        indices = []
        adjs = []
        
        for b, (orig_idx, states_list, actions_list, reward, adj) in enumerate(group):
            L = len(actions_list)
            rewards[b] = float(reward)
            indices.append(orig_idx)
            adjs.append(adj)
            
            s_np = np.stack(states_list)
            s_t = torch.from_numpy(s_np).to(device=device, dtype=torch.long)
            states[:L+1, b, :] = s_t
            
            if L > 0:
                a_t = torch.tensor(actions_list, dtype=torch.long, device=device)
                actions[:L, b] = a_t
                step_mask[:L, b] = True
        
        # Stack adjacency matrices if all same, else keep list
        if len(set(id(a) for a in adjs)) == 1:
            # All same adjacency matrix
            adj_tensor = torch.from_numpy(adjs[0]).to(device=device, dtype=torch.float32)
        else:
            # Different adjacency matrices - stack them
            adj_tensor = torch.stack([
                torch.from_numpy(a).to(device=device, dtype=torch.float32) for a in adjs
            ])
        
        result[N] = {
            'states': states,
            'actions': actions,
            'step_mask': step_mask,
            'rewards': rewards,
            'adj': adj_tensor,
            'indices': indices,
            'K': num_colors
        }
    
    return result
