# problems/tsp/sampler.py

import torch
import numpy as np


def sample_trajectory(env, forward_policy, device, epsilon=0.1):
    """Sample a single trajectory with epsilon-greedy exploration."""
    state = env.reset()
    traj_states, traj_actions = [], []

    done = False
    while not done:
        traj_states.append(state.copy())

        logits = forward_policy(state, device=device)
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
            probs = torch.softmax(masked_logits, dim=0)
            action = torch.multinomial(probs, 1).item()

        next_state, reward, done = env.step(state, action)
        traj_actions.append(action)
        state = next_state

    traj_states.append(state.copy())
    return traj_states, traj_actions, reward


def get_batched_mask_tsp(states, env, device):
    """
    Compute allowed actions mask for a batch of states.
    
    Args:
        states: (B, N) numpy array or tensor
        env: TSPEnv
        device: torch device
        
    Returns:
        mask: (B, N) boolean tensor
    """
    if isinstance(states, np.ndarray):
        states_t = torch.from_numpy(states).to(device=device, dtype=torch.long)
    else:
        states_t = states.to(dtype=torch.long)
    
    B, N = states_t.shape
    
    # Unvisited cities are valid actions
    mask = (states_t == -1)
    
    return mask


def sample_trajectories_batched(env, forward_policy, device, batch_size, 
                                 epsilon=0.1, temperature=1.0, top_p=1.0):
    """
    Sample multiple trajectories in parallel using batched processing.
    """
    N = env.N
    
    # Handle epsilon
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
        active_states = states[active_indices]
        A = len(active_indices)
        
        with torch.no_grad():
            active_states_t = torch.from_numpy(active_states).to(device=device, dtype=torch.long)
            logits_batch = forward_policy(active_states_t, device=device)
            masks_batch = get_batched_mask_tsp(active_states, env, device)
        
        # Check for stuck states (no valid actions)
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
        
        # Sampling Logic
        logits_batch[~masks_batch] = float('-inf')
        probs = torch.softmax(logits_batch / temperature, dim=1)
        
        # Top-P Sampling
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
        
        # Uniform random actions
        uniform_logits = torch.zeros_like(logits_batch)
        uniform_logits[~masks_batch] = float('-inf')
        uniform_probs = torch.softmax(uniform_logits, dim=1)
        random_actions = torch.multinomial(uniform_probs, 1).squeeze(-1)
        
        # Choose based on epsilon
        active_eps = epsilon[active_indices]
        rand_draw = torch.rand(A, device=device)
        explore_mask = (rand_draw < active_eps)
        
        final_actions = torch.where(explore_mask, random_actions, policy_actions)
        actions_np = final_actions.cpu().numpy()
        
        # Take step
        next_states, rewards, dones = env.step_batch(states[active_indices], actions_np)
        states[active_indices] = next_states
        
        for i, idx in enumerate(active_indices):
            trajectories[idx]['actions'].append(int(actions_np[i]))
            trajectories[idx]['states'].append(next_states[i].copy())
            
            if dones[i]:
                trajectories[idx]['done'] = True
                trajectories[idx]['reward'] = float(rewards[i])
                active_mask[idx] = False
    
    results = [(t['states'], t['actions'], t['reward']) for t in trajectories]
    return results


def collate_trajectories(trajectories, env, device):
    """
    Collate a list of trajectories into padded tensors.
    """
    B = len(trajectories)
    N = env.N
    
    lengths = [len(actions) for (_, actions, _) in trajectories]
    T_max = max(lengths)
    
    states = torch.full((T_max + 1, B, N), -1, dtype=torch.long, device=device)
    actions = torch.zeros((T_max, B), dtype=torch.long, device=device)
    step_mask = torch.zeros((T_max, B), dtype=torch.bool, device=device)
    rewards = torch.zeros(B, dtype=torch.float32, device=device)
    
    for b, (states_list, actions_list, reward) in enumerate(trajectories):
        L = len(actions_list)
        rewards[b] = float(reward)
        
        s_np = np.stack(states_list)
        states[:L+1, b] = torch.from_numpy(s_np).to(device=device, dtype=torch.long)
        
        if L > 0:
            actions[:L, b] = torch.tensor(actions_list, dtype=torch.long, device=device)
            step_mask[:L, b] = True
            
    return states, actions, step_mask, rewards


# ============================================================================
# Conditional Sampling Functions
# ============================================================================

def sample_trajectory_conditional(env, forward_policy, device, instance_idx=None,
                                   epsilon=0.1, temperature=1.0, top_p=1.0):
    """
    Sample a single trajectory for conditional GFlowNet.
    """
    state, instance_idx, instance = env.reset(instance_idx)
    coords = instance['coords']
    distance_matrix = instance['distance_matrix']
    
    traj_states = [state.copy()]
    traj_actions = []
    
    done = False
    while not done:
        with torch.no_grad():
            logits = forward_policy(state, coords, distance_matrix, device=device)
            mask = torch.tensor(env.allowed_actions(state, instance_idx),
                              dtype=torch.float32, device=device)
        
        if mask.sum() == 0:
            done = True
            reward = env.reward(state, instance_idx)
            break
        
        if torch.rand(1).item() < epsilon:
            valid_actions = torch.where(mask > 0)[0]
            action = valid_actions[torch.randint(len(valid_actions), (1,))].item()
        else:
            masked_logits = logits.clone()
            masked_logits[mask == 0] = float('-inf')
            probs = torch.softmax(masked_logits / temperature, dim=0)
            
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=0)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                probs[indices_to_remove] = 0.0
                probs = probs / (probs.sum() + 1e-8)
            
            action = torch.multinomial(probs, 1).item()
        
        next_state, reward, done = env.step(state, action, instance_idx)
        traj_actions.append(action)
        state = next_state
        traj_states.append(state.copy())
    
    return traj_states, traj_actions, reward, instance_idx, instance


def sample_trajectories_conditional(env, forward_policy, device, batch_size,
                                    epsilon=0.1, temperature=1.0, top_p=1.0):
    """
    Sample multiple trajectories sequentially for conditional GFlowNet.
    """
    trajectories = []
    for _ in range(batch_size):
        traj = sample_trajectory_conditional(
            env, forward_policy, device,
            instance_idx=None,
            epsilon=epsilon, temperature=temperature, top_p=top_p
        )
        trajectories.append(traj)
    return trajectories


def sample_trajectories_conditional_batched(env, forward_policy, device, batch_size,
                                            epsilon=0.1, temperature=1.0, top_p=1.0,
                                            same_instance=True):
    """
    Sample multiple trajectories in parallel for conditional GFlowNet.
    """
    if same_instance:
        instance_idx = np.random.randint(env.num_instances)
        instance = env.get_instance(instance_idx)
        coords = instance['coords']
        distance_matrix = instance['distance_matrix']
        N = instance['N']
        
        if isinstance(epsilon, float):
            epsilon_t = torch.full((batch_size,), epsilon, device=device)
        else:
            epsilon_t = epsilon
        
        states = np.stack([env.envs[instance_idx].reset() for _ in range(batch_size)])
        trajectories = [{'states': [s.copy()], 'actions': [], 'reward': 0.0, 'done': False}
                       for s in states]
        
        active_mask = np.ones(batch_size, dtype=bool)
        
        while active_mask.any():
            active_indices = np.where(active_mask)[0]
            active_states = states[active_indices]
            A = len(active_indices)
            
            with torch.no_grad():
                active_states_t = torch.from_numpy(active_states).to(device=device, dtype=torch.long)
                logits_batch = forward_policy(active_states_t, coords, distance_matrix, device=device)
                masks_batch = get_batched_mask_tsp_conditional(active_states, device)
            
            valid_counts = masks_batch.sum(dim=1)
            stuck_rel = (valid_counts == 0).cpu().numpy()
            
            if stuck_rel.any():
                stuck_indices = active_indices[stuck_rel]
                for idx in stuck_indices:
                    trajectories[idx]['done'] = True
                    trajectories[idx]['reward'] = env.reward(states[idx], instance_idx)
                    active_mask[idx] = False
                
                keep_rel = ~stuck_rel
                if not keep_rel.any():
                    break
                
                active_indices = active_indices[keep_rel]
                active_states = active_states[keep_rel]
                logits_batch = logits_batch[torch.from_numpy(keep_rel).to(device)]
                masks_batch = masks_batch[torch.from_numpy(keep_rel).to(device)]
                A = len(active_indices)
            
            logits_batch[~masks_batch] = float('-inf')
            probs = torch.softmax(logits_batch / temperature, dim=1)
            
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
            
            uniform_logits = torch.zeros_like(logits_batch)
            uniform_logits[~masks_batch] = float('-inf')
            uniform_probs = torch.softmax(uniform_logits, dim=1)
            random_actions = torch.multinomial(uniform_probs, 1).squeeze(-1)
            
            active_eps = epsilon_t[active_indices]
            rand_draw = torch.rand(A, device=device)
            explore_mask = (rand_draw < active_eps)
            
            final_actions = torch.where(explore_mask, random_actions, policy_actions)
            actions_np = final_actions.cpu().numpy()
            
            for i, idx in enumerate(active_indices):
                action = int(actions_np[i])
                next_state, reward, done = env.step(states[idx], action, instance_idx)
                states[idx] = next_state
                
                trajectories[idx]['actions'].append(action)
                trajectories[idx]['states'].append(next_state.copy())
                
                if done:
                    trajectories[idx]['done'] = True
                    trajectories[idx]['reward'] = float(reward)
                    active_mask[idx] = False
        
        results = [(t['states'], t['actions'], t['reward'], instance_idx, instance)
                   for t in trajectories]
        return results
    else:
        return sample_trajectories_conditional(
            env, forward_policy, device, batch_size,
            epsilon=epsilon, temperature=temperature, top_p=top_p
        )


def get_batched_mask_tsp_conditional(states, device):
    """
    Compute allowed actions mask for a batch of states.
    """
    if isinstance(states, np.ndarray):
        states_t = torch.from_numpy(states).to(device=device, dtype=torch.long)
    else:
        states_t = states.to(dtype=torch.long)
    
    mask = (states_t == -1)
    return mask


def collate_trajectories_conditional(trajectories, device):
    """
    Collate conditional trajectories grouped by instance size.
    """
    by_size = {}
    for traj in trajectories:
        states_list, actions_list, reward, instance_idx, instance = traj
        N = instance['N']
        
        if N not in by_size:
            by_size[N] = []
        by_size[N].append(traj)
    
    collated = {}
    for N, group in by_size.items():
        B = len(group)
        lengths = [len(actions) for (_, actions, _, _, _) in group]
        T_max = max(lengths) if lengths else 1
        
        states = torch.full((T_max + 1, B, N), -1, dtype=torch.long, device=device)
        actions = torch.zeros((T_max, B), dtype=torch.long, device=device)
        step_mask = torch.zeros((T_max, B), dtype=torch.bool, device=device)
        rewards = torch.zeros(B, dtype=torch.float32, device=device)
        instance_indices = []
        
        first_instance = group[0][4]
        coords = torch.tensor(first_instance['coords'], dtype=torch.float32, device=device)
        distance_matrix = torch.tensor(first_instance['distance_matrix'], dtype=torch.float32, device=device)
        
        for b, (states_list, actions_list, reward, instance_idx, instance) in enumerate(group):
            L = len(actions_list)
            rewards[b] = float(reward)
            instance_indices.append(instance_idx)
            
            s_np = np.stack(states_list)
            states[:L+1, b] = torch.from_numpy(s_np).to(device=device, dtype=torch.long)
            
            if L > 0:
                actions[:L, b] = torch.tensor(actions_list, dtype=torch.long, device=device)
                step_mask[:L, b] = True
        
        collated[N] = {
            'states': states,
            'actions': actions,
            'step_mask': step_mask,
            'rewards': rewards,
            'coords': coords,
            'distance_matrix': distance_matrix,
            'instance_indices': instance_indices,
        }
    
    return collated
