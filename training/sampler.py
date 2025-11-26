import torch
import random

def sample_trajectory(env, forward_policy, device, epsilon=0.1):
    """Sample a trajectory with epsilon-greedy exploration."""
    state = env.reset()
    traj_states, traj_actions = [], []

    done = False
    while not done:
        traj_states.append(state.copy())   # state before action

        logits = forward_policy(state, env.adj, device=device)

        mask = torch.tensor(env.allowed_actions(state), dtype=torch.float32).to(device)
        
        # Check if any valid actions exist
        if mask.sum() == 0:
            # No valid actions - stuck state, get reward for partial coloring
            done = True
            reward = env.reward(state)
            break
        
        # Epsilon-greedy: with probability epsilon, choose random valid action
        if random.random() < epsilon:
            valid_actions = torch.where(mask > 0)[0]
            action = valid_actions[torch.randint(len(valid_actions), (1,))].item()
        else:
            # Use -inf for invalid actions to ensure zero probability
            masked_logits = logits.clone()
            masked_logits[mask == 0] = float('-inf')
            probs = torch.softmax(masked_logits, dim=0)
            action = torch.multinomial(probs, 1).item()

        next_state, reward, done = env.step(state, action)
        traj_actions.append(action)
        state = next_state

    # append terminal state
    traj_states.append(state.copy())

    return traj_states, traj_actions, reward
