import torch

def sample_trajectory(env, forward_policy, device):
    state = env.reset()
    traj_states, traj_actions = [], []

    done = False
    while not done:
        traj_states.append(state.copy())   # state before action

        logits = forward_policy(state, env.adj, device=device)

        mask = torch.tensor(env.allowed_actions(state), dtype=torch.float32).to(device)
        masked_logits = logits + (mask + 1e-8).log()

        probs = torch.softmax(masked_logits, dim=0)
        action = torch.multinomial(probs, 1).item()

        next_state, reward, done = env.step(state, action)
        traj_actions.append(action)
        state = next_state

    # append terminal state
    traj_states.append(state.copy())

    return traj_states, traj_actions, reward
