# training/trainer.py

import torch
from training.sampler import sample_trajectory

def train(env, forward_policy, backward_policy, loss_fn, optimizer,
          steps=2000, device="cpu"):

    last_terminal_state = None

    for step in range(steps):
        traj_states, traj_actions, reward = sample_trajectory(env, forward_policy, device)
        last_terminal_state = traj_states[-1]

        #forward
        logprobs_f = 0
        for i in range(len(traj_actions)): 
            state_i = traj_states[i]
            logits = forward_policy(state_i, env.adj, device=device)

            mask = torch.tensor(env.allowed_actions(state_i), dtype=torch.float32).to(device)
            masked = logits + (mask + 1e-8).log()
            probs = torch.softmax(masked, dim=0)

            logprobs_f += torch.log(probs[traj_actions[i]] + 1e-8)

        #backward
        logprobs_b = 0
        for i in reversed(range(len(traj_actions))):
            state_i = traj_states[i]
            logits = backward_policy(state_i, env.adj, device=device)

            mask = torch.tensor(env.allowed_actions(state_i), dtype=torch.float32).to(device)
            masked = logits + (mask + 1e-8).log()
            probs = torch.softmax(masked, dim=0)

            logprobs_b += torch.log(probs[traj_actions[i]] + 1e-8)

        # log reward and TB loss
        logreward = torch.log(torch.tensor([reward], dtype=torch.float32) + 1e-8)
        loss = loss_fn(logprobs_f, logprobs_b, logreward)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 200 == 0:
            print(f"[{step}] loss={loss.item():.4f}, reward={reward:.4f}")

    return last_terminal_state
