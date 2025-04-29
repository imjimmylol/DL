next_q_values, _ = target_net(next_states).max(dim=1)
``` | Decouples selection (policy_net) from evaluation (target_net):  
```python
next_actions = policy_net(next_states).argmax(dim=1)
next_q_values = target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
``` |
| **Target formulation** |  
```python
expected_q = rewards + gamma * next_q_values.detach()
``` |  
```python
expected_q = rewards + gamma * next_q_values.detach()
``` |
| **Loss**            |  
```python
loss = F.mse_loss(curr_q, expected_q.unsqueeze(1))
``` |  
```python
loss = F.mse_loss(curr_q, expected_q.unsqueeze(1))
``` |

---

### 1. Vanilla DQN (with Target Network)
```python
# 1. Sample batch
states, actions, rewards, next_states, dones = batch

# 2. Current Q-values
q_pred = policy_net(states)                          # [batch, num_actions]
curr_q = q_pred.gather(1, actions.unsqueeze(1))     # [batch,1]

# 3. Next Q-values (max over actions, using target_net)
next_q_values, _ = target_net(next_states).max(dim=1)  # [batch]
next_q_values = next_q_values * (1 - dones)           # zero if terminal

# 4. Compute target
expected_q = rewards + gamma * next_q_values.detach() # [batch]

# 5. Loss and optimize
loss = F.mse_loss(curr_q, expected_q.unsqueeze(1))
optimizer.zero_grad()
loss.backward()
optimizer.step()
