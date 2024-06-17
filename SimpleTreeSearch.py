import numpy as np
import random


# Define the simple tree search function
def simple_tree_search(env, steps=10, tries=4, log_callback=None):
    best_actions = []
    best_rewards = []
    obs = env.reset()
    previous_best_state = env.save_state()
    current_best_state = None

    for step in range(steps):
        best_reward_for_step = float('-inf')
        best_action_for_step = None
        action_had_positive_reward = False  # Flag to track if any action had a positive reward

        random_actions = np.random.choice(range(24), tries, replace=False)  # Draw actions

        for action in random_actions:
            env.load_state(previous_best_state)
            _, reward, done, _, _ = env.step(action)
            #print(f"action: {action}, reward: {reward}")

            if reward > best_reward_for_step:
                best_reward_for_step = reward
                best_action_for_step = action
                current_best_state = env.save_state()

        # If no drawing action had a positive reward, choose a random move action
        if best_reward_for_step < -1.0:
            if random.random() < 0.6:
                best_action_for_step = random.randint(24, 35)  # Move action with 60% probability
                #print("MOVE")
            else:
                best_action_for_step = random.randint(0, 23)  # Random Draw action with 40% probability
                #print("RANDOM DRAW_ACTION")

            env.load_state(previous_best_state)  # Reload the best state before executing the move action
            _, best_reward_for_step, done, _, _ = env.step(best_action_for_step)  # Execute the move action
            current_best_state = env.save_state()  # Save the new state after the move action

        log_message = f"{step + 1}/{steps} ; Action: {best_action_for_step} ; Reward: {best_reward_for_step * 100:.2f}"
        if log_callback:
            log_callback(log_message)
        best_actions.append(best_action_for_step)
        best_rewards.append(best_reward_for_step)

        # Update the previous best state for the next iteration
        if current_best_state is not None:
            previous_best_state = current_best_state

    return best_actions


# Define the evaluate actions function
def evaluate_actions(env, actions):
    total_reward = 0
    env.reset()
    for action in actions:
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        
        if done:
        
            break

    return total_reward * 100




