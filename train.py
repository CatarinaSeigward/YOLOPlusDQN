from game_environment import GameEnv
from dqn_agent import Agent
import keyboard
import time
import torch
import os
import json

def load_training_state(save_dir):
    """Load training state"""
    state_path = os.path.join(save_dir, 'training_state.json')
    if os.path.exists(state_path):
        with open(state_path, 'r') as f:
            return json.load(f)
    return {
        'episode': 0,
        'best_reward': -float('inf'),
        'epsilon': 1.0
    }

def save_training_state(save_dir, state):
    """Save training state"""
    state_path = os.path.join(save_dir, 'training_state.json')
    with open(state_path, 'w') as f:
        json.dump(state, f)

def train(resume=True):
    env = GameEnv()
    agent = Agent()
    episodes = 1000
    batch_size = 32
    save_dir = 'checkpoints'
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Load training state
    training_state = load_training_state(save_dir)
    start_episode = training_state['episode']
    best_reward = training_state['best_reward']
    
    # If resuming training and checkpoints exist
    if resume:
        latest_checkpoint = None
        # Find the latest checkpoint
        for filename in os.listdir(save_dir):
            if filename.startswith('checkpoint_'):
                episode_num = int(filename.split('_')[1].split('.')[0])
                if latest_checkpoint is None or episode_num > latest_checkpoint[1]:
                    latest_checkpoint = (filename, episode_num)
        
        # Load the latest checkpoint
        if latest_checkpoint:
            checkpoint_path = os.path.join(save_dir, latest_checkpoint[0])
            print(f"Resuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
            agent.target_net.load_state_dict(checkpoint['model_state_dict'])
            agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            agent.epsilon = training_state['epsilon']
    
    try:
        for episode in range(start_episode, episodes):
            print("\nPress 'o' to start new episode...")
            keyboard.wait('o')
            
            state = env.get_state()
            total_reward = 0
            episode_steps = 0
            episode_loss = 0
            n_steps = 0
            
            print(f"Starting episode {episode} with epsilon {agent.epsilon:.3f}")
            
            while True:
                if keyboard.is_pressed('q'): 
                    raise KeyboardInterrupt
                    
                if keyboard.is_pressed('r'): 
                    print("\nEpisode ended by user")
                    break
                    
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                
                agent.memory.append((state, action, reward, next_state))
                loss = agent.train(batch_size)
                if loss > 0:  
                    episode_loss += loss
                    n_steps += 1
                
                state = next_state
                total_reward += reward
                episode_steps += 1
                
                time.sleep(0.1)  # Control execution speed
                
                if done or episode_steps > 1500:
                    break
            
            print(f"Episode: {episode}, Total Reward: {total_reward}, Steps: {episode_steps}")
            
            # Update and save training state
            training_state['episode'] = episode + 1
            training_state['epsilon'] = agent.epsilon
            
            if total_reward > best_reward:
                best_reward = total_reward
                training_state['best_reward'] = best_reward
                torch.save({
                    'episode': episode,
                    'model_state_dict': agent.policy_net.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'reward': total_reward,
                }, f'{save_dir}/best_model.pth')
            
            avg_loss = episode_loss / n_steps if n_steps > 0 else 0
            agent.episode_losses.append(avg_loss)
            print(f"Episode: {episode}, Total Reward: {total_reward}, Avg Loss: {avg_loss:.4f}")
            
            # Save checkpoint every 10 episodes
            if episode % 10 == 0:
                checkpoint_path = f'{save_dir}/checkpoint_{episode}.pth'
                torch.save({
                    'episode': episode,
                    'model_state_dict': agent.policy_net.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'reward': total_reward,
                    'losses': agent.episode_losses, 
                }, checkpoint_path)
                # Save training state
                save_training_state(save_dir, training_state)
                
                # Delete old checkpoints (keep only the last two)
                checkpoints = sorted([f for f in os.listdir(save_dir) if f.startswith('checkpoint_')],
                                  key=lambda x: int(x.split('_')[1].split('.')[0]))
                for old_checkpoint in checkpoints[:-2]:
                    os.remove(os.path.join(save_dir, old_checkpoint))
            
    except KeyboardInterrupt:
        print("\nTraining interrupted, saving final state...")
        training_state['episode'] = episode
        training_state['epsilon'] = agent.epsilon
        save_training_state(save_dir, training_state)
        torch.save({
            'episode': episode,
            'model_state_dict': agent.policy_net.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'reward': total_reward,
        }, f'{save_dir}/interrupted_model.pth')

if __name__ == "__main__":
    print("Control Instructions:")
    print("- Press 'T' to start training program")
    print("- Press 'o' to start new episode")
    print("- Press 'R' to end current episode")
    print("- Press 'Q' to quit training")
    print("\nContinue previous training? (y/n)")
    resume = keyboard.read_event(suppress=True).name == 'y'
    keyboard.wait('t')
    train(resume)