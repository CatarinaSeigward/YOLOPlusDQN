import matplotlib.pyplot as plt
import torch
import os
import json
import numpy as np

def visualize_training_loss(checkpoint_dir='checkpoints'):

   
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_')]
        
    latest_checkpoint = max(checkpoints, 
                          key=lambda x: int(x.split('_')[1].split('.')[0]))
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    

    checkpoint = torch.load(checkpoint_path)
    losses = checkpoint['losses']
    episodes = range(1, len(losses) + 1)
    

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, losses, 'b-', label='Average Loss per Episode')
    

    window_size = 10
    if len(losses) >= window_size:
        moving_avg = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
        plt.plot(episodes[window_size-1:], moving_avg, 'r-', 
                label=f'{window_size}-Episode Moving Average')
    

    plt.title('Training Loss Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    

    plt.savefig(os.path.join(checkpoint_dir, 'loss_curve.png'))
    plt.show()

if __name__ == "__main__":
    visualize_training_loss()