from game_environment import GameEnv
from dqn_agent import Agent
import torch
import keyboard
import time
import cv2

def test(model_path='checkpoints/best_model.pth'):
    env = GameEnv()
    agent = Agent()
    
    # 加载训练好的模型
    checkpoint = torch.load(model_path)
    agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
    agent.epsilon = 0.0  #
    
    print(f"Loading model from episode {checkpoint.get('episode', 'unknown')}")
    print(f"Model reward: {checkpoint.get('reward', 'unknown')}")
    
    try:
        print("按T开始测试，按Q退出")
        keyboard.wait('t')
        
        total_reward = 0
        state = env.get_state()
        steps = 0
        
        while True:
            if keyboard.is_pressed('q'):
                break
                
            # 获取模型预测的动作
            action = agent.act(state)
            
            # 执行动作
            next_state, reward, done = env.step(action)
            
            # 显示当前状态
            screen = env.get_screen()  # 需要在GameEnv中添加此方法
            cv2.imshow('Game State', screen)
            cv2.waitKey(1)
            
            total_reward += reward
            state = next_state
            steps += 1
            
            print(f"Step: {steps}, Action: {action}, Reward: {reward}, Total Reward: {total_reward}")
            
            if done:
                print("Boss defeated!" if total_reward > 0 else "Failed to defeat boss")
                break
                
            time.sleep(0.1)  # 控制执行速度
            
    except KeyboardInterrupt:
        print("Testing interrupted")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test()