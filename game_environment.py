import numpy as np
import cv2
from ultralytics import YOLO
from grabscreen import grab_screen 
from directkeys import attack, attack3, dodge1, dodge2

class GameEnv:
    def __init__(self):
        self.model = YOLO("./boss_only.pt")
        self.action_labels = {
            0: 'boss_jump_attack', 
            1: 'boss_feet_attack', 
            2: 'boss_clap_attack', 
            3: 'boss_far_attack', 
            4: 'boss_boom_attack', 
            5: 'boss_sweep_attack', 
            6: 'boss_cry', 
            7: 'boss_down', 
            8: 'boss_stand',
            99: 'no_action'
        }
        # Action space setup
        self.actions = {
            0: attack,     # Normal attack
            1: dodge2,     
            2: attack3     # Special attack
        }
        
        #screen region and scaling settings
        self.screen_region = (1, 45, 1282, 767)
        self.resize_width = 640
        self.resize_height = 384
        
        #blood bar position
        self.boss_blood_window = (512, 654, 778, 657)  
        self.player_blood_window = (140, 700, 385, 703)
        
        #Calculate scaled blood bar positions
        self.scale_x = self.resize_width / (self.screen_region[2] - self.screen_region[0])
        self.scale_y = self.resize_height / (self.screen_region[3] - self.screen_region[1])
        
        self.boss_blood_resize = (
            int((self.boss_blood_window[0] - self.screen_region[0]) * self.scale_x),
            int((self.boss_blood_window[1] - self.screen_region[1]) * self.scale_y),
            int((self.boss_blood_window[2] - self.screen_region[0]) * self.scale_x),
            int((self.boss_blood_window[3] - self.screen_region[1]) * self.scale_y)
        )
        
        self.player_blood_resize = (
            int((self.player_blood_window[0] - self.screen_region[0]) * self.scale_x),
            int((self.player_blood_window[1] - self.screen_region[1]) * self.scale_y),
            int((self.player_blood_window[2] - self.screen_region[0]) * self.scale_x),
            int((self.player_blood_window[3] - self.screen_region[1]) * self.scale_y)
        )
        
        self.previous_boss_blood = 0
        self.previous_player_blood = 0
        
        # Create windows for visualization
        cv2.namedWindow('Game Screen', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Boss Blood', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Player Blood', cv2.WINDOW_NORMAL)
    
    def draw_detection_regions(self, screen_resized):
        vis_image = screen_resized.copy()
        

        cv2.rectangle(vis_image, 
                     (self.boss_blood_resize[0], self.boss_blood_resize[1]),
                     (self.boss_blood_resize[2], self.boss_blood_resize[3]),
                     (0, 0, 255), 2)
        
        cv2.rectangle(vis_image, 
                     (self.player_blood_resize[0], self.player_blood_resize[1]),
                     (self.player_blood_resize[2], self.player_blood_resize[3]),
                     (0, 255, 0), 2)
        
        cv2.putText(vis_image, f'Boss Blood: {self.previous_boss_blood}',
                    (self.boss_blood_resize[0], self.boss_blood_resize[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        cv2.putText(vis_image, f'Player Blood: {self.previous_player_blood}',
                    (self.player_blood_resize[0], self.player_blood_resize[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return vis_image

    def visualize_blood_detection(self, boss_blood_image, player_blood_image, boss_gray, player_gray):

        # Create color maps for grayscale images
        boss_gray_colored = cv2.applyColorMap(boss_gray, cv2.COLORMAP_JET)
        player_gray_colored = cv2.applyColorMap(player_gray, cv2.COLORMAP_JET)
        
        # Stack original and processed images
        boss_vis = np.vstack([
            cv2.resize(boss_blood_image, (200, 50)),
            cv2.resize(boss_gray_colored, (200, 50))
        ])
        
        player_vis = np.vstack([
            cv2.resize(player_blood_image, (200, 50)),
            cv2.resize(player_gray_colored, (200, 50))
        ])
        
        return boss_vis, player_vis
    
    def get_state(self):
        screen = grab_screen(self.screen_region)
        screen = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)
        
        #Resize
        screen_resized = cv2.resize(screen, (self.resize_width, self.resize_height))
        
        #Extract blood regions
        boss_blood_image = screen_resized[
            self.boss_blood_resize[1]:self.boss_blood_resize[3],
            self.boss_blood_resize[0]:self.boss_blood_resize[2]
        ]
        
        player_blood_image = screen_resized[
            self.player_blood_resize[1]:self.player_blood_resize[3],
            self.player_blood_resize[0]:self.player_blood_resize[2]
        ]
        
        # Convert to grayscale
        boss_gray = cv2.cvtColor(boss_blood_image, cv2.COLOR_BGR2GRAY)
        player_gray = cv2.cvtColor(player_blood_image, cv2.COLOR_BGR2GRAY)
        
        # Get blood values
        boss_blood = self.get_boss_blood(boss_gray)
        player_blood = self.get_player_blood(player_gray)
        
        # Create visualizations
        vis_image = self.draw_detection_regions(screen_resized)
        boss_vis, player_vis = self.visualize_blood_detection(
            boss_blood_image, player_blood_image, boss_gray, player_gray)
        
        # Display visualizations
        cv2.imshow('Game Screen', vis_image)
        cv2.imshow('Boss Blood', boss_vis)
        cv2.imshow('Player Blood', player_vis)
        cv2.waitKey(1)
        
        # YOLO detection and state composition
        results = self.model.predict(
            source=screen_resized,
            conf=0.4,
            iou=0.5,
            imgsz=(self.resize_width, self.resize_height),
            stream=True
        )
        results = list(results)
        
        state = []
        if len(results) > 0 and len(results[0].boxes) > 0:
            box = results[0].boxes[0]
            state.extend([int(box.cls.item()), float(box.conf.item())])
        else:
            state.extend([99, 0.0])
            
        state.extend([boss_blood, player_blood])
        return np.array(state)

    def get_boss_blood(self, boss_gray):
        height = boss_gray.shape[0]
        middle_row = boss_gray[height // 2, :]
        
        threshold = 145
        boss_blood = np.sum(middle_row > threshold)
        
        if boss_blood < 10:
            boss_blood = self.previous_boss_blood
        else:
            self.previous_boss_blood = boss_blood
            
        return boss_blood

    def get_player_blood(self, player_gray):
        height = player_gray.shape[0]
        middle_row = player_gray[height // 2, :]
        
        threshold = 147
        player_blood = np.sum(middle_row > threshold)
        
        if player_blood < 10:
            player_blood = self.previous_player_blood
        else:
            self.previous_player_blood = player_blood
            
        return player_blood

    def step(self, action):
        initial_boss_blood = self.previous_boss_blood
        initial_player_blood = self.previous_player_blood
        
        self.actions[action]()
        
        next_state = self.get_state()
        boss_blood = next_state[2]
        player_blood = next_state[3]
        
        boss_blood_change = initial_boss_blood - boss_blood
        player_blood_change = initial_player_blood - player_blood
        
        reward = 0
        if boss_blood_change > 0:
            reward += boss_blood_change * 30
        if player_blood_change > 0:
            reward -= player_blood_change * 5
            
        self.previous_boss_blood = boss_blood
        self.previous_player_blood = player_blood
        
        done = boss_blood <= 0 or player_blood <= 0
        
        return next_state, reward, done

    def __del__(self):
        cv2.destroyAllWindows()