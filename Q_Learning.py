import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import time

class QLearningGridWorld:
    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.start = (0, 0)
        self.goal = (4, 4)
        self.obstacles = [(1, 1), (2, 2), (3, 1)]
        
        # Khởi tạo Q-table
        self.q_table = np.zeros((grid_size, grid_size, 4))
        # Actions: 0=up, 1=down, 2=left, 3=right
        self.actions = ['up', 'down', 'left', 'right']
        self.action_effects = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1)    # right
        }
        
        # Hyperparameters
        self.alpha = 0.1      # learning rate
        self.gamma = 0.9      # discount factor
        self.epsilon = 1.0    # exploration rate
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01
        
        # Thống kê
        self.episode = 0
        self.total_rewards = []
        self.path_lengths = []
        
    def is_valid_position(self, pos):
        """Kiểm tra vị trí hợp lệ"""
        row, col = pos
        if row < 0 or row >= self.grid_size or col < 0 or col >= self.grid_size:
            return False
        if pos in self.obstacles:
            return False
        return True
    
    def get_reward(self, pos):
        """Tính reward cho vị trí"""
        if pos == self.goal:
            return 100
        if pos in self.obstacles:
            return -100
        return -1
    
    def choose_action(self, state):
        """Chọn action bằng epsilon-greedy"""
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 4)  # exploration
        else:
            return np.argmax(self.q_table[state[0], state[1], :])  # exploitation
    
    def take_action(self, state, action):
        """Thực hiện action và trả về state mới, reward"""
        effect = self.action_effects[action]
        new_state = (state[0] + effect[0], state[1] + effect[1])
        
        if not self.is_valid_position(new_state):
            return state, -10  # hình phạt cho nước đi không hợp lệ
        
        reward = self.get_reward(new_state)
        return new_state, reward
    
    def train_episode(self):
        """Huấn luyện một episode"""
        state = self.start
        path = [state]
        total_reward = 0
        steps = 0
        max_steps = 100
        
        while state != self.goal and steps < max_steps:
            # Chọn action
            action = self.choose_action(state)
            
            # Thực hiện action
            new_state, reward = self.take_action(state, action)
            
            # Cập nhật Q-value
            current_q = self.q_table[state[0], state[1], action]
            max_next_q = np.max(self.q_table[new_state[0], new_state[1], :])
            new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
            self.q_table[state[0], state[1], action] = new_q
            
            # Chuyển sang state mới
            state = new_state
            path.append(state)
            total_reward += reward
            steps += 1
        
        # Giảm epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        self.episode += 1
        self.total_rewards.append(total_reward)
        self.path_lengths.append(len(path))
        
        return path, total_reward
    
    def get_optimal_path(self):
        """Lấy đường đi tối ưu theo chính sách đã học"""
        state = self.start
        path = [state]
        steps = 0
        max_steps = 50
        
        while state != self.goal and steps < max_steps:
            action = np.argmax(self.q_table[state[0], state[1], :])
            new_state, _ = self.take_action(state, action)
            
            if new_state == state:  # stuck
                break
                
            state = new_state
            path.append(state)
            steps += 1
        
        return path
    
    def visualize_grid(self, path=None, title="Q-Learning Grid World"):
        """Vẽ lưới với Q-values"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Subplot 1: Grid with path
        ax1.set_xlim(0, self.grid_size)
        ax1.set_ylim(0, self.grid_size)
        ax1.set_aspect('equal')
        ax1.set_title(f'{title}\nEpisode: {self.episode}, Epsilon: {self.epsilon:.3f}')
        ax1.grid(True, linewidth=0.5, alpha=0.3)
        ax1.invert_yaxis()
        
        # Vẽ grid với Q-values
        max_q = np.max(self.q_table)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Tính màu dựa trên max Q-value
                q_max = np.max(self.q_table[i, j, :])
                color_intensity = q_max / max_q if max_q > 0 else 0
                
                if (i, j) == self.goal:
                    color = 'green'
                elif (i, j) == self.start:
                    color = 'blue'
                elif (i, j) in self.obstacles:
                    color = 'black'
                else:
                    color = plt.cm.YlOrRd(color_intensity * 0.5)
                
                rect = patches.Rectangle((j, i), 1, 1, 
                                        linewidth=1, 
                                        edgecolor='gray', 
                                        facecolor=color,
                                        alpha=0.7)
                ax1.add_patch(rect)
                
                # Hiển thị text
                if (i, j) == self.start:
                    ax1.text(j+0.5, i+0.5, 'S', ha='center', va='center', 
                           fontsize=16, fontweight='bold', color='white')
                elif (i, j) == self.goal:
                    ax1.text(j+0.5, i+0.5, 'G', ha='center', va='center', 
                           fontsize=16, fontweight='bold', color='white')
        
        # Vẽ đường đi
        if path:
            path_x = [p[1] + 0.5 for p in path]
            path_y = [p[0] + 0.5 for p in path]
            ax1.plot(path_x, path_y, 'r-', linewidth=3, alpha=0.6, label='Path')
            ax1.plot(path_x[-1], path_y[-1], 'ro', markersize=15, label='Agent')
            ax1.legend()
        
        # Subplot 2: Learning curves
        if len(self.total_rewards) > 0:
            episodes = range(1, len(self.total_rewards) + 1)
            
            ax2_twin = ax2.twinx()
            
            line1 = ax2.plot(episodes, self.total_rewards, 'b-', alpha=0.6, label='Total Reward')
            line2 = ax2_twin.plot(episodes, self.path_lengths, 'r-', alpha=0.6, label='Path Length')
            
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Total Reward', color='b')
            ax2_twin.set_ylabel('Path Length', color='r')
            ax2.set_title('Learning Progress')
            ax2.grid(True, alpha=0.3)
            
            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax2.legend(lines, labels, loc='upper right')
        
        plt.tight_layout()
        return fig
    
    def train(self, episodes=100, visualize_every=10):
        """Huấn luyện với visualization"""
        print(f"Bắt đầu huấn luyện {episodes} episodes...")
        print(f"Tham số: α={self.alpha}, γ={self.gamma}, ε={self.epsilon}")
        print("-" * 60)
        
        for ep in range(episodes):
            path, reward = self.train_episode()
            
            if (ep + 1) % visualize_every == 0:
                print(f"Episode {ep+1}/{episodes} | "
                      f"Reward: {reward:6.1f} | "
                      f"Steps: {len(path):3d} | "
                      f"Epsilon: {self.epsilon:.3f}")
                
                # Visualize
                optimal_path = self.get_optimal_path()
                fig = self.visualize_grid(optimal_path, 
                                         f"Training Progress (Episode {ep+1})")
                plt.savefig(f'training_ep_{ep+1}.png', dpi=100, bbox_inches='tight')
                plt.close()
        
        print("\n" + "=" * 60)
        print("Huấn luyện hoàn tất!")
        print(f"Tổng số episodes: {self.episode}")
        print(f"Epsilon cuối cùng: {self.epsilon:.3f}")
        
        # Hiển thị kết quả cuối cùng
        optimal_path = self.get_optimal_path()
        print(f"\nĐường đi tối ưu ({len(optimal_path)} bước):")
        print(" → ".join([f"({p[0]},{p[1]})" for p in optimal_path]))
        
        fig = self.visualize_grid(optimal_path, "Final Policy")
        plt.show()
        
        return optimal_path


# ============= CHẠY DEMO =============

if __name__ == "__main__":
    # Tạo environment
    env = QLearningGridWorld(grid_size=5)
    
    print("=" * 60)
    print("Q-LEARNING: ROBOT TÌM ĐƯỜNG TRONG LƯỚI")
    print("=" * 60)
    print("\nMô tả:")
    print("- Start (S): Màu xanh dương tại (0,0)")
    print("- Goal (G): Màu xanh lá tại (4,4)")
    print("- Obstacles: Màu đen")
    print("- Q-values cao: Màu vàng/đỏ đậm hơn")
    print("\nRewards:")
    print("- Đến đích: +100")
    print("- Mỗi bước: -1")
    print("- Va vật cản: -100")
    print("- Đi ra ngoài: -10")
    print("\n" + "=" * 60 + "\n")
    
    # Huấn luyện
    optimal_path = env.train(episodes=100, visualize_every=20)
    
    # Demo chính sách đã học
    print("\n" + "=" * 60)
    print("DEMO: Chính sách đã học")
    print("=" * 60)
    
    for i, pos in enumerate(optimal_path):
        print(f"Bước {i}: {pos}")
    
    print(f"\nTổng số bước: {len(optimal_path)}")
    print(f"Đã đến đích: {'✓' if optimal_path[-1] == env.goal else '✗'}")