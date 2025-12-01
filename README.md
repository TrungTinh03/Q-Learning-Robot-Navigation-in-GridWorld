# 🧭 Q-Learning Robot Navigation in GridWorld

Dự án này mô phỏng một robot điều hướng trong môi trường GridWorld bằng thuật toán **Q-Learning**.  
Mục tiêu của robot là học cách đi từ vị trí bắt đầu (Start) đến vị trí đích (Goal) trong khi tránh chướng ngại vật.

---

## 📌 1. Giới thiệu
GridWorld là một môi trường dạng lưới (grid) 2D, trong đó mỗi ô đại diện cho một trạng thái.  
Robot có thể di chuyển theo 4 hướng:

- ⬆ Up  
- ⬇ Down  
- ⬅ Left  
- ➡ Right  

Mục tiêu của robot:

- Tối ưu hóa phần thưởng (reward) trong dài hạn  
- Tìm đường đi tốt nhất sau quá trình học lặp lại (episodes)

---

## 📌 2. Thuật toán Q-Learning
Q-Learning là một thuật toán học tăng cường (Reinforcement Learning) giúp agent học chính sách tối ưu mà **không cần mô hình môi trường**.

Công thức cập nhật Q-value:

Q(s, a) = Q(s, a) + α * (r + γ * max(Q(s’, :)) - Q(s, a))

Trong đó:
- \(α) — learning rate  
- \(γ) — discount factor  
- \(r) — reward nhận được  
- \(s -> s') — trạng thái cũ → trạng thái mới  

---

## 📁 3. Cấu trúc thư mục
Q-Learning-Robot-Navigation-in-GridWorld/  
│── main.py  
│── gridworld.py  
│── q_learning.py  
│── animation.py  
│── README.md  

## 📚 Nguồn tham khảo
- Sutton & Barto – Reinforcement Learning (2nd Edition)  
- OpenAI Gym GridWorld samples  
- Môi trường GridWorld cổ điển (RL-Classic)  

## 👤 Tác giả
- Phạm Nguyễn Trung Tính  
- Sinh viên Đại học Công Thương TP.HCM  
- Github: https://github.com/TrungTinh03  
