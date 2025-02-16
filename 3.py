import numpy as np
import random
from scipy.stats import norm

def simulate_dice_rolls(n_simulations=10000):
    """Mô phỏng kết quả của ba viên xúc xắc để tính xác suất tài xỉu với số lần mô phỏng lớn hơn."""
    rolls = np.random.randint(1, 7, (n_simulations, 3)).sum(axis=1)
    return rolls

def calculate_probabilities(rolls):
    """Tính xác suất tài (11-17) và xỉu (4-10) với mô hình chính xác hơn."""
    unique, counts = np.unique(rolls, return_counts=True)
    probabilities = dict(zip(unique, counts / len(rolls)))
    tai_prob = sum(probabilities.get(i, 0) for i in range(11, 18))
    xiu_prob = sum(probabilities.get(i, 0) for i in range(4, 11))
    return tai_prob, xiu_prob

def get_user_input():
    """Nhập 5 kết quả gần nhất từ người dùng"""
    rolls = []
    for i in range(1, 6):
        while True:
            try:
                roll = int(input(f"Nhập tổng điểm phiên {i} (từ 3 đến 18): "))
                if 3 <= roll <= 18:
                    rolls.append(roll)
                    break
                else:
                    print("Vui lòng nhập số từ 3 đến 18.")
            except ValueError:
                print("Vui lòng nhập một số nguyên hợp lệ.")
    return rolls

def predict_tai_xiu():
    """Dự đoán kết quả tài xỉu dựa trên mô phỏng xác suất nâng cao và dữ liệu người dùng nhập."""
    simulated_rolls = simulate_dice_rolls()
    tai_prob_sim, xiu_prob_sim = calculate_probabilities(simulated_rolls)
    
    user_rolls = get_user_input()
    tai_prob_user, xiu_prob_user = calculate_probabilities(user_rolls)
    
    avg_tai_prob = (tai_prob_sim + tai_prob_user) / 2
    avg_xiu_prob = (xiu_prob_sim + xiu_prob_user) / 2
    
    prediction = "Tài" if avg_tai_prob > avg_xiu_prob else "Xỉu"
    confidence = abs(avg_tai_prob - avg_xiu_prob) * 100
    
    print("\nDự đoán tài xỉu dựa trên mô phỏng và dữ liệu người dùng:")
    print(f"Xác suất Tài: {avg_tai_prob:.2%}")
    print(f"Xác suất Xỉu: {avg_xiu_prob:.2%}")
    print(f"Dự đoán: {prediction}")
    print(f"Độ tin cậy: {confidence:.2f}%")
    
    return {
        "Xác suất Tài": avg_tai_prob,
        "Xác suất Xỉu": avg_xiu_prob,
        "Dự đoán": prediction,
        "Độ tin cậy": f"{confidence:.2f}%"
    }

if __name__ == "__main__":
    result = predict_tai_xiu()
