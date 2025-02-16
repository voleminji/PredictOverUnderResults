import numpy as np
import random
from scipy.stats import norm

def simulate_dice_rolls(n_simulations=1000000):
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

def predict_tai_xiu():
    """Dự đoán kết quả tài xỉu dựa trên mô phỏng xác suất nâng cao."""
    rolls = simulate_dice_rolls()
    tai_prob, xiu_prob = calculate_probabilities(rolls)
    prediction = "Tài" if tai_prob > xiu_prob else "Xỉu"
    confidence = abs(tai_prob - xiu_prob) * 100
    return {
        "Xác suất Tài": tai_prob,
        "Xác suất Xỉu": xiu_prob,
        "Dự đoán": prediction,
        "Độ tin cậy": f"{confidence:.2f}%"
    }

if __name__ == "__main__":
    result = predict_tai_xiu()
    print("Dự đoán tài xỉu:")
    print(f"Xác suất Tài: {result['Xác suất Tài']:.2%}")
    print(f"Xác suất Xỉu: {result['Xác suất Xỉu']:.2%}")
    print(f"Dự đoán: {result['Dự đoán']}")
    print(f"Độ tin cậy: {result['Độ tin cậy']}")
