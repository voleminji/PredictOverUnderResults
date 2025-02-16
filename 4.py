import numpy as np

def simulate_dice_rolls(n_simulations=10000):
    """
    Mô phỏng kết quả của ba viên xúc xắc để tính xác suất Tài Xỉu.
    """
    rolls = np.random.randint(1, 7, (n_simulations, 3)).sum(axis=1)
    return rolls

def calculate_probabilities(rolls):
    """
    Tính xác suất Tài (tổng 11-17) và Xỉu (tổng 4-10) từ danh sách kết quả.
    """
    unique, counts = np.unique(rolls, return_counts=True)
    probabilities = dict(zip(unique, counts / len(rolls)))
    tai_prob = sum(probabilities.get(i, 0) for i in range(11, 18))
    xiu_prob = sum(probabilities.get(i, 0) for i in range(4, 11))
    return tai_prob, xiu_prob

def get_user_input():
    """
    Nhập 5 kết quả gần nhất từ người dùng (1 là tay mới nhất, 5 là tay xa nhất).
    """
    rolls = []
    for i in range(1, 6):
        while True:
            try:
                roll = int(input(f"Nhập tổng điểm phiên {i} (tay mới nhất là 1, tay xa nhất là 5, từ 3 đến 18): "))
                if 3 <= roll <= 18:
                    rolls.append(roll)
                    break
                else:
                    print("Vui lòng nhập số từ 3 đến 18.")
            except ValueError:
                print("Vui lòng nhập một số nguyên hợp lệ.")
    return rolls

def calculate_user_probabilities(rolls):
    """
    Tính xác suất Tài và Xỉu từ dữ liệu người dùng nhập.
    - Tài: tổng từ 11 đến 17
    - Xỉu: tổng từ 4 đến 10
    """
    tai_count = sum(1 for x in rolls if 11 <= x <= 17)
    xiu_count = sum(1 for x in rolls if 4 <= x <= 10)
    total = len(rolls)
    tai_prob = tai_count / total
    xiu_prob = xiu_count / total
    return tai_prob, xiu_prob

def exact_probabilities():
    """
    Tính xác suất lý thuyết của tổng điểm từ 3 viên xúc xắc qua phân tích tổ hợp.
    Trả về:
        probabilities: dict, chứa xác suất của từng tổng (3 đến 18)
        tai_prob: xác suất Tài (tổng 11 đến 17)
        xiu_prob: xác suất Xỉu (tổng 4 đến 10)
    """
    outcomes = [i + j + k for i in range(1, 7)
                        for j in range(1, 7)
                        for k in range(1, 7)]
    total = 216  # 6^3
    counts = {s: outcomes.count(s) for s in range(3, 19)}
    probabilities = {s: count / total for s, count in counts.items()}
    # Theo quy tắc thông dụng: tính Tài (11-17) và Xỉu (4-10), bỏ qua 3 và 18
    tai_prob = sum(probabilities[s] for s in range(11, 18))
    xiu_prob = sum(probabilities[s] for s in range(4, 11))
    return probabilities, tai_prob, xiu_prob

def predict_tai_xiu():
    """
    Dự đoán Tài Xỉu bằng cách kết hợp:
    1. Xác suất lý thuyết (phân tích tổ hợp).
    2. Xác suất mô phỏng (với số lần mô phỏng do người dùng nhập).
    3. Xác suất từ dữ liệu người dùng nhập (5 phiên gần nhất).
    Sau đó, kết hợp theo trung bình cộng để đưa ra dự đoán cuối cùng.
    """
    # 1. Xác suất lý thuyết
    probabilities, tai_prob_theory, xiu_prob_theory = exact_probabilities()
    print("Kết quả xác suất lý thuyết từ phân tích tổ hợp (bỏ qua 3 và 18):")
    print(f"Xác suất Tài (11-17): {tai_prob_theory:.2%}")
    print(f"Xác suất Xỉu (4-10): {xiu_prob_theory:.2%}\n")
    
    # 2. Xác suất mô phỏng: cho phép người dùng nhập số lần mô phỏng
    while True:
        try:
            n_simulations = int(input("Nhập số lần mô phỏng (ví dụ: 1000000): "))
            if n_simulations > 0:
                break
            else:
                print("Vui lòng nhập một số nguyên dương.")
        except ValueError:
            print("Vui lòng nhập một số nguyên hợp lệ.")
    
    simulated_rolls = simulate_dice_rolls(n_simulations)
    tai_prob_sim, xiu_prob_sim = calculate_probabilities(simulated_rolls)
    print("\nKết quả xác suất từ mô phỏng:")
    print(f"Xác suất Tài (11-17): {tai_prob_sim:.2%}")
    print(f"Xác suất Xỉu (4-10): {xiu_prob_sim:.2%}\n")
    
    # 3. Xác suất từ dữ liệu người dùng nhập (5 phiên gần nhất)
    user_rolls = get_user_input()
    tai_prob_user, xiu_prob_user = calculate_user_probabilities(user_rolls)
    print("\nKết quả xác suất từ dữ liệu người dùng nhập:")
    print(f"Xác suất Tài: {tai_prob_user:.2%}")
    print(f"Xác suất Xỉu: {xiu_prob_user:.2%}\n")
    
    # Kết hợp ba nguồn: trung bình cộng
    avg_tai_prob = (tai_prob_theory + tai_prob_sim + tai_prob_user) / 3
    avg_xiu_prob = (xiu_prob_theory + xiu_prob_sim + xiu_prob_user) / 3
    
    prediction = "Tài" if avg_tai_prob > avg_xiu_prob else "Xỉu"
    confidence = abs(avg_tai_prob - avg_xiu_prob) * 100

    print("Dự đoán Tài Xỉu kết hợp:")
    print(f"Xác suất Tài: {avg_tai_prob:.2%}")
    print(f"Xác suất Xỉu: {avg_xiu_prob:.2%}")
    print(f"Dự đoán: {prediction}")
    print(f"Độ tin cậy: {confidence:.2f}%")
    
    return {
        "Xác suất Tài (lý thuyết)": tai_prob_theory,
        "Xác suất Xỉu (lý thuyết)": xiu_prob_theory,
        "Xác suất Tài (mô phỏng)": tai_prob_sim,
        "Xác suất Xỉu (mô phỏng)": xiu_prob_sim,
        "Xác suất Tài (người dùng)": tai_prob_user,
        "Xác suất Xỉu (người dùng)": xiu_prob_user,
        "Xác suất Tài (trung bình)": avg_tai_prob,
        "Xác suất Xỉu (trung bình)": avg_xiu_prob,
        "Dự đoán": prediction,
        "Độ tin cậy": f"{confidence:.2f}%"
    }

if __name__ == "__main__":
    predict_tai_xiu()
