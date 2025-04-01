import random

# Hàm yêu cầu người dùng nhập lịch sử kết quả
def get_history_from_js():
    history = []
    print("Nhập 7 kết quả gần nhất của tài/xỉu (0 = Xỉu, 1 = Tài):")
    for i in range(1, 8):
        while True:
            data = input(f"Nhập kết quả {i}/7 (0 hoặc 1): ")
            if data in ['0', '1']:
                history.append(int(data))
                break
            else:
                print("Dữ liệu không hợp lệ. Vui lòng nhập 0 hoặc 1.")
    return history

# Hàm phát hiện mẫu trong dữ liệu lịch sử
def detect_pattern(history):
    if len(history) < 7:
        return None
    
    last_7 = ''.join(map(str, history[-7:]))
    common_patterns = {
        "1111111": "Xỉu", "0000000": "Tài", 
        "1010101": "Tài", "0101010": "Xỉu", 
        "1100110": "Xỉu", "0011001": "Tài", 
        "1110001": "Xỉu", "0001110": "Tài",
        "1001001": "Xỉu", "0110110": "Tài", 
        "1101011": "Xỉu", "0010100": "Tài", 
        "1110011": "Xỉu", "0001100": "Tài", 
        "1011101": "Xỉu", "0100010": "Tài",
        "1001010": "Xỉu", "0110101": "Tài",
        "1101100": "Xỉu", "0010011": "Tài",
        "1010111": "Xỉu", "0101000": "Tài",
        "1110100": "Xỉu", "0001011": "Tài",
        "1001101": "Xỉu", "0110010": "Tài",
        "1101001": "Xỉu", "0010110": "Tài",
        "1010010": "Xỉu", "0101101": "Tài"
    }
    return common_patterns.get(last_7, None)

# Hàm dự đoán tài xỉu
def predict_tai_xiu(history):
    if not history:
        return random.choice(["Tài", "Xỉu"])
    
    # Kiểm tra mẫu xuất hiện
    pattern_prediction = detect_pattern(history)
    if pattern_prediction:
        return pattern_prediction
    
    # Tính toán xác suất
    last_results = history[-7:]
    probability = sum(last_results) / len(last_results)
    
    # Kiểm tra chuỗi liên tục
    if history[-5:].count(1) == 5:
        return "Xỉu"
    elif history[-5:].count(0) == 5:
        return "Tài"
    
    return "Tài" if probability > 0.55 else "Xỉu"

# Chạy chương trình
if __name__ == "__main__":
    history = get_history_from_js()
    prediction = predict_tai_xiu(history)
    print(f"Dự đoán kết quả tiếp theo: {prediction}")
    print(f"Lịch sử gần đây: {history}")
    print("Tool by @vsk111")
