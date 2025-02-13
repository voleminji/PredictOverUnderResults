import random

# Hàm yêu cầu người dùng nhập lịch sử kết quả
def get_history_from_js():
    history = []
    print("Nhập 12 kết quả gần nhất của tài/xỉu (0 = Xỉu, 1 = Tài):")
    for i in range(1, 13):
        while True:
            data = input(f"Nhập kết quả {i}/12 (0 hoặc 1): ")
            if data in ['0', '1']:
                history.append(int(data))
                break
            else:
                print("Dữ liệu không hợp lệ. Vui lòng nhập 0 hoặc 1.")
    return history

# Hàm phát hiện mẫu trong dữ liệu lịch sử
def detect_pattern(history):
    if len(history) < 12:
        return None
    
    last_12 = ''.join(map(str, history[-12:]))
    
    # Danh sách mẫu phổ biến
    common_patterns = {
        "111111111111": "Xỉu", "000000000000": "Tài", 
        "101010101010": "Tài", "010101010101": "Xỉu", 
        "110011001100": "Xỉu", "001100110011": "Tài", 
        "111000111000": "Xỉu", "000111000111": "Tài",
        "100100100100": "Xỉu", "011011011011": "Tài", 
        "110101101011": "Xỉu", "001010010100": "Tài", 
        "111001110011": "Xỉu", "000110001100": "Tài", 
        "101110111011": "Xỉu", "010001000100": "Tài",
        "100101001010": "Xỉu", "011010110101": "Tài",
        "110110011001": "Xỉu", "001001100110": "Tài",
        "101011110111": "Xỉu", "010100001000": "Tài",
        "111010011100": "Xỉu", "000101100011": "Tài",
        "100110110011": "Xỉu", "011001001100": "Tài",
        "110100101001": "Xỉu", "001011010110": "Tài",
        "101001010010": "Xỉu", "010110101101": "Tài",
        "111100000011": "Xỉu", "000011111100": "Tài",
        "110010101110": "Xỉu", "001101010001": "Tài",
        "101101100011": "Xỉu", "010010011100": "Tài",
        "100011110101": "Xỉu", "011100001010": "Tài",
        "111000100011": "Xỉu", "000111011100": "Tài",
        "110011101100": "Xỉu", "001100010011": "Tài",
    }

    return common_patterns.get(last_12, None)

# Hàm dự đoán tài xỉu
def predict_tai_xiu(history):
    if not history:
        return random.choice(["Tài", "Xỉu"])
    
    # Kiểm tra mẫu xuất hiện
    pattern_prediction = detect_pattern(history)
    if pattern_prediction:
        return pattern_prediction
    
    # Tính toán xác suất
    last_results = history[-12:]
    probability = sum(last_results) / len(last_results)
    
    # Kiểm tra chuỗi liên tục
    if history[-6:].count(1) == 6:
        return "Xỉu"
    elif history[-6:].count(0) == 6:
        return "Tài"
    
    return "Tài" if probability > 0.55 else "Xỉu"

# Chạy chương trình
if __name__ == "__main__":
    history = get_history_from_js()
    prediction = predict_tai_xiu(history)
    print(f"Dự đoán kết quả tiếp theo: {prediction}")
    print(f"Lịch sử gần đây: {history}")
    print("Tool by @vsk111")
