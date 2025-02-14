import random

# Hàm yêu cầu người dùng nhập lịch sử kết quả
def get_history_from_js():
    history = []
    print("Nhập 13 kết quả gần nhất của tài/xỉu (0 = Xỉu, 1 = Tài):")
    for i in range(1, 14):
        while True:
            data = input(f"Nhập kết quả {i}/13 (0 hoặc 1): ")
            if data in ['0', '1']:
                history.append(int(data))
                break
            else:
                print("Dữ liệu không hợp lệ. Vui lòng nhập 0 hoặc 1.")
    return history

# Hàm phát hiện mẫu trong dữ liệu lịch sử
def detect_pattern(history):
    if len(history) < 13:
        return None
    
    last_13 = ''.join(map(str, history[-13:]))
    
    # Danh sách mẫu phổ biến cập nhật
    common_patterns = {
        "1111111111111": "Xỉu", "0000000000000": "Tài", 
        "1010101010101": "Tài", "0101010101010": "Xỉu", 
        "1100110011001": "Xỉu", "0011001100110": "Tài", 
        "1110001110001": "Xỉu", "0001110001110": "Tài",
        "1001001001001": "Xỉu", "0110110110110": "Tài", 
        "1101011010110": "Xỉu", "0010100101001": "Tài", 
        "1010101101011": "Tài", "0101010010100": "Xỉu", 
        "1101101101101": "Xỉu", "0010010010010": "Tài", 
        "1011001011001": "Xỉu", "0100110100110": "Tài", 
        "1110000001110": "Xỉu", "0001111110001": "Tài", 
        "1010011010011": "Xỉu", "0101100101100": "Tài", 
        "1100101100101": "Xỉu", "0011010011010": "Tài", 
        "1001101001101": "Xỉu", "0110010110010": "Tài", 
        "1010100110111": "Tài", "0101011001000": "Xỉu", 
        "1111000111101": "Xỉu", "0000111000010": "Tài", 
        "1101010001011": "Xỉu", "0010101110100": "Tài", 
        "1010110110001": "Xỉu", "0101001001110": "Tài", 
        "1001110001101": "Xỉu", "0110001110010": "Tài",
        "1110101010110": "Xỉu", "0001010101001": "Tài",
        "1101101010001": "Xỉu", "0010010101110": "Tài",
        "1011010111001": "Xỉu", "0100101000110": "Tài",
        "1110111100001": "Xỉu", "0001000011110": "Tài",
        "1100101011001": "Xỉu", "0011010100110": "Tài"
    }

    return common_patterns.get(last_13, None)

# Hàm dự đoán tài xỉu và xác suất xúc xắc
def predict_tai_xiu(history):
    if not history:
        return random.choice(["Tài", "Xỉu"]), (3, 18)
    
    # Kiểm tra mẫu xuất hiện
    pattern_prediction = detect_pattern(history)
    if pattern_prediction:
        return pattern_prediction, (3, 18)
    
    # Tính toán xác suất có trọng số động
    last_results = history[-13:]
    weights = [1.3 ** i for i in range(1, 14)]  # Trọng số động mạnh hơn để phản ứng nhanh
    weighted_sum = sum(val * weight for val, weight in zip(last_results, weights))
    probability = weighted_sum / sum(weights)
    
    # Xác suất kết quả xúc xắc
    dice_range = (3, 10) if probability < 0.5 else (11, 18)
    
    # Kiểm tra xu hướng cầu dài
    if history[-13:].count(1) >= 10:
        return "Xỉu", (3, 10)
    elif history[-13:].count(0) >= 10:
        return "Tài", (11, 18)
    
    # Kiểm tra bẻ cầu nhanh (nhà cái can thiệp)
    if history[-8:] == [1, 0] * 4 or history[-8:] == [0, 1] * 4:
        return "Bẻ cầu - đổi chiến thuật", (3, 18)
    
    # Nhận diện cầu đảo và nhảy cầu
    if history[-6:] in [[1, 1, 0, 0, 1, 1], [0, 0, 1, 1, 0, 0]]:
        return "Cầu đảo", (3, 18)
    elif history[-6:] in [[1, 0, 1, 1, 0, 1], [0, 1, 0, 0, 1, 0]]:
        return "Nhảy cầu", (3, 18)
    
    # Nhận diện cầu ngược (Anti-pattern)
    if history[-5:] in [[1, 1, 1, 0, 0], [0, 0, 0, 1, 1]]:
        return "Cầu ngược", (3, 18)
    
    return ("Tài", dice_range) if probability > 0.55 else ("Xỉu", dice_range)

# Chạy chương trình
if __name__ == "__main__":
    history = get_history_from_js()
    prediction, dice_range = predict_tai_xiu(history)
    print(f"Dự đoán kết quả tiếp theo: {prediction}")
    print(f"Xác suất xúc xắc có thể ra từ: {dice_range[0]} đến {dice_range[1]}")
    print(f"Lịch sử gần đây: {history}")
    print("Tool by SunWinClub")