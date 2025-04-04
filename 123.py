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
    
    # Danh sách mẫu cầu cập nhật
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
        "1010010": "Xỉu", "0101101": "Tài",
        "1101110": "Tài", "0010001": "Xỉu",
        "1011011": "Xỉu", "0100100": "Tài",
        "1111000": "Tài", "0000111": "Xỉu",
        "1001110": "Tài", "0110001": "Xỉu",
        "1100001": "Xỉu", "0011110": "Tài",
        "1011000": "Xỉu", "0100111": "Tài",
        "1110111": "Xỉu", "0001000": "Tài",
        "1000110": "Xỉu", "0111001": "Tài",
        "1111010": "Xỉu", "0000101": "Tài",
        "1001000": "Xỉu", "0110111": "Tài",
        "1101101": "Xỉu", "0010010": "Tài",
        "1010001": "Xỉu", "0101110": "Tài",
        "1010100": "Xỉu", "0101011": "Tài",
        "1100111": "Xỉu", "0011000": "Tài",
        "1101010": "Xỉu", "0010101": "Tài",
        "1010011": "Xỉu", "0101100": "Tài",
        "1111100": "Xỉu", "0000011": "Tài",
        "1011110": "Xỉu", "0100001": "Tài",
        "1000101": "Xỉu", "0111010": "Tài",
        "1100101": "Xỉu", "0011010": "Tài",
        "1110110": "Xỉu", "0001001": "Tài",
        "1010000": "Xỉu", "0101111": "Tài",
        "1001100": "Xỉu", "0110011": "Tài",
        "1111011": "Xỉu", "0000100": "Tài",
        "1000111": "Xỉu", "0111000": "Tài",
        "1100000": "Xỉu", "0011111": "Tài",
        "1011100": "Xỉu", "0100011": "Tài",
        "1111110": "Xỉu", "0000001": "Tài",
        "1010110": "Xỉu", "0101001": "Tài",
        "1000011": "Xỉu", "0111100": "Tài",
        "1110000": "Xỉu", "0001111": "Tài",
        "1100100": "Xỉu", "0011011": "Tài",
        "1001011": "Xỉu", "0110100": "Tài",
        "1101111": "Xỉu", "0010000": "Tài",
        "1011111": "Xỉu", "0100000": "Tài",
        "1000001": "Xỉu", "0111110": "Tài",
        "1110101": "Xỉu", "0001010": "Tài",
        "1001111": "Xỉu", "0110000": "Tài",
        "1101000": "Xỉu", "0010111": "Tài",
        "1111101": "Xỉu", "0000010": "Tài",
        "1011001": "Xỉu", "0100110": "Tài",
        "1110010": "Xỉu", "0001101": "Tài",
        "1011010": "Tài", "0100101": "Xỉu",
        "1111001": "Xỉu", "0000110": "Tài",
        "1000100": "Tài", "1000000": "Xỉu", 
        "0111111": "Tài", "0011100": "Xỉu",
        "1100011": "Tài", "0111011": "Xỉu",
        "1100010": "Xỉu", "0011101": "Tài",
        "1000010": "Xỉu", "0111101": "Tài"

    }
    return common_patterns.get(last_7, None)

# Hàm dự đoán tài xỉu với việc phân tích toàn bộ chuỗi khi không có mẫu cầu
def predict_tai_xiu(history):
    if not history:
        return random.choice(["Tài", "Xỉu"])
    
    # Kiểm tra mẫu xuất hiện trong 7 kết quả gần nhất
    pattern_prediction = detect_pattern(history)
    if pattern_prediction:
        return pattern_prediction
    
    # Nếu không có mẫu, phân tích toàn bộ chuỗi lịch sử
    overall_probability = sum(history) / len(history)
    
    # Tìm chuỗi kết quả liên tục dài nhất trong toàn bộ lịch sử
    longest_run = current_run = 1
    for i in range(1, len(history)):
        if history[i] == history[i - 1]:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 1

    # Nếu có chuỗi liên tục dài (>= 5), dự đoán kết quả ngược lại của chuỗi đó
    if longest_run >= 5:
        # Xác định giá trị của chuỗi liên tục cuối cùng
        last_value = history[-1]
        return "Xỉu" if last_value == 1 else "Tài"
    
    # Nếu không có chuỗi dài, sử dụng xác suất tổng hợp từ toàn bộ lịch sử
    # Điều chỉnh ngưỡng dựa trên xu hướng chung
    if overall_probability > 0.55:
        return "Tài"
    elif overall_probability < 0.45:
        return "Xỉu"
    else:
        return random.choice(["Tài", "Xỉu"])

# Chạy chương trình
if __name__ == "__main__":
    history = get_history_from_js()
    prediction = predict_tai_xiu(history)
    print(f"Dự đoán kết quả tiếp theo: {prediction}")
    print(f"Lịch sử gần đây: {history}")
    print(f"Tool by VSK696")
