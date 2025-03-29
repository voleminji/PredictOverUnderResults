import random
import difflib

# Hàm yêu cầu người dùng nhập lịch sử kết quả
def get_history_from_js():
    history = []
    print("Nhập 10 kết quả gần nhất của tài/xỉu (0 = Xỉu, 1 = Tài):")
    for i in range(1, 11):
        while True:
            data = input(f"Nhập kết quả {i}/10 (0 hoặc 1): ")
            if data in ['0', '1']:
                history.append(int(data))
                break
            else:
                print("Dữ liệu không hợp lệ. Vui lòng nhập 0 hoặc 1.")
    return history

# Dictionary chứa các mẫu với tên cầu và kết quả dự đoán
pattern_info = {
    "11100":    ("Cầu 3-2-1", "Tài"),
    "00011":    ("Cầu 3-2-1", "Xỉu"),
    "10011":    ("Cầu 1-2-3", "Tài"),
    "01100":    ("Cầu 1-2-3", "Xỉu"),
    "111111":   ("Cầu bệt", "Tài"),
    "000000":   ("Cầu bệt", "Xỉu"),
    "101010":   ("Cầu đảo 1-1", "Tài"),
    "010101":   ("Cầu đảo 1-1", "Xỉu"),
    "110011":   ("Cầu 2:2", "Xỉu"),
    "001100":   ("Cầu 2:2", "Tài"),
    "011100111":    ("Cầu 1-3-2-4", "Tài"),
    "100011000":    ("Cầu 1-3-2-4", "Xỉu"),
    "000011100":   ("Cầu 4-3-2-1", "Tài"),
    "111100011":   ("Cầu 4-3-2-1", "Xỉu"),
    "011000":   ("Cầu 1-2-3-1", "Tài"),
    "100111":   ("Cầu 1-2-3-1", "Xỉu"),
    "000110":   ("Cầu 3-2-1-1", "Tài"),
    "111001":   ("Cầu 3-2-1-1", "Xỉu"),
    "110010":   ("Cầu Nghiêng", "Tài"),
    "001101":   ("Cầu Nghiêng", "Xỉu"),
    "111010":   ("Cầu Gãy", "Tài"),
    "000101":   ("Cầu Gãy", "Xỉu"),
    "100101":   ("Cầu Ngẫu Nhiên", "Tài"),
    "011010":   ("Cầu Ngẫu Nhiên", "Xỉu"),
    "10001":    ("Cầu 1-3-2", "Tài"),
    "01110":    ("Cầu 1-3-2", "Xỉu"),
    "110001":   ("Cầu 2-3-2", "Tài"),
    "001110":   ("Cầu 2-3-2", "Xỉu"),
    "101":      ("Cầu 1-1-2", "Tài"),
    "010":      ("Cầu 1-1-2", "Xỉu"),
    "1101":     ("Cầu 2-1-2", "Tài"),
    "0010":     ("Cầu 2-1-2", "Xỉu"),
    "100":      ("Cầu 1-2-1", "Tài"),
    "011":      ("Cầu 1-2-1", "Xỉu"),
    "1011":     ("Cầu 1-1-3", "Tài"),
    "0100":     ("Cầu 1-1-3", "Xỉu"),
    "0000111":  ("Cầu 4-4", "Tài"),
    "1111000":  ("Cầu 4-4", "Xỉu"),
    "00011":    ("Cầu 3-3", "Tài"),
    "11100":    ("Cầu 3-3", "Xỉu"),
    "10010":    ("Cầu nhảy", "Xỉu"),
    "01101":    ("Cầu nhảy", "Tài"),
    "00100":    ("Cầu Zigzag", "Tài"),
    "11011":    ("Cầu Zigzag", "Xỉu")
}

# Phát hiện mẫu dựa trên dictionary pattern_info với độ tương đồng nâng cao
def detect_pattern(history):
    if len(history) < 4:
        return None
    history_str = ''.join(map(str, history))
    
    best_match = None
    best_similarity = 0
    matched_result = None

    # Duyệt qua từng mẫu và quét toàn bộ chuỗi lịch sử
    for pattern, (pattern_name, result) in pattern_info.items():
        pattern_len = len(pattern)
        if len(history_str) < pattern_len:
            continue
        for i in range(len(history_str) - pattern_len + 1):
            sub_history = history_str[i:i+pattern_len]
            similarity = difflib.SequenceMatcher(None, sub_history, pattern).ratio()
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = pattern_name
                matched_result = result

    if best_similarity > 0.8:
        return best_match, matched_result
    return None

# Phát hiện các mẫu bổ sung nhằm đảm bảo dự đoán chính xác nhất
def detect_additional_patterns(history):
    # 1. Cầu đảo ngược hoàn toàn:
    if len(history) >= 6:
        seg1 = history[-3:]
        seg2 = history[-6:-3]
        if all(x == 1 - y for x, y in zip(seg1, seg2)):
            predicted = "Tài" if seg1[-1] == 0 else "Xỉu"
            return ("Cầu đảo ngược hoàn toàn", predicted)
    
    # 2. Cầu đôi: nếu 3 kết quả cuối giống nhau
    if len(history) >= 3:
        if history[-1] == history[-2] == history[-3]:
            predicted = "Tài" if history[-1] == 1 else "Xỉu"
            return ("Cầu đôi", predicted)
    
    # 3. Cầu thống kê tổng hợp: nếu tỷ lệ Tài/Xỉu lệch quá 70% trong 10 kết quả cuối
    if len(history) >= 10:
        count_tai = sum(history[-10:])
        ratio = count_tai / 10
        if ratio > 0.7:
            return ("Cầu thống kê tổng hợp", "Xỉu")
        elif (10 - count_tai) / 10 > 0.7:
            return ("Cầu thống kê tổng hợp", "Tài")
    
    # 4. Cầu chẵn – lẻ: sử dụng tổng của 4 kết quả cuối (nếu tổng chẵn dự đoán Tài, lẻ dự đoán Xỉu)
    if len(history) >= 4:
        s = sum(history[-4:])
        if s % 2 == 0:
            return ("Cầu chẵn-lẻ", "Tài")
        else:
            return ("Cầu chẵn-lẻ", "Xỉu")
    
    return None

# Hàm dự đoán tài/xỉu chỉ trả về "Tài" hoặc "Xỉu", kết hợp nhận diện mẫu nếu có
def predict_tai_xiu(history):
    if not history:
        return random.choice(["Tài", "Xỉu"])
    
    # Ưu tiên sử dụng mẫu từ dictionary pattern_info
    pattern_detection = detect_pattern(history)
    if pattern_detection:
        _, result = pattern_detection
        return result
    
    # Kiểm tra các mẫu bổ sung
    additional = detect_additional_patterns(history)
    if additional:
        _, result = additional
        return result
    
    # Nếu không tìm được mẫu nào, sử dụng EMA để dự đoán
    alpha = 0.7
    ema = history[0]
    for val in history[1:]:
        ema = alpha * val + (1 - alpha) * ema
    return "Tài" if ema > 0.52 else "Xỉu"

# Chương trình chính
if __name__ == "__main__":
    history = get_history_from_js()
    prediction = predict_tai_xiu(history)
    
    print(f"Dự đoán kết quả tiếp theo: {prediction}")
