import telebot
import numpy as np
from collections import Counter

TOKEN = "7789180148:AAHzAdGMxWS3IWXkk-VoVpP8zoAsGkITALQ"
bot = telebot.TeleBot(TOKEN)

# Lưu lịch sử kết quả Tài/Xỉu và tổng điểm xúc xắc
history = []
dice_totals = []

# Xác định mẫu cầu phổ biến
def detect_pattern():
    if len(history) < 6:
        return "Chưa đủ dữ liệu để phân tích cầu!"

    last_6 = history[-6:]

    # Cầu bệt dài (từ 5 lần trở lên)
    if len(set(last_6[-5:])) == 1:
        return f"Cầu bệt dài: {last_6[-1]}"

    # Cầu nhảy đơn (1-1)
    if all(last_6[i] != last_6[i+1] for i in range(5)):
        return "Cầu nhảy đơn (1-1)"

    # Cầu nhảy đôi (2-2)
    if last_6[:2] == [last_6[0]] * 2 and last_6[2:4] == [last_6[2]] * 2 and last_6[4:] == [last_6[4]] * 2:
        return "Cầu nhảy đôi (2-2)"

    # Cầu nhảy ba (3-3)
    if last_6[:3] == [last_6[0]] * 3 and last_6[3:] == [last_6[3]] * 3:
        return "Cầu nhảy ba (3-3)"

    # Cầu đảo 1
    if last_6[0] == last_6[1] and last_6[2] != last_6[1] and last_6[3] == last_6[4] and last_6[5] != last_6[4]:
        return "Cầu đảo 1"

    # Cầu 3-2-1
    if last_6[:3] == [last_6[0]] * 3 and last_6[3:5] == [last_6[3]] * 2 and last_6[5] != last_6[3]:
        return "Cầu 3-2-1"

    # Cầu 1-2-3
    if last_6[0] != last_6[1] and last_6[1] == last_6[2] and last_6[3:6] == [last_6[3]] * 3:
        return "Cầu 1-2-3"

    # Cầu 1-3-2
    if last_6[:1] == [last_6[0]] * 1 and last_6[1:4] == [last_6[1]] * 3 and last_6[4:] != last_6[4]:
        return "Cầu 1-3-2"

    # Cầu 4-1
    if last_6[:4] == [last_6[0]] * 4 and last_6[4:] == [last_6[4]] * 1:
        return "Cầu 4-1"

    # Cầu tăng dần (Tài - Xỉu - Tài - Xỉu - …)
    if last_6[0] != last_6[1] and last_6[2] != last_6[3] and last_6[4] != last_6[5]:
        return "Cầu tăng dần"

    # Cầu giảm dần (Xỉu - Tài - Xỉu - Tài - …)
    if last_6[0] == last_6[1] and last_6[2] == last_6[3] and last_6[4] == last_6[5]:
        return "Cầu giảm dần"

    # Cầu lặp lại chu kỳ 3 phiên
    if history[-6:] == history[-12:-6]:
        return "Cầu lặp lại (Chu kỳ 3 phiên)"
    
    # Cầu nhịp 1-2-1
    if len(last_6) >= 3:
        if last_6[-3:] == ["Tài", "Xỉu", "Xỉu"]:
            return "Nhịp 1-2-1 - Dự đoán Tài"
        elif last_6[-3:] == ["Xỉu", "Tài", "Tài"]:
            return "Nhịp 1-2-1 - Dự đoán Xỉu"
        
    # Cầu đảo
    if len(set(last_6)) == 2 and all(last_6[i] != last_6[i + 1] for i in range(5)):
        return "Cầu đảo (Đảo ngược kết quả)"  
    
    # Cầu bệt (5 lần liên tiếp cùng kết quả)
    if len(set(last_6[:3])) == 1:
        return f"Cầu bệt đặc biệt: {last_6[0]}"
    
    # Cầu 2-1
    if len(last_6) >= 3:
        if last_6[:2] == ["Tài", "Tài"] and last_6[2] == "Xỉu":
            return "Cầu 2-1 - Dự đoán Tài"
        elif last_6[:2] == ["Xỉu", "Xỉu"] and last_6[2] == "Tài":
            return "Cầu 2-1 - Dự đoán Xỉu"
        

    # Cầu 3-1
    if len(last_6) >= 4:
        if last_6[:3] == ["Tài", "Tài", "Tài"] and last_6[3] == "Xỉu":
            return "Cầu 3-1 - Dự đoán Tài"
        elif last_6[:3] == ["Xỉu", "Xỉu", "Xỉu"] and last_6[3] == "Tài":
            return "Cầu 3-1 - Dự đoán Xỉu"

    # Cầu đảo 1-2
    if len(last_6) >= 4:
        if last_6[0] != last_6[1] and last_6[1] != last_6[2] and last_6[2] != last_6[3] and last_6[3] != last_6[4]:
            return "Cầu đảo 1-2 (Tài - Xỉu thay phiên)"       

    return "Không phát hiện cầu rõ ràng!"

# Kiểm tra cầu đặc biệt (số lẻ = Xỉu, số chẵn = Tài)
def special_case(dice_total):
    if dice_total % 2 == 0:
        return "Tài"
    else:
        return "Xỉu"

# Tính xác suất dự đoán dựa trên lịch sử
def calculate_probability():
    if len(history) < 5:
        return "Chưa đủ dữ liệu để tính xác suất!"

    count_tai = history.count("Tài")
    count_xiu = history.count("Xỉu")
    prob_tai = round((count_tai / len(history)) * 100, 2)
    prob_xiu = round((count_xiu / len(history)) * 100, 2)

    return f"Xác suất: Tài {prob_tai}% - Xỉu {prob_xiu}%"

# Phân tích tổng xúc xắc và xu hướng
def analyze_dice_totals():
    if len(dice_totals) < 3:
        return "Chưa đủ dữ liệu tổng xúc xắc để phân tích!"

    avg_total = np.mean(dice_totals)
    most_common = Counter(dice_totals).most_common(1)[0][0]
    
    trend = "Xu hướng: "
    if avg_total > 10.5:
        trend += "Nghiêng về Tài"
    else:
        trend += "Nghiêng về Xỉu"

    return f"Trung bình tổng: {avg_total:.2f}\nSố xuất hiện nhiều nhất: {most_common}\n{trend}"

# Người dùng nhập tổng 3 viên xúc xắc (tự động đổi thành Tài/Xỉu)
@bot.message_handler(func=lambda message: message.text.isdigit() and 3 <= int(message.text) <= 18)
def receive_dice_total(message):
    total = int(message.text)
    dice_totals.append(total)
    if len(dice_totals) > 10:
        dice_totals.pop(0)

    # Tính toán xác suất và cầu
    result = special_case(total)  # Áp dụng cầu đặc biệt (số lẻ là Xỉu, số chẵn là Tài)
    history.append(result)
    if len(history) > 10:
        history.pop(0)

    # Xử lý nếu cầu không xác định hoặc xác suất gần 50-50
    probability = calculate_probability()
    if probability == "Chưa đủ dữ liệu để tính xác suất!" or abs(history.count("Tài") - history.count("Xỉu")) < 2:
        result = special_case(total)  # Lấy kết quả cầu đặc biệt nếu không có xu hướng rõ ràng

    dice_analysis = analyze_dice_totals()
    pattern = detect_pattern()

    bot.send_message(message.chat.id, f"🎲 Đã lưu tổng xúc xắc: {total} ({result})\n{dice_analysis}\n{pattern}\n{probability}")

# Xử lý lệnh /start
@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, "🎲 Chào mừng bạn! Nhập tổng 3 viên xúc xắc (3-18) để phân tích.")

# Xử lý lệnh /reset
@bot.message_handler(commands=['reset'])
def reset_data(message):
    global history, dice_totals
    history = []  # Xóa lịch sử kết quả Tài/Xỉu
    dice_totals = []  # Xóa tổng xúc xắc
    bot.send_message(message.chat.id, "🚨 Đã xóa toàn bộ dữ liệu trước đó. Hãy nhập lại tổng 3 viên xúc xắc để bắt đầu phân tích.")

@bot.message_handler(commands=['history'])
def show_history(message):
    if not history:
        bot.send_message(message.chat.id, "🚨 Lịch sử chưa có dữ liệu.")
    else:
        history_text = "\n".join(f"{i+1}. {result}" for i, result in enumerate(history))
        bot.send_message(message.chat.id, f"📜 Lịch sử các phiên:\n{history_text}")

# Chạy bot
bot.polling()