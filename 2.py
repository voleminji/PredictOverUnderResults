import statistics
import random

def tim_seed_tot_nhat(lich_su):
    best_seed = None
    best_accuracy = 0
    
    for seed in range(1, 10000):  # Thử nghiệm từ seed 1 đến 9999 để tìm seed tốt hơn
        random.seed(seed)
        predicted_results = [du_doan_tai_xiu(lich_su, seed) for _ in range(10)]
        correct_predictions = sum(1 for result in predicted_results if result in lich_su)
        accuracy = correct_predictions / 10
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_seed = seed
    
    return best_seed, best_accuracy

def du_doan_tai_xiu(lich_su, seed=42):
    random.seed(seed)  # Cố định seed để tăng tính nhất quán
    
    if not lich_su or len(lich_su) < 21:
        return "Không đủ dữ liệu để dự đoán"
    
    # Sử dụng trung bình động có trọng số của 21 phiên gần nhất
    recent_games = lich_su[-21:]
    weights = [i for i in range(1, 22)]  # Trọng số cao hơn cho các phiên gần
    weighted_avg = sum(r * w for r, w in zip(recent_games, weights)) / sum(weights)
    
    # Tính độ lệch chuẩn để xác định mức độ dao động
    do_lech_chuan = statistics.stdev(recent_games) if len(recent_games) > 1 else 1
    
    # Mô phỏng chiến thuật của nhà cái (thay đổi nhỏ để gây nhiễu)
    bien_do = random.choice([-2, -1, 0, 1, 2])
    du_doan_min = max(3, int(weighted_avg - do_lech_chuan + bien_do))
    du_doan_max = min(18, int(weighted_avg + do_lech_chuan + bien_do))
    
    # Bẻ số xúc xắc dựa trên mô hình biến động thực tế của nhà cái
    xac_suat_bien_dong = random.uniform(0.1, 0.4)  # 10-40% xác suất biến động lớn
    if random.random() < xac_suat_bien_dong:
        bien_dong_manh = random.choice([-4, -3, 3, 4])
        du_doan_min = max(3, du_doan_min + bien_dong_manh)
        du_doan_max = min(18, du_doan_max + bien_dong_manh)
    
    # Xác suất có trọng số theo xu hướng gần đây
    tai_count = sum(1 for x in range(du_doan_min, du_doan_max + 1) if x >= 11)
    xiu_count = sum(1 for x in range(du_doan_min, du_doan_max + 1) if x < 11)
    total_count = tai_count + xiu_count
    
    ti_le_tai = (tai_count / total_count) * 100 if total_count > 0 else 0
    ti_le_xiu = (xiu_count / total_count) * 100 if total_count > 0 else 0
    
    du_doan = f"Dự đoán: Biên độ dao động ({du_doan_min} -> {du_doan_max})"
    
    return f"{du_doan}\nXác suất Tài: {ti_le_tai:.2f}%\nXác suất Xỉu: {ti_le_xiu:.2f}%\nSeed sử dụng: {seed}"

def nhap_lich_su():
    lich_su = []
    for i in range(21):
        while True:
            try:
                tong_diem = int(input(f"Nhập tổng điểm của phiên {i + 1}: "))
                if 3 <= tong_diem <= 18:
                    break
                else:
                    print("Giá trị không hợp lệ. Vui lòng nhập lại.")
            except ValueError:
                print("Vui lòng nhập một số nguyên hợp lệ.")
        
        lich_su.append(tong_diem)
    
    return lich_su

if __name__ == "__main__":
    lich_su_xuc_xac = nhap_lich_su()
    best_seed, best_accuracy = tim_seed_tot_nhat(lich_su_xuc_xac)
    ket_qua_du_doan = du_doan_tai_xiu(lich_su_xuc_xac, seed=best_seed)
    print(f"\nDự đoán kết quả tiếp theo với seed tối ưu ({best_seed}): {ket_qua_du_doan}")
