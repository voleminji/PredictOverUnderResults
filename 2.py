import statistics

def du_doan_tai_xiu(lich_su):
    if not lich_su or len(lich_su) < 5:
        return "Không đủ dữ liệu để dự đoán"
    
    # Sử dụng trung bình động của 5 phiên gần nhất
    recent_games = lich_su[-5:]
    trung_binh = sum(recent_games) / len(recent_games)
    
    # Tính độ lệch chuẩn để xác định mức độ dao động
    do_lech_chuan = statistics.stdev(recent_games) if len(recent_games) > 1 else 1
    
    # Dự đoán khoảng kết quả dựa trên trung bình và độ lệch chuẩn
    du_doan_min = max(3, int(trung_binh - do_lech_chuan))
    du_doan_max = min(18, int(trung_binh + do_lech_chuan))
    
    # Xác suất có trọng số
    tai_count = sum(1 for x in range(du_doan_min, du_doan_max + 1) if x >= 11)
    xiu_count = sum(1 for x in range(du_doan_min, du_doan_max + 1) if x < 11)
    total_count = (du_doan_max - du_doan_min + 1)
    
    ti_le_tai = (tai_count / total_count) * 100 if total_count > 0 else 0
    ti_le_xiu = (xiu_count / total_count) * 100 if total_count > 0 else 0
    
    du_doan = f"Dự đoán: Biên độ dao động ({du_doan_min} -> {du_doan_max})"
    
    return f"{du_doan}\nXác suất Tài: {ti_le_tai:.2f}%\nXác suất Xỉu: {ti_le_xiu:.2f}%"

def nhap_lich_su():
    lich_su = []
    for i in range(13):
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
    ket_qua_du_doan = du_doan_tai_xiu(lich_su_xuc_xac)
    print(f"\nDự đoán kết quả tiếp theo: {ket_qua_du_doan}")
