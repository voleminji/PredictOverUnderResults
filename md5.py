import hashlib
import numpy as np
from collections import Counter
from scipy.spatial.distance import cityblock, chebyshev, minkowski, canberra, braycurtis
from scipy.stats import wasserstein_distance
from math import sqrt, log
import math

# Hàm tính mã MD5 của một chuỗi
def md5_hash(value):
    md5 = hashlib.md5()
    md5.update(value.strip().lower().encode('utf-8'))
    return md5.hexdigest()

# Hàm tính khoảng cách Hamming giữa hai mã MD5
def hamming_distance(md5_1, md5_2):
    return sum(c1 != c2 for c1, c2 in zip(md5_1, md5_2))

# Hàm tính Levenshtein Distance (còn gọi là khoảng cách chỉnh sửa)
def levenshtein_distance(s1, s2):
    n, m = len(s1), len(s2)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[n][m]


def damerau_levenshtein_distance(s1, s2):
    n, m = len(s1), len(s2)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,
                d[i][j - 1] + 1,
                d[i - 1][j - 1] + cost
            )
            if i > 1 and j > 1 and s1[i - 1] == s2[j - 2] and s1[i - 2] == s2[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + cost)
    return d[n][m]

def manhattan_distance(s1, s2):
    vec1 = np.array([int(c, 16) for c in s1])
    vec2 = np.array([int(c, 16) for c in s2])
    return cityblock(vec1, vec2)

def chebyshev_distance(s1, s2):
    vec1 = np.array([int(c, 16) for c in s1])
    vec2 = np.array([int(c, 16) for c in s2])
    return chebyshev(vec1, vec2)

def minkowski_distance(s1, s2, p=3):
    vec1 = np.array([int(c, 16) for c in s1])
    vec2 = np.array([int(c, 16) for c in s2])
    return minkowski(vec1, vec2, p)

def sorensen_dice_similarity(s1, s2):
    set1, set2 = set(s1), set(s2)
    return 2 * len(set1 & set2) / (len(set1) + len(set2))


# Hàm tính Cosine Similarity giữa hai chuỗi
def cosine_similarity(s1, s2):
    vec1 = Counter(s1)
    vec2 = Counter(s2)
    intersection = set(vec1) & set(vec2)
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    sum1 = sum([vec1[x] ** 2 for x in vec1])
    sum2 = sum([vec2[x] ** 2 for x in vec2])
    denominator = (sum1 ** 0.5) * (sum2 ** 0.5)
    return numerator / denominator if denominator != 0 else 0

# Hàm tính Jaccard Similarity giữa hai chuỗi
def jaccard_similarity(s1, s2):
    set1 = set(s1)
    set2 = set(s2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

# Hàm tính Jaro-Winkler Distance
def jaro_winkler(s1, s2):
    def jaro(s1, s2):
        len_s1 = len(s1)
        len_s2 = len(s2)
        if len_s1 == 0 or len_s2 == 0:
            return 0
        max_dist = (max(len_s1, len_s2) // 2) - 1
        match = 0
        s1_flags = [False] * len_s1
        s2_flags = [False] * len_s2

        for i in range(len_s1):
            start = max(0, i - max_dist)
            end = min(i + max_dist + 1, len_s2)
            for j in range(start, end):
                if s1[i] == s2[j] and not s2_flags[j]:
                    s1_flags[i] = True
                    s2_flags[j] = True
                    match += 1
                    break

        if match == 0:
            return 0
        t = 0
        k = 0
        for i in range(len_s1):
            if s1_flags[i]:
                while not s2_flags[k]:
                    k += 1
                if s1[i] != s2[k]:
                    t += 1
                k += 1
        t /= 2
        return (match / len_s1 + match / len_s2 + (match - t) / match) / 3
    


    def jaro_winkler_distance(s1, s2):
        jaro_dist = jaro(s1, s2)
        prefix_len = 0
        for i in range(min(len(s1), len(s2))):
            if s1[i] == s2[i]:
                prefix_len += 1
            else:
                break
        prefix_len = min(prefix_len, 4)
        return jaro_dist + (prefix_len * 0.1) * (1 - jaro_dist)

    return jaro_winkler_distance(s1, s2)


# Bhattacharyya Distance
def bhattacharyya_distance(s1, s2):
    vec1 = np.array([int(c, 16) for c in s1])
    vec2 = np.array([int(c, 16) for c in s2])
    return -log(np.sum(np.sqrt(vec1 * vec2)) + 1e-10)

# Canberra Distance
def canberra_distance(s1, s2):
    vec1 = np.array([int(c, 16) for c in s1])
    vec2 = np.array([int(c, 16) for c in s2])
    return canberra(vec1, vec2)

# Tversky Index
def tversky_index(s1, s2, alpha=0.5, beta=0.5):
    set1, set2 = set(s1), set(s2)
    intersection = len(set1 & set2)
    return intersection / (intersection + alpha * len(set1 - set2) + beta * len(set2 - set1))

# Bhattacharyya Coefficient
def bhattacharyya_coefficient(s1, s2):
    vec1 = Counter(s1)
    vec2 = Counter(s2)
    common_keys = set(vec1.keys()) & set(vec2.keys())
    return sum(sqrt(vec1[k] * vec2[k]) for k in common_keys) / (sqrt(sum(vec1.values())) * sqrt(sum(vec2.values())) + 1e-10)

# Editex Distance (Giả lập đơn giản)
def editex_distance(s1, s2):
    return sum(1 if c1 != c2 else 0 for c1, c2 in zip(s1, s2))

# Wasserstein Distance
def wasserstein_dist(s1, s2):
    vec1 = np.array([int(c, 16) for c in s1])
    vec2 = np.array([int(c, 16) for c in s2])
    return wasserstein_distance(vec1, vec2)

# Bray-Curtis Distance
def bray_curtis_distance(s1, s2):
    vec1 = np.array([int(c, 16) for c in s1])
    vec2 = np.array([int(c, 16) for c in s2])
    return braycurtis(vec1, vec2)


# Hàm tính Euclidean Distance giữa hai chuỗi
def euclidean_distance(s1, s2):
    vec1 = np.array([int(c, 16) for c in s1])
    vec2 = np.array([int(c, 16) for c in s2])
    return np.linalg.norm(vec1 - vec2)

# Hàm phân tích mã MD5 người dùng nhập và so sánh với các dãy số Tài và Xỉu
def analyze_md5(user_md5):
    # Gắn các số 3-10 là "Xỉu" và các số 11-18 là "Tài"
    xiu_range = list(range(3, 11))   # Xỉu: 3-10
    tai_range = list(range(11, 19))  # Tài: 11-18

    print(f"Mã MD5 của bạn: {user_md5}\n")

    # Tính khoảng cách cho Xỉu
    xiu_distances = { 
    'hamming': {}, 'levenshtein': {}, 'cosine': {}, 'jaccard': {}, 
    'jaro_winkler': {}, 'euclidean': {}, 'damerau_levenshtein': {}, 
    'manhattan': {}, 'sorensen_dice': {}, 'chebyshev': {}, 'minkowski': {},
    'bhattacharyya': {}, 'canberra': {}, 'tversky': {}, 'bhattacharyya_coefficient': {}, 'editex': {},
    'wasserstein': {}, 'bray_curtis': {}
}
    for number in xiu_range:
        number_md5 = md5_hash(str(number))  # Tính mã MD5 của số
        xiu_distances['hamming'][number] = hamming_distance(user_md5, number_md5)
        xiu_distances['levenshtein'][number] = levenshtein_distance(user_md5, number_md5)
        xiu_distances['cosine'][number] = cosine_similarity(user_md5, number_md5)
        xiu_distances['jaccard'][number] = jaccard_similarity(user_md5, number_md5)
        xiu_distances['damerau_levenshtein'][number] = damerau_levenshtein_distance(user_md5, number_md5)
        xiu_distances['manhattan'][number] = manhattan_distance(user_md5, number_md5)
        xiu_distances['sorensen_dice'][number] = sorensen_dice_similarity(user_md5, number_md5)
        xiu_distances['chebyshev'][number] = chebyshev_distance(user_md5, number_md5)
        xiu_distances['minkowski'][number] = minkowski_distance(user_md5, number_md5)
        xiu_distances['bhattacharyya'][number] = bhattacharyya_distance(user_md5, number_md5)
        xiu_distances['canberra'][number] = canberra_distance(user_md5, number_md5)
        xiu_distances['tversky'][number] = tversky_index(user_md5, number_md5)
        xiu_distances['bhattacharyya_coefficient'][number] = bhattacharyya_coefficient(user_md5, number_md5)
        xiu_distances['editex'][number] = editex_distance(user_md5, number_md5)
        xiu_distances['wasserstein'][number] = wasserstein_dist(user_md5, number_md5)
        xiu_distances['bray_curtis'][number] = bray_curtis_distance(user_md5, number_md5)
        xiu_distances['jaro_winkler'][number] = jaro_winkler(user_md5, number_md5)
        xiu_distances['euclidean'][number] = euclidean_distance(user_md5, number_md5)

    # Tính khoảng cách cho Tài
    tai_distances = { 
    'hamming': {}, 'levenshtein': {}, 'cosine': {}, 'jaccard': {}, 
    'jaro_winkler': {}, 'euclidean': {}, 'damerau_levenshtein': {}, 
    'manhattan': {}, 'sorensen_dice': {}, 'chebyshev': {}, 'minkowski': {},
    'bhattacharyya': {}, 'canberra': {}, 'tversky': {}, 'bhattacharyya_coefficient': {}, 'editex': {},
    'wasserstein': {}, 'bray_curtis': {}
}
    for number in tai_range:
        number_md5 = md5_hash(str(number))  # Tính mã MD5 của số
        tai_distances['hamming'][number] = hamming_distance(user_md5, number_md5)
        tai_distances['levenshtein'][number] = levenshtein_distance(user_md5, number_md5)
        tai_distances['cosine'][number] = cosine_similarity(user_md5, number_md5)
        tai_distances['damerau_levenshtein'][number] = damerau_levenshtein_distance(user_md5, number_md5)
        tai_distances['sorensen_dice'][number] = sorensen_dice_similarity(user_md5, number_md5)
        tai_distances['manhattan'][number] = manhattan_distance(user_md5, number_md5)
        tai_distances['chebyshev'][number] = chebyshev_distance(user_md5, number_md5)
        tai_distances['minkowski'][number] = minkowski_distance(user_md5, number_md5)
        tai_distances['jaccard'][number] = jaccard_similarity(user_md5, number_md5)
        tai_distances['bhattacharyya'][number] = bhattacharyya_distance(user_md5, number_md5)
        tai_distances['canberra'][number] = canberra_distance(user_md5, number_md5)
        tai_distances['tversky'][number] = tversky_index(user_md5, number_md5)
        tai_distances['bhattacharyya_coefficient'][number] = bhattacharyya_coefficient(user_md5, number_md5)
        tai_distances['wasserstein'][number] = wasserstein_dist(user_md5, number_md5)
        tai_distances['bray_curtis'][number] = bray_curtis_distance(user_md5, number_md5)
        tai_distances['editex'][number] = editex_distance(user_md5, number_md5)
        tai_distances['jaro_winkler'][number] = jaro_winkler(user_md5, number_md5)
        tai_distances['euclidean'][number] = euclidean_distance(user_md5, number_md5)

    # Xác định kết quả dựa trên khoảng cách gần nhất
    def best_match(distances):
        best_result = {}
        for method in distances:
            best_match_value = min(distances[method].values())
            best_result[method] = [k for k, v in distances[method].items() if v == best_match_value][0]
        return best_result

    xiu_best = best_match(xiu_distances)
    tai_best = best_match(tai_distances)

    print("\nKhoảng cách cho Xỉu (3-10):")
    print(xiu_best)

    print("\nKhoảng cách cho Tài (11-18):")
    print(tai_best)

    distances = { 'hamming': {}, 'levenshtein': {}, 'cosine': {}, 'jaccard': {}, 
    'jaro_winkler': {}, 'euclidean': {}, 'damerau_levenshtein': {}, 
    'manhattan': {}, 'sorensen_dice': {}, 'chebyshev': {}, 'minkowski': {},
    'bhattacharyya': {}, 'canberra': {}, 'tversky': {}, 'bhattacharyya_coefficient': {}, 'editex': {}, 
    'wasserstein': {}, 'bray_curtis': {}  }
    
    for number in xiu_range + tai_range:
        number_md5 = md5_hash(str(number))
        distances['hamming'][number] = hamming_distance(user_md5, number_md5)
        distances['levenshtein'][number] = levenshtein_distance(user_md5, number_md5)
        distances['cosine'][number] = cosine_similarity(user_md5, number_md5)
        distances['damerau_levenshtein'][number] = damerau_levenshtein_distance(user_md5, number_md5)
        distances['sorensen_dice'][number] = sorensen_dice_similarity(user_md5, number_md5)
        distances['manhattan'][number] = manhattan_distance(user_md5, number_md5)
        distances['chebyshev'][number] = chebyshev_distance(user_md5, number_md5)
        distances['minkowski'][number] = minkowski_distance(user_md5, number_md5)
        distances['jaccard'][number] = jaccard_similarity(user_md5, number_md5)
        distances['bhattacharyya'][number] = bhattacharyya_distance(user_md5, number_md5)
        distances['canberra'][number] = canberra_distance(user_md5, number_md5)
        distances['tversky'][number] = tversky_index(user_md5, number_md5)
        distances['bhattacharyya_coefficient'][number] = bhattacharyya_coefficient(user_md5, number_md5)
        distances['editex'][number] = editex_distance(user_md5, number_md5)
        distances['jaro_winkler'][number] = jaro_winkler(user_md5, number_md5)
        distances['wasserstein'][number] = wasserstein_dist(user_md5, number_md5)
        distances['bray_curtis'][number] = bray_curtis_distance(user_md5, number_md5)
        distances['euclidean'][number] = euclidean_distance(user_md5, number_md5)

    # Dự đoán kết quả
    predictions = {}
    tai_count, xiu_count = 0, 0
    for method in distances:
        if min({num: v for num, v in distances[method].items() if num in xiu_range}.values()) <= min({num: v for num, v in distances[method].items() if num in tai_range}.values()):
            predictions[method] = "Xỉu"
            xiu_count += 1
        else:
            predictions[method] = "Tài"
            tai_count += 1
    
    print("\nDự đoán kết quả theo từng thuật toán:")
    for method, result in predictions.items():
        print(f"{method}: {result}")
    
    print(f"\nTổng số lần dự đoán Tài: {tai_count}")
    print(f"Tổng số lần dự đoán Xỉu: {xiu_count}")
    print(f"Giai ma MD5 nay khong chinh xac hoan toan")

# Hàm chính
def main():
    user_md5 = input("Nhập mã MD5 của kết quả Tài hoặc Xỉu bạn muốn phân tích: ").strip()
    analyze_md5(user_md5)

if __name__ == "__main__":
    main()

