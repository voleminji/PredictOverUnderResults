import hashlib
import numpy as np
from collections import Counter
from scipy.spatial.distance import cityblock, chebyshev, minkowski, canberra, braycurtis, mahalanobis
from scipy.stats import wasserstein_distance, entropy, pearsonr, spearmanr
from math import sqrt, log
import zlib
import math
import os


# Hàm tính mã MD5 của một chuỗi
def md5_hash(value, iterations=100000):
    """
    Tính toán mã băm với độ bảo mật cao nhất,
    sử dụng MD5, SHA256, SHA3-256, BLAKE2b, SHA512, BLAKE2s và PBKDF2.
    """
    normalized = value.strip().lower().encode('utf-8')
    
    # Tính các hàm băm cơ bản
    md5_initial = hashlib.md5(normalized).hexdigest()
    sha256 = hashlib.sha256(md5_initial.encode('utf-8')).hexdigest()
    sha3_256 = hashlib.sha3_256(md5_initial.encode('utf-8')).hexdigest()
    blake2b = hashlib.blake2b(md5_initial.encode('utf-8')).hexdigest()
    sha512 = hashlib.sha512(md5_initial.encode('utf-8')).hexdigest()
    blake2s = hashlib.blake2s(md5_initial.encode('utf-8')).hexdigest()
    
    # Kết hợp các kết quả băm
    combined_hash = sha256 + sha3_256 + blake2b + sha512 + blake2s
    final_hash = hashlib.md5(combined_hash.encode('utf-8')).hexdigest()
    
    # Thực hiện nhiều vòng băm để tăng cường độ bảo mật
    for i in range(iterations):
        final_hash = hashlib.md5(final_hash.encode('utf-8')).hexdigest()
    
    return final_hash[:32]
  
def hamming_distance(md5_1, md5_2):
    return sum(c1 != c2 for c1, c2 in zip(md5_1, md5_2))


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

def cosine_similarity(s1, s2):
    vec1 = Counter(s1)
    vec2 = Counter(s2)
    intersection = set(vec1) & set(vec2)
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    sum1 = sum([vec1[x] ** 2 for x in vec1])
    sum2 = sum([vec2[x] ** 2 for x in vec2])
    denominator = (sum1 ** 0.5) * (sum2 ** 0.5)
    return numerator / denominator if denominator != 0 else 0

def jaccard_similarity(s1, s2):
    set1 = set(s1)
    set2 = set(s2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

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

def bhattacharyya_distance(s1, s2):
    vec1 = np.array([int(c, 16) for c in s1])
    vec2 = np.array([int(c, 16) for c in s2])
    return -log(np.sum(np.sqrt(vec1 * vec2)) + 1e-10)

def canberra_distance(s1, s2):
    vec1 = np.array([int(c, 16) for c in s1])
    vec2 = np.array([int(c, 16) for c in s2])
    return canberra(vec1, vec2)

def tversky_index(s1, s2, alpha=0.5, beta=0.5):
    set1, set2 = set(s1), set(s2)
    intersection = len(set1 & set2)
    return intersection / (intersection + alpha * len(set1 - set2) + beta * len(set2 - set1))

def bhattacharyya_coefficient(s1, s2):
    vec1 = Counter(s1)
    vec2 = Counter(s2)
    common_keys = set(vec1.keys()) & set(vec2.keys())
    return sum(sqrt(vec1[k] * vec2[k]) for k in common_keys) / (sqrt(sum(vec1.values())) * sqrt(sum(vec2.values())) + 1e-10)

def editex_distance(s1, s2):
    return sum(1 if c1 != c2 else 0 for c1, c2 in zip(s1, s2))

def wasserstein_dist(s1, s2):
    vec1 = np.array([int(c, 16) for c in s1])
    vec2 = np.array([int(c, 16) for c in s2])
    return wasserstein_distance(vec1, vec2)

def bray_curtis_distance(s1, s2):
    vec1 = np.array([int(c, 16) for c in s1])
    vec2 = np.array([int(c, 16) for c in s2])
    return braycurtis(vec1, vec2)

def euclidean_distance(s1, s2):
    vec1 = np.array([int(c, 16) for c in s1])
    vec2 = np.array([int(c, 16) for c in s2])
    return np.linalg.norm(vec1 - vec2)

def chi_square_distance(s1, s2):
    vec1 = np.array([int(c, 16) for c in s1])
    vec2 = np.array([int(c, 16) for c in s2])
    return np.sum((vec1 - vec2) ** 2 / (vec1 + vec2 + 1e-10))

def mahalanobis_distance(s1, s2):
    vec1 = np.array([int(c, 16) for c in s1])
    vec2 = np.array([int(c, 16) for c in s2])
    cov_matrix = np.cov(np.array([vec1, vec2]).T)
    cov_inv = np.linalg.pinv(cov_matrix)
    diff = vec1 - vec2
    return sqrt(np.dot(np.dot(diff.T, cov_inv), diff))

def kl_divergence(s1, s2):
    vec1 = np.array([int(c, 16) + 1e-10 for c in s1])
    vec2 = np.array([int(c, 16) + 1e-10 for c in s2])
    return entropy(vec1, vec2)

def pearson_correlation(s1, s2):
    vec1 = np.array([int(c, 16) for c in s1])
    vec2 = np.array([int(c, 16) for c in s2])
    return pearsonr(vec1, vec2)[0]

def spearman_correlation(s1, s2):
    vec1 = np.array([int(c, 16) for c in s1])
    vec2 = np.array([int(c, 16) for c in s2])
    return spearmanr(vec1, vec2)[0]

def jensen_shannon_divergence(s1, s2):
    vec1 = np.array([int(c, 16) + 1e-10 for c in s1])
    vec2 = np.array([int(c, 16) + 1e-10 for c in s2])
    m = 0.5 * (vec1 + vec2)
    return 0.5 * (entropy(vec1, m) + entropy(vec2, m))

def earth_movers_distance(s1, s2):
    vec1 = np.array([int(c, 16) for c in s1])
    vec2 = np.array([int(c, 16) for c in s2])
    return wasserstein_distance(vec1, vec2)

def rank_biased_overlap(s1, s2, p=0.9):
    set1 = set(s1)
    set2 = set(s2)
    intersection = len(set1 & set2)
    return (1 - p) * sum([(p ** i) * (intersection / (i + 1)) for i in range(min(len(s1), len(s2)))])

def normalized_compression_distance(s1, s2):
    c1 = len(zlib.compress(s1.encode('utf-8')))
    c2 = len(zlib.compress(s2.encode('utf-8')))
    c12 = len(zlib.compress((s1 + s2).encode('utf-8')))
    return (c12 - min(c1, c2)) / max(c1, c2)

def dice_coefficient(s1, s2):
    bigrams1 = set([s1[i:i+2] for i in range(len(s1)-1)])
    bigrams2 = set([s2[i:i+2] for i in range(len(s2)-1)])
    return 2 * len(bigrams1 & bigrams2) / (len(bigrams1) + len(bigrams2))

def hellinger_distance(s1, s2):
    vec1 = np.array([int(c, 16) for c in s1])
    vec2 = np.array([int(c, 16) for c in s2])
    return np.linalg.norm(np.sqrt(vec1) - np.sqrt(vec2)) / sqrt(2)

def tanimoto_coefficient(s1, s2):
    set1, set2 = set(s1), set(s2)
    intersection = len(set1 & set2)
    return intersection / (len(set1) + len(set2) - intersection)





def analyze_md5(user_md5):
    xiu_range = list(range(3, 11))   
    tai_range = list(range(11, 19))

    print(f"Mã MD5 của bạn: {user_md5}\n")

    # Tính khoảng cách cho Xỉu
    xiu_distances = { 
    'hamming': {}, 'levenshtein': {}, 'cosine': {}, 'jaccard': {}, 
    'jaro_winkler': {}, 'euclidean': {}, 'damerau_levenshtein': {}, 
    'manhattan': {}, 'sorensen_dice': {}, 'chebyshev': {}, 'minkowski': {},
    'bhattacharyya': {}, 'canberra': {}, 'tversky': {}, 'bhattacharyya_coefficient': {}, 'editex': {},
    'wasserstein': {}, 'bray_curtis': {}, 'chi_square': {}, 'mahalanobis': {}, 'kl_divergence': {},
    'pearson': {}, 'spearman': {}, 'jensen_shannon': {}, 'earth_movers': {}, 'rank_biased_overlap': {}, 'normalized_compression': {}, 
    'dice_coefficient': {}, 'hellinger_distance': {}, 'tanimoto_coefficient': {}
}
    for number in xiu_range:
        number_md5 = md5_hash(str(number))  
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
        xiu_distances['chi_square'][number] = chi_square_distance(user_md5, number_md5)
        xiu_distances['mahalanobis'][number] = mahalanobis_distance(user_md5, number_md5)
        xiu_distances['kl_divergence'][number] = kl_divergence(user_md5, number_md5)
        xiu_distances['pearson'][number] = pearson_correlation(user_md5, number_md5)
        xiu_distances['spearman'][number] = spearman_correlation(user_md5, number_md5)
        xiu_distances['jensen_shannon'][number] = jensen_shannon_divergence(user_md5, number_md5)
        xiu_distances['bhattacharyya'][number] = bhattacharyya_distance(user_md5, number_md5)
        xiu_distances['earth_movers'][number] = earth_movers_distance(user_md5, number_md5)
        xiu_distances['rank_biased_overlap'][number] = rank_biased_overlap(user_md5, number_md5)
        xiu_distances['normalized_compression'][number] = normalized_compression_distance(user_md5, number_md5)
        xiu_distances['dice_coefficient'][number] = dice_coefficient(user_md5, number_md5)
        xiu_distances['hellinger_distance'][number] = hellinger_distance(user_md5, number_md5)
        xiu_distances['tanimoto_coefficient'][number] = tanimoto_coefficient(user_md5, number_md5)


    tai_distances = { 
    'hamming': {}, 'levenshtein': {}, 'cosine': {}, 'jaccard': {}, 
    'jaro_winkler': {}, 'euclidean': {}, 'damerau_levenshtein': {}, 
    'manhattan': {}, 'sorensen_dice': {}, 'chebyshev': {}, 'minkowski': {},
    'bhattacharyya': {}, 'canberra': {}, 'tversky': {}, 'bhattacharyya_coefficient': {}, 'editex': {},
    'wasserstein': {}, 'bray_curtis': {}, 'chi_square': {}, 'mahalanobis': {}, 'kl_divergence': {},
    'pearson': {}, 'spearman': {}, 'jensen_shannon': {}, 'earth_movers': {}, 'rank_biased_overlap': {}, 'normalized_compression': {},
    'dice_coefficient': {}, 'hellinger_distance': {}, 'tanimoto_coefficient': {}
}
    for number in tai_range:
        number_md5 = md5_hash(str(number)) 
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
        tai_distances['chi_square'][number] = chi_square_distance(user_md5, number_md5)
        tai_distances['mahalanobis'][number] = mahalanobis_distance(user_md5, number_md5)
        tai_distances['kl_divergence'][number] = kl_divergence(user_md5, number_md5)
        tai_distances['pearson'][number] = pearson_correlation(user_md5, number_md5)
        tai_distances['spearman'][number] = spearman_correlation(user_md5, number_md5)
        tai_distances['jensen_shannon'][number] = jensen_shannon_divergence(user_md5, number_md5)
        tai_distances['bhattacharyya'][number] = bhattacharyya_distance(user_md5, number_md5)
        tai_distances['earth_movers'][number] = earth_movers_distance(user_md5, number_md5)
        tai_distances['rank_biased_overlap'][number] = rank_biased_overlap(user_md5, number_md5)
        tai_distances['normalized_compression'][number] = normalized_compression_distance(user_md5, number_md5)
        tai_distances['dice_coefficient'][number] = dice_coefficient(user_md5, number_md5)
        tai_distances['hellinger_distance'][number] = hellinger_distance(user_md5, number_md5)
        tai_distances['tanimoto_coefficient'][number] = tanimoto_coefficient(user_md5, number_md5)

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
    'wasserstein': {}, 'bray_curtis': {}, 'chi_square': {}, 'mahalanobis': {}, 'kl_divergence': {},
    'pearson': {}, 'spearman': {}, 'jensen_shannon': {}, 'earth_movers': {}, 'rank_biased_overlap': {}, 'normalized_compression': {},
    'dice_coefficient': {}, 'hellinger_distance': {}, 'tanimoto_coefficient': {}  }
    
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
        distances['chi_square'][number] = chi_square_distance(user_md5, number_md5)
        distances['mahalanobis'][number] = mahalanobis_distance(user_md5, number_md5)
        distances['kl_divergence'][number] = kl_divergence(user_md5, number_md5)
        distances['pearson'][number] = pearson_correlation(user_md5, number_md5)
        distances['spearman'][number] = spearman_correlation(user_md5, number_md5)
        distances['jensen_shannon'][number] = jensen_shannon_divergence(user_md5, number_md5)
        distances['bhattacharyya'][number] = bhattacharyya_distance(user_md5, number_md5)
        distances['earth_movers'][number] = earth_movers_distance(user_md5, number_md5)
        distances['rank_biased_overlap'][number] = rank_biased_overlap(user_md5, number_md5)
        distances['normalized_compression'][number] = normalized_compression_distance(user_md5, number_md5)
        distances['dice_coefficient'][number] = dice_coefficient(user_md5, number_md5)
        distances['hellinger_distance'][number] = hellinger_distance(user_md5, number_md5)
        distances['tanimoto_coefficient'][number] = tanimoto_coefficient(user_md5, number_md5)


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

