import statistics
import math

#########################################
# Basic Statistical Functions
#########################################

def normal_cdf(x, mean, std):
    """Compute the CDF of a normal distribution with given mean and std."""
    if std == 0:
        return 0 if x < mean else 1
    z = (x - mean) / (std * math.sqrt(2))
    return 0.5 * (1 + math.erf(z))

def tinh_ema(data, alpha):
    """Calculate Exponential Moving Average (EMA) using smoothing factor alpha."""
    if not data:
        return None
    ema = data[0]
    for value in data[1:]:
        ema = alpha * value + (1 - alpha) * ema
    return ema

def adaptive_alpha(data, base_alpha=0.3):
    """Adjust alpha based on the volatility (standard deviation) of the data."""
    if not data or len(data) < 2:
        return base_alpha
    std_dev = statistics.stdev(data)
    if std_dev > 3:
        return max(0.1, base_alpha - 0.1)
    elif std_dev < 2:
        return min(0.5, base_alpha + 0.1)
    else:
        return base_alpha

#########################################
# Functions for Enhanced Accuracy
#########################################

def filter_outliers(data, factor=1.5):
    """
    Filter out outlier sessions using the interquartile range.
    Returns a filtered copy of the data.
    """
    if not data:
        return data
    sorted_data = sorted(data)
    mid = len(sorted_data) // 2
    q1 = statistics.median(sorted_data[:mid])
    q3 = statistics.median(sorted_data[mid:])
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    return [x for x in data if lower_bound <= x <= upper_bound]

def calibrate_probabilities(prob, scale=1.1):
    """Calibrate probability using logistic scaling."""
    return 1 / (1 + math.exp(-scale * (prob - 0.5)))

def recalibrate_confidence(recent_accuracy, base_confidence):
    """Adjust the confidence score based on recent accuracy relative to a target of 90%."""
    target = 0.90
    if recent_accuracy >= target:
        return base_confidence
    else:
        return base_confidence * (recent_accuracy / target)

def ensemble_boost(weights, recent_window_accuracy):
    """Boost model weights if recent window accuracy is high."""
    boosted_weights = {}
    for model, w in weights.items():
        multiplier = 1.1 if recent_window_accuracy > 0.9 else 1.0
        boosted_weights[model] = w * multiplier
    return boosted_weights

#########################################
# Detect Potential Manipulation
#########################################

def detect_manipulation(lich_su, recent_window=10, threshold=1.5):
    """
    Detect if there's a significant shift in recent sessions compared to overall data.
    Returns True if the recent mean deviates from the overall mean by more than threshold * overall_std.
    """
    if len(lich_su) < recent_window + 1:
        return False
    overall_mean = statistics.mean(lich_su)
    overall_std = statistics.stdev(lich_su) if len(lich_su) > 1 else 1
    recent_data = lich_su[-recent_window:]
    recent_mean = statistics.mean(recent_data)
    return abs(recent_mean - overall_mean) > threshold * overall_std

#########################################
# Prediction Functions
#########################################

def du_doan_ema(lich_su, k=1.0, bias=0, alpha=0.3, window_size=12):
    """
    EMA-based prediction with adaptive alpha.
    Returns probabilities for "Tài" and "Xỉu" computed via normal CDF.
    """
    if not lich_su or len(lich_su) < window_size:
        return None
    recent_games = lich_su[-window_size:]
    alpha_adapt = adaptive_alpha(recent_games, alpha)
    ema = tinh_ema(recent_games, alpha_adapt)
    if ema is None:
        return None
    std_dev = statistics.stdev(recent_games) if len(recent_games) > 1 else 1
    center = ema + bias
    prob_tai = 1 - normal_cdf(10.5, center, std_dev)
    prob_xiu = normal_cdf(10.5, center, std_dev)
    prob_tai = calibrate_probabilities(prob_tai)
    prob_xiu = calibrate_probabilities(prob_xiu)
    label = "Tài" if prob_tai >= prob_xiu else "Xỉu"
    confidence = max(prob_tai, prob_xiu)
    return {"label": label, "prob_tai": prob_tai, "prob_xiu": prob_xiu, "confidence": confidence}

def du_doan_linear(lich_su, window_size=12):
    """Linear regression-based prediction on the last window_size sessions."""
    if not lich_su or len(lich_su) < window_size:
        return None
    data = lich_su[-window_size:]
    n = len(data)
    x_vals = list(range(1, n+1))
    sum_x = sum(x_vals)
    sum_y = sum(data)
    sum_xy = sum(x*y for x, y in zip(x_vals, data))
    sum_x2 = sum(x*x for x in x_vals)
    denominator = n * sum_x2 - sum_x**2
    slope = 0 if denominator == 0 else (n * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / n
    y_pred = intercept + slope * (n+1)
    label = "Tài" if y_pred >= 11 else "Xỉu"
    return {"label": label, "confidence": 0.75}

def overall_prediction(lich_su):
    """Overall regression prediction using all sessions."""
    if not lich_su or len(lich_su) < 2:
        return None
    n = len(lich_su)
    x_vals = list(range(1, n+1))
    sum_x = sum(x_vals)
    sum_y = sum(lich_su)
    sum_xy = sum(x*y for x, y in zip(x_vals, lich_su))
    sum_x2 = sum(x*x for x in x_vals)
    denominator = n * sum_x2 - sum_x**2
    slope = 0 if denominator == 0 else (n * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / n
    y_pred = intercept + slope * (n+1)
    label = "Tài" if y_pred >= 11 else "Xỉu"
    return {"label": label, "confidence": 0.70}

def du_doan_median(lich_su, window_size=12):
    """Median-based prediction using median and MAD of the last window_size sessions."""
    if not lich_su or len(lich_su) < window_size:
        return None
    recent = lich_su[-window_size:]
    median_val = statistics.median(recent)
    deviations = [abs(x - median_val) for x in recent]
    mad = statistics.median(deviations) if deviations else 1
    center = median_val
    prob_tai = 1 - normal_cdf(10.5, center, mad)
    prob_xiu = normal_cdf(10.5, center, mad)
    label = "Tài" if prob_tai >= prob_xiu else "Xỉu"
    confidence = max(prob_tai, prob_xiu)
    return {"label": label, "prob_tai": prob_tai, "prob_xiu": prob_xiu, "confidence": confidence}

#########################################
# Additional Models: Holt’s and Polynomial Regression
#########################################

def holt_linear_prediction(lich_su, alpha=0.3, beta=0.1):
    """
    Holt's Linear Trend method for prediction.
    Returns a prediction: "Tài" if forecast >= 11, else "Xỉu".
    """
    if not lich_su or len(lich_su) < 2:
        return None
    level = lich_su[0]
    trend = lich_su[1] - lich_su[0]
    for i in range(1, len(lich_su)):
        value = lich_su[i]
        prev_level = level
        level = alpha * value + (1 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend
    forecast = level + trend
    label = "Tài" if forecast >= 11 else "Xỉu"
    return {"label": label, "confidence": 0.70}

def poly_regression_prediction(lich_su):
    """
    Quadratic (degree 2) regression prediction.
    Returns "Tài" if predicted value >= 11, else "Xỉu".
    """
    n = len(lich_su)
    if n < 3:
        return None
    x_vals = list(range(1, n+1))
    sum_x = sum(x_vals)
    sum_x2 = sum(x**2 for x in x_vals)
    sum_x3 = sum(x**3 for x in x_vals)
    sum_x4 = sum(x**4 for x in x_vals)
    sum_y = sum(lich_su)
    sum_xy = sum(x*y for x, y in zip(x_vals, lich_su))
    sum_x2y = sum((x**2)*y for x, y in zip(x_vals, lich_su))
    D = (sum_x4*(sum_x2*n - sum_x**2) -
         sum_x3*(sum_x3*n - sum_x*sum_x2) +
         sum_x2*(sum_x3*sum_x - sum_x2**2))
    if D == 0:
        return None
    D_a = (sum_x2y*(sum_x2*n - sum_x**2) -
           sum_x3*(sum_xy*n - sum_x*sum_y) +
           sum_x2*(sum_xy*sum_x - sum_x2*sum_y))
    D_b = (sum_x4*(sum_xy*n - sum_x*sum_y) -
           sum_x2y*(sum_x3*n - sum_x*sum_x2) +
           sum_x2*(sum_x3*sum_y - sum_xy*sum_x2))
    D_c = (sum_x4*(sum_x2*sum_y - sum_xy*sum_x) -
           sum_x3*(sum_x3*sum_y - sum_xy*sum_x2) +
           sum_x2*(sum_x3*sum_xy - sum_x2*sum_x2y))
    a = D_a / D
    b = D_b / D
    c = D_c / D
    x_next = n + 1
    y_pred = a*(x_next**2) + b*x_next + c
    label = "Tài" if y_pred >= 11 else "Xỉu"
    return {"label": label, "confidence": 0.70}

#########################################
# Trend Analysis and Adaptive Weighting
#########################################

def analyze_trend_local(lich_su, window_size=12):
    """
    Analyze the last window_size sessions to propose local bias and k adjustments.
    """
    if len(lich_su) < window_size:
        return 0, 1.0
    recent = lich_su[-window_size:]
    tai_count = sum(1 for x in recent if x >= 11)
    xiu_count = window_size - tai_count
    recommended_bias = (tai_count - xiu_count) // 2
    recommended_k = 0.8 if abs(tai_count - xiu_count) >= 6 else 1.0
    return recommended_bias, recommended_k

def analyze_trend_overall(lich_su):
    """
    Analyze overall trend (global slope) to adjust bias.
    """
    if not lich_su or len(lich_su) < 2:
        return 0
    n = len(lich_su)
    x_vals = list(range(1, n+1))
    sum_x = sum(x_vals)
    sum_y = sum(lich_su)
    sum_xy = sum(x*y for x,y in zip(x_vals, lich_su))
    sum_x2 = sum(x*x for x in x_vals)
    denominator = n * sum_x2 - sum_x**2
    slope = 0 if denominator == 0 else (n * sum_xy - sum_x * sum_y) / denominator
    overall_bias_adjust = 0.5 * slope
    return overall_bias_adjust

def update_model_weights(lich_su, window=10, k=1.0, bias=0, alpha=0.3, window_size=12):
    """
    Evaluate performance of each model (EMA, Linear, Overall, Median, Holt, Poly)
    over the last 'window' sessions and return weights based on the proportion
    of correct predictions, with additional boosting if recent accuracy is high.
    """
    if len(lich_su) < window + window_size:
        return {"ema": 1.0, "linear": 1.0, "overall": 1.0, "median": 1.0, "holt": 1.0, "poly": 1.0}
    weights = {"ema": 0, "linear": 0, "overall": 0, "median": 0, "holt": 0, "poly": 0}
    count = 0
    for i in range(len(lich_su) - window, len(lich_su)):
        train_data = lich_su[:i]
        actual = "Tài" if lich_su[i] >= 11 else "Xỉu"
        for key, func in zip(
            ["ema", "linear", "overall", "median", "holt", "poly"],
            [lambda d: du_doan_ema(d, k, bias, alpha, window_size),
             lambda d: du_doan_linear(d, window_size),
             overall_prediction,
             lambda d: du_doan_median(d, window_size),
             holt_linear_prediction,
             poly_regression_prediction]
        ):
            pred = func(train_data)
            if pred is not None and pred["label"] == actual:
                weights[key] += 1
        count += 1
    for key in weights:
        weights[key] = weights[key] / count
    recent_acc = tinh_chinh_xac(lich_su, k, bias, alpha, window_size)
    boosted = ensemble_boost(weights, recent_window_accuracy=recent_acc)
    return boosted

#########################################
# Global History Analysis
#########################################

def analyze_entire_history(lich_su):
    """
    Analyze the entire historical sequence to compute global statistics.
    Returns a dictionary with overall frequency, global mean, median, and a recommended global bias.
    """
    if not lich_su:
        return {"global_bias": 0}
    overall_mean = statistics.mean(lich_su)
    overall_median = statistics.median(lich_su)
    count_tai = sum(1 for x in lich_su if x >= 11)
    count_xiu = len(lich_su) - count_tai
    freq_bias = (count_tai - count_xiu) // 2
    global_bias = freq_bias + int(round((overall_median - overall_mean)/2))
    return {"global_bias": global_bias, "overall_mean": overall_mean, "overall_median": overall_median}

#########################################
# Ensemble Prediction and Accuracy Evaluation
#########################################

def ensemble_prediction(lich_su, k, bias, alpha, window_size):
    """
    Combine predictions from EMA, Linear, Overall, Median, Holt, and Polynomial Regression
    using weighted voting and incorporate global history analysis.
    """
    pred_ema = du_doan_ema(lich_su, k, bias, alpha, window_size)
    pred_linear = du_doan_linear(lich_su, window_size)
    pred_overall = overall_prediction(lich_su)
    pred_median = du_doan_median(lich_su, window_size)
    pred_holt = holt_linear_prediction(lich_su)
    pred_poly = poly_regression_prediction(lich_su)
    
    models = {
        "ema": pred_ema,
        "linear": pred_linear,
        "overall": pred_overall,
        "median": pred_median,
        "holt": pred_holt,
        "poly": pred_poly
    }
    weights = update_model_weights(lich_su, window=10, k=k, bias=bias, alpha=alpha, window_size=window_size)
    vote_score = {}
    for key, pred in models.items():
        if pred is None:
            continue
        label = pred["label"]
        vote_score[label] = vote_score.get(label, 0) + weights.get(key, 1.0) * pred.get("confidence", 0)
    # Incorporate global history analysis adjustment:
    global_stats = analyze_entire_history(lich_su)
    if global_stats["global_bias"] > 0:
        vote_score["Tài"] = vote_score.get("Tài", 0) + abs(global_stats["global_bias"])
    elif global_stats["global_bias"] < 0:
        vote_score["Xỉu"] = vote_score.get("Xỉu", 0) + abs(global_stats["global_bias"])
        
    if not vote_score:
        return None
    final_label = max(vote_score, key=vote_score.get)
    conf_list = [pred.get("confidence", 0) for pred in models.values() if pred and pred["label"] == final_label]
    final_conf = sum(conf_list) / len(conf_list) if conf_list else 0
    return {"label": final_label, "confidence": final_conf}

def tinh_chinh_xac(lich_su, k=1.0, bias=0, alpha=0.3, window_size=12):
    """
    Calculate overall ensemble accuracy using a sliding window from window_size onward.
    """
    if len(lich_su) < window_size + 1:
        return 0
    correct = 0
    total = 0
    for i in range(window_size, len(lich_su)):
        train_data = lich_su[:i]
        pred = ensemble_prediction(train_data, k, bias, alpha, window_size)
        if pred is None:
            continue
        actual = "Tài" if lich_su[i] >= 11 else "Xỉu"
        if pred["label"] == actual:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0

def tim_tham_so_tot_nhat(lich_su, 
                          k_values=[0.6, 0.8, 1.0, 1.2, 1.4],
                          bias_values=range(-4, 5),
                          alpha_values=[0.1, 0.2, 0.3, 0.4, 0.5],
                          window_sizes=range(12, 17)):
    """
    Perform grid search to find optimal parameters (k, bias, alpha, window_size)
    based on internal cross-validation accuracy.
    """
    best_k = None
    best_bias = None
    best_alpha = None
    best_window = None
    best_accuracy = 0
    for k in k_values:
        for bias in bias_values:
            for alpha in alpha_values:
                for window_size in window_sizes:
                    acc = tinh_chinh_xac(lich_su, k, bias, alpha, window_size)
                    if acc > best_accuracy:
                        best_accuracy = acc
                        best_k = k
                        best_bias = bias
                        best_alpha = alpha
                        best_window = window_size
    return best_k, best_bias, best_alpha, best_window, best_accuracy

#########################################
# Data Input and Continuous Update
#########################################

def nhap_lich_su():
    """
    Input historical session results (minimum 13 sessions) and filter out outliers.
    """
    lich_su = []
    so_phan = int(input("Enter the number of sessions available (minimum 13): "))
    while so_phan < 13:
        print("At least 13 sessions are required to optimize and predict.")
        so_phan = int(input("Enter the number of sessions available (minimum 13): "))
    for i in range(so_phan):
        while True:
            try:
                tong_diem = int(input(f"Enter total points for session {i+1} (between 3 and 18): "))
                if 3 <= tong_diem <= 18:
                    break
                else:
                    print("Invalid value, please try again.")
            except ValueError:
                print("Please enter a valid integer.")
        lich_su.append(tong_diem)
    return filter_outliers(lich_su)

def update_prediction_loop(lich_su):
    """
    Continuously accept new session results (up to 100 additional sessions),
    update optimal parameters, dynamically adjust bias, and output ensemble predictions.
    Also, detect potential manipulation.
    The interface displays only the final prediction ("Tài" or "Xỉu").
    """
    additional_max = 100
    count = 0
    while count < additional_max:
        best_k, best_bias, best_alpha, best_window, best_accuracy = tim_tham_so_tot_nhat(lich_su)
        if best_window is None:
            best_window = 12
        local_bias, local_k = analyze_trend_local(lich_su, window_size=best_window)
        overall_bias = analyze_trend_overall(lich_su)
        adjusted_bias = (best_bias + local_bias + overall_bias) / 3
        adjusted_k = (best_k + local_k) / 2
        pred = ensemble_prediction(lich_su, adjusted_k, adjusted_bias, best_alpha, best_window)
        if pred is None:
            print("Insufficient data for prediction.")
        else:
            print("Prediction:", pred["label"])
        if detect_manipulation(lich_su):
            print("Warning: Recent data shows significant changes; the bookmaker may be altering outcomes.")
        if best_accuracy < 0.85:
            print("Warning: Model accuracy is below 85%; consider updating with more data.")
        tiep = input("Enter the result of the recent session (between 3 and 18) to update (or press Enter to quit): ")
        if tiep.strip() == "":
            break
        try:
            ket_qua = int(tiep)
            if ket_qua < 3 or ket_qua > 18:
                print("Invalid value; session ignored.")
            else:
                lich_su.append(ket_qua)
                count += 1
        except ValueError:
            print("Invalid input; session ignored.")
    return lich_su

#########################################
# Main Execution
#########################################

if __name__ == "__main__":
    lich_su = nhap_lich_su()
    lich_su = update_prediction_loop(lich_su)
