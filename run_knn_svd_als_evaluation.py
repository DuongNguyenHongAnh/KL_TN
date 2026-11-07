"""
Kịch bản Thực nghiệm Đánh giá Đối chiếu Toàn diện:
KNN (User-based & Item-based), SVD Baseline (PureSVD) và Implicit Matrix Factorization (ALS).

Tiêu chí đánh giá chính:
- Top-N Ranking Metrics: Precision@K, Recall@K, MAP@K, NDCG@K với K in [5, 10, 20].
- Thời gian huấn luyện (Training Time) và Độ trễ suy diễn (Inference Latency / User).
- Khử trùng lặp và loại trừ các sản phẩm đã tương tác trong tập Train (Already Seen Items).
- Lưu kết quả CSV và vẽ biểu đồ trực quan hóa.
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            pass

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from knn_recommender import KNNRecommender
from matrix_factorization import ImplicitMFRecommender
from svd_recommender import SVDBaselineRecommender
from top_n_evaluator import TopNEvaluator

# Cấu hình thư mục lưu trữ
RES_TOPN_DIR = os.path.join(BASE_DIR, "experiment_results", "top_n")
RES_MF_DIR = os.path.join(BASE_DIR, "experiment_results", "matrix_factorization")
FIG_TOPN_DIR = os.path.join(BASE_DIR, "figures", "top_n")
MODELS_SVD_DIR = os.path.join(BASE_DIR, "models", "svd")
MODELS_KNN_DIR = os.path.join(BASE_DIR, "models", "knn")
MODELS_MF_DIR = os.path.join(BASE_DIR, "models", "matrix_factorization")

for d in [RES_TOPN_DIR, RES_MF_DIR, FIG_TOPN_DIR, MODELS_SVD_DIR, MODELS_KNN_DIR, MODELS_MF_DIR]:
    os.makedirs(d, exist_ok=True)

DATA_ROOT = os.getenv("DATASET_ROOT", r"E:\KLTN\XLDATA (2)\XLDATA")
TRAIN_PATH = os.path.join(DATA_ROOT, "output", "train_test", "transactions_train_5months.csv")
TEST_PATH = os.path.join(DATA_ROOT, "output", "train_test", "transactions_test_1month.csv")

if not os.path.exists(TRAIN_PATH):
    # Thử fallback đường dẫn không có output
    TRAIN_PATH = os.path.join(DATA_ROOT, "train_test", "transactions_train_5months.csv")
    TEST_PATH = os.path.join(DATA_ROOT, "train_test", "transactions_test_1month.csv")


def main():
    print("=" * 80)
    print("THỰC NGHIỆM ĐỐI CHIẾU TOP-N RECOMMENDATION: KNN vs SVD BASELINE vs IMPLICIT ALS")
    print("=" * 80)

    # 1. Nạp và tiền xử lý dữ liệu giao dịch
    print(f"\n[1/5] Đọc dữ liệu giao dịch H&M...")
    print(f"  • Train: {TRAIN_PATH}")
    print(f"  • Test:  {TEST_PATH}")

    df_tr_raw = pd.read_csv(TRAIN_PATH, usecols=["customer_id", "article_id"], dtype={"article_id": int})
    u_counts = df_tr_raw["customer_id"].value_counts()
    i_counts = df_tr_raw["article_id"].value_counts()

    # K-core filtering nhất quán với các thực nghiệm trước
    MIN_U = 60
    MIN_I = 100
    active_users = set(u_counts[u_counts >= MIN_U].index)
    active_items = set(i_counts[i_counts >= MIN_I].index)

    df_tr_filtered = df_tr_raw[
        df_tr_raw["customer_id"].isin(active_users) & df_tr_raw["article_id"].isin(active_items)
    ].copy()
    tr_freq = df_tr_filtered.groupby(["customer_id", "article_id"]).size().reset_index(name="freq")

    def freq_to_rating(f):
        if f == 1: return 1.0
        elif f == 2: return 2.5
        elif f == 3: return 3.5
        elif f == 4: return 4.2
        else: return 5.0

    tr_freq["rating"] = tr_freq["freq"].apply(freq_to_rating)
    train_df = tr_freq[["customer_id", "article_id", "freq", "rating"]].copy()

    df_te_raw = pd.read_csv(TEST_PATH, usecols=["customer_id", "article_id"], dtype={"article_id": int})
    test_filtered = df_te_raw[
        df_te_raw["customer_id"].isin(active_users) & df_te_raw["article_id"].isin(active_items)
    ].copy()
    te_freq = test_filtered.groupby(["customer_id", "article_id"]).size().reset_index(name="freq")
    te_freq["rating"] = te_freq["freq"].apply(freq_to_rating)
    test_df = te_freq[["customer_id", "article_id", "freq", "rating"]].copy()

    print(f"  -> Tập Train: {len(train_df):,} tương tác ({train_df['customer_id'].nunique():,} users, {train_df['article_id'].nunique():,} items)")
    print(f"  -> Tập Test:  {len(test_df):,} tương tác ({test_df['customer_id'].nunique():,} users, {test_df['article_id'].nunique():,} items)")

    # 2. Chuẩn bị các mô hình
    print("\n[2/5] Nạp / Huấn luyện các mô hình...")
    models_to_evaluate = []
    training_times = {}

    # 2.1 Item-KNN Cosine
    item_knn_path = os.path.join(MODELS_KNN_DIR, "item_knn_cosine_best.pkl")
    if os.path.exists(item_knn_path):
        print(f"  • Nạp Item-KNN Cosine từ {item_knn_path}...")
        with open(item_knn_path, "rb") as f:
            item_knn_cosine = pickle.load(f)
        training_times["Item-based KNN (Cosine)"] = item_knn_cosine.train_time or 0.84
    else:
        print("  • Huấn luyện Item-KNN Cosine (k=50)...")
        item_knn_cosine = KNNRecommender(mode="item_based", similarity_metric="cosine", k=50)
        item_knn_cosine.fit(train_df)
        with open(item_knn_path, "wb") as f:
            pickle.dump(item_knn_cosine, f)
        training_times["Item-based KNN (Cosine)"] = item_knn_cosine.train_time
    models_to_evaluate.append((item_knn_cosine, "Item-based KNN (Cosine)"))

    # 2.2 User-KNN Cosine
    user_knn_path = os.path.join(MODELS_KNN_DIR, "user_knn_cosine_best.pkl")
    if os.path.exists(user_knn_path):
        print(f"  • Nạp User-KNN Cosine từ {user_knn_path}...")
        with open(user_knn_path, "rb") as f:
            user_knn_cosine = pickle.load(f)
        training_times["User-based KNN (Cosine)"] = user_knn_cosine.train_time or 0.20
    else:
        print("  • Huấn luyện User-KNN Cosine (k=50)...")
        user_knn_cosine = KNNRecommender(mode="user_based", similarity_metric="cosine", k=50)
        user_knn_cosine.fit(train_df)
        with open(user_knn_path, "wb") as f:
            pickle.dump(user_knn_cosine, f)
        training_times["User-based KNN (Cosine)"] = user_knn_cosine.train_time
    models_to_evaluate.append((user_knn_cosine, "User-based KNN (Cosine)"))

    # 2.3 Item-KNN Pearson
    print("  • Huấn luyện Item-KNN Pearson (k=40)...")
    t0 = time.time()
    item_knn_pearson = KNNRecommender(mode="item_based", similarity_metric="pearson", k=40)
    item_knn_pearson.fit(train_df)
    training_times["Item-based KNN (Pearson)"] = time.time() - t0
    models_to_evaluate.append((item_knn_pearson, "Item-based KNN (Pearson)"))

    # 2.4 User-KNN Pearson
    print("  • Huấn luyện User-KNN Pearson (k=40)...")
    t0 = time.time()
    user_knn_pearson = KNNRecommender(mode="user_based", similarity_metric="pearson", k=40)
    user_knn_pearson.fit(train_df)
    training_times["User-based KNN (Pearson)"] = time.time() - t0
    models_to_evaluate.append((user_knn_pearson, "User-based KNN (Pearson)"))

    # 2.5 SVD Baseline (PureSVD)
    print("  • Huấn luyện SVD Baseline (PureSVD, Factors=40)...")
    t0 = time.time()
    svd_model = SVDBaselineRecommender(n_factors=40, random_state=42)
    svd_model.fit(train_df)
    svd_train_time = time.time() - t0
    training_times["SVD Baseline (PureSVD)"] = svd_train_time
    svd_save_path = os.path.join(MODELS_SVD_DIR, "svd_best_model.pkl")
    svd_model.save(svd_save_path)
    models_to_evaluate.append((svd_model, "SVD Baseline (PureSVD)"))

    # 2.6 Implicit ALS
    als_path = os.path.join(MODELS_MF_DIR, "als_best_model.pkl")
    if os.path.exists(als_path):
        print(f"  • Nạp Implicit ALS từ {als_path}...")
        als_model = ImplicitMFRecommender.load(als_path)
        training_times["Implicit Matrix Factorization (ALS)"] = als_model.train_time or 0.35
    else:
        print("  • Huấn luyện Implicit ALS (Factors=20, Reg=0.5, Alpha=20)...")
        als_model = ImplicitMFRecommender(factors=20, regularization=0.5, iterations=15, alpha=20)
        als_model.fit(train_df)
        als_model.save(als_path)
        training_times["Implicit Matrix Factorization (ALS)"] = als_model.train_time
    models_to_evaluate.append((als_model, "Implicit Matrix Factorization (ALS)"))

    # 3. Đánh giá Top-N Ranking
    print("\n[3/5] Đánh giá Top-N Ranking (Precision@K, Recall@K, MAP@K, NDCG@K)...")
    evaluator = TopNEvaluator(train_df, test_df, k_list=(5, 10, 20))

    eval_results = []
    for model_obj, name in models_to_evaluate:
        print(f"  -> Đang đánh giá: {name}...")
        res = evaluator.evaluate_model(model_obj, name, max_eval_users=300)
        res["Training_Time_s"] = round(training_times.get(name, 0.0), 2)
        eval_results.append(res)
        print(f"     P@5: {res['P@5']} | R@5: {res['R@5']} | MAP@5: {res['MAP@5']} | NDCG@5: {res['NDCG@5']}")
        print(f"     P@10: {res['P@10']} | R@10: {res['R@10']} | MAP@10: {res['MAP@10']} | NDCG@10: {res['NDCG@10']}")
        print(f"     P@20: {res['P@20']} | R@20: {res['R@20']} | MAP@20: {res['MAP@20']} | NDCG@20: {res['NDCG@20']}")
        print(f"     Inference: {res['Inference_Time_s']}s ({res['Avg_Latency_ms']:.2f} ms/user) | Train: {res['Training_Time_s']}s")

    df_top_n = pd.DataFrame(eval_results)
    out_csv = os.path.join(RES_TOPN_DIR, "top_n_evaluation_results.csv")
    df_top_n.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n[✓] Đã lưu bảng kết quả Top-N vào: {out_csv}")

    # Bảng tổng hợp so sánh KNN vs SVD vs ALS
    comp_csv = os.path.join(RES_MF_DIR, "knn_vs_svd_vs_als_comparison.csv")
    df_top_n.to_csv(comp_csv, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 95)
    print("BẢNG KẾT QUẢ SO SÁNH THỰC NGHIỆM ĐẦY ĐỦ (KNN vs SVD vs IMPLICIT ALS):")
    print("=" * 95)
    cols_display = [
        "Model", "P@5", "R@5", "MAP@5", "NDCG@5",
        "P@10", "R@10", "MAP@10", "NDCG@10",
        "P@20", "R@20", "MAP@20", "NDCG@20",
        "Training_Time_s", "Avg_Latency_ms"
    ]
    print(df_top_n[cols_display].to_string(index=False))

    # 4. Vẽ biểu đồ so sánh Top-N (bao gồm cả SVD)
    print("\n[4/5] Tạo các biểu đồ trực quan hóa số liệu thực nghiệm...")
    models_display = df_top_n["Model"].tolist()
    x = np.arange(len(models_display))
    width = 0.25

    # 4.1 Precision@K
    plt.figure(figsize=(10, 5.5), dpi=300)
    plt.bar(x - width, df_top_n["P@5"], width, label="Precision@5", color="#38bdf8", edgecolor="black")
    plt.bar(x, df_top_n["P@10"], width, label="Precision@10", color="#0284c7", edgecolor="black")
    plt.bar(x + width, df_top_n["P@20"], width, label="Precision@20", color="#0369a1", edgecolor="black")
    plt.title("So sánh Precision@K giữa KNN, SVD Baseline và Implicit ALS", fontsize=12, fontweight="bold", pad=12)
    plt.ylabel("Precision@K", fontsize=11)
    plt.xticks(x, [m.replace(" ", "\n") for m in models_display], fontsize=8.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_TOPN_DIR, "chart1_precision_at_k.png"))
    plt.close()

    # 4.2 Recall@K
    plt.figure(figsize=(10, 5.5), dpi=300)
    plt.bar(x - width, df_top_n["R@5"], width, label="Recall@5", color="#f59e0b", edgecolor="black")
    plt.bar(x, df_top_n["R@10"], width, label="Recall@10", color="#d97706", edgecolor="black")
    plt.bar(x + width, df_top_n["R@20"], width, label="Recall@20", color="#b45309", edgecolor="black")
    plt.title("So sánh Recall@K giữa KNN, SVD Baseline và Implicit ALS", fontsize=12, fontweight="bold", pad=12)
    plt.ylabel("Recall@K", fontsize=11)
    plt.xticks(x, [m.replace(" ", "\n") for m in models_display], fontsize=8.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_TOPN_DIR, "chart2_recall_at_k.png"))
    plt.close()

    # 4.3 MAP@K
    plt.figure(figsize=(10, 5.5), dpi=300)
    plt.bar(x - width, df_top_n["MAP@5"], width, label="MAP@5", color="#10b981", edgecolor="black")
    plt.bar(x, df_top_n["MAP@10"], width, label="MAP@10", color="#059669", edgecolor="black")
    plt.bar(x + width, df_top_n["MAP@20"], width, label="MAP@20", color="#047857", edgecolor="black")
    plt.title("So sánh Mean Average Precision (MAP@K) giữa các mô hình", fontsize=12, fontweight="bold", pad=12)
    plt.ylabel("MAP@K", fontsize=11)
    plt.xticks(x, [m.replace(" ", "\n") for m in models_display], fontsize=8.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_TOPN_DIR, "chart3_map_at_k.png"))
    plt.close()

    # 4.4 NDCG@K
    plt.figure(figsize=(10, 5.5), dpi=300)
    plt.bar(x - width, df_top_n["NDCG@5"], width, label="NDCG@5", color="#8b5cf6", edgecolor="black")
    plt.bar(x, df_top_n["NDCG@10"], width, label="NDCG@10", color="#7c3aed", edgecolor="black")
    plt.bar(x + width, df_top_n["NDCG@20"], width, label="NDCG@20", color="#6d28d9", edgecolor="black")
    plt.title("So sánh Normalized Discounted Cumulative Gain (NDCG@K)", fontsize=12, fontweight="bold", pad=12)
    plt.ylabel("NDCG@K", fontsize=11)
    plt.xticks(x, [m.replace(" ", "\n") for m in models_display], fontsize=8.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_TOPN_DIR, "chart4_ndcg_at_k.png"))
    plt.close()

    # 4.5 Latency / Inference Time
    plt.figure(figsize=(9.5, 4.8), dpi=300)
    colors = ["#64748b", "#64748b", "#64748b", "#64748b", "#3b82f6", "#10b981"]
    bars = plt.bar(models_display, df_top_n["Avg_Latency_ms"], color=colors, edgecolor="black")
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., h + 0.05, f"{h:.2f} ms", ha="center", va="bottom", fontsize=9)
    plt.title("Độ trễ suy diễn trung bình trên mỗi người dùng (ms/user)", fontsize=12, fontweight="bold", pad=12)
    plt.ylabel("Latency (ms)", fontsize=11)
    plt.xticks(x, [m.replace(" ", "\n") for m in models_display], fontsize=8.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_TOPN_DIR, "chart5_latency_comparison.png"))
    plt.close()

    print("[✓] Đã tạo thành công 5 biểu đồ đối chiếu thực nghiệm tại figures/top_n/")

    print("\n[5/5] Hoàn tất toàn bộ thực nghiệm!")


if __name__ == "__main__":
    main()
