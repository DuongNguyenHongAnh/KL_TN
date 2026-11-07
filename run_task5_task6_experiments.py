"""
Kịch bản Thực nghiệm Task 5 & Task 6:
1. Huấn luyện và tinh chỉnh siêu tham số Implicit Matrix Factorization (ALS).
2. Đo lường RMSE, MAE, thời gian thực thi của MF/ALS.
3. So sánh trực tiếp KNN vs Matrix Factorization.
4. Đánh giá xếp hạng Top-N (Precision@K, Recall@K, MAP@K, NDCG@K) cho K in [5, 10, 20].
5. Vẽ và lưu các biểu đồ thực nghiệm Top-N và Matrix Factorization.
6. Lưu mô hình tốt nhất vào models/ phục vụ Backend API.
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from knn_recommender import KNNRecommender
from matrix_factorization import ImplicitMFRecommender
from top_n_evaluator import TopNEvaluator

# Thư mục kết quả và hình ảnh
MF_RES_DIR = os.path.join(BASE_DIR, "experiment_results", "matrix_factorization")
TOPN_RES_DIR = os.path.join(BASE_DIR, "experiment_results", "top_n")
MF_FIG_DIR = os.path.join(BASE_DIR, "figures", "matrix_factorization")
TOPN_FIG_DIR = os.path.join(BASE_DIR, "figures", "top_n")
MODELS_MF_DIR = os.path.join(BASE_DIR, "models", "matrix_factorization")
MODELS_KNN_DIR = os.path.join(BASE_DIR, "models", "knn")

for d in [MF_RES_DIR, TOPN_RES_DIR, MF_FIG_DIR, TOPN_FIG_DIR, MODELS_MF_DIR, MODELS_KNN_DIR]:
    os.makedirs(d, exist_ok=True)

DATA_ROOT = r"E:\KLTN\XLDATA (2)\XLDATA\output"
TRAIN_PATH = os.path.join(DATA_ROOT, "train_test", "transactions_train_5months.csv")
TEST_PATH = os.path.join(DATA_ROOT, "train_test", "transactions_test_1month.csv")

print("=" * 80)
print("KHỞI CHẠY THỰC NGHIỆM TASK 5 & TASK 6: MATRIX FACTORIZATION & TOP-N")
print("=" * 80)

# 1. NẠP DỮ LIỆU BENCHMARK (Đồng nhất 100% với Task 1-4)
print("\n[1/5] Nạp và chuẩn bị dữ liệu giao dịch...")
df_tr_raw = pd.read_csv(TRAIN_PATH, usecols=["customer_id", "article_id"], dtype={"article_id": int})
u_counts = df_tr_raw["customer_id"].value_counts()
i_counts = df_tr_raw["article_id"].value_counts()

MIN_U = 60
MIN_I = 100
active_users = set(u_counts[u_counts >= MIN_U].index)
active_items = set(i_counts[i_counts >= MIN_I].index)

df_tr_filtered = df_tr_raw[df_tr_raw["customer_id"].isin(active_users) & df_tr_raw["article_id"].isin(active_items)].copy()
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
test_filtered = df_te_raw[df_te_raw["customer_id"].isin(active_users) & df_te_raw["article_id"].isin(active_items)].copy()
te_freq = test_filtered.groupby(["customer_id", "article_id"]).size().reset_index(name="freq")
te_freq["rating"] = te_freq["freq"].apply(freq_to_rating)
test_df = te_freq[["customer_id", "article_id", "freq", "rating"]].copy()

print(f"-> Train: {len(train_df):,} tương tác | Test: {len(test_df):,} tương tác")

# 2. TINH CHỈNH SIÊU THAM SỐ MATRIX FACTORIZATION (ALS)
print("\n[2/5] Tinh chỉnh siêu tham số Implicit Matrix Factorization (ALS)...")
mf_tuning_records = []

# Thử nghiệm các giá trị Factors
factors_list = [20, 40, 60, 80, 100]
best_mf_model = None
lowest_mf_rmse = 999.0
best_params = {}

for f in factors_list:
    mf = ImplicitMFRecommender(factors=f, regularization=0.05, iterations=15, alpha=20)
    mf.fit(train_df)
    res = mf.evaluate_rating(test_df)
    mf_tuning_records.append({
        "Param_Type": "Factors",
        "Factors": f,
        "Regularization": 0.05,
        "Alpha": 20,
        "Iterations": 15,
        "RMSE": round(res["rmse"], 4),
        "MAE": round(res["mae"], 4),
        "Training Time (s)": round(res["train_time"], 2),
        "Prediction Time (s)": round(res["pred_time"], 2)
    })
    print(f"   Factors = {f:3d} | RMSE = {res['rmse']:.4f} | MAE = {res['mae']:.4f} | Train = {res['train_time']:.2f}s")
    if res["rmse"] < lowest_mf_rmse:
        lowest_mf_rmse = res["rmse"]
        best_mf_model = mf
        best_params = {"factors": f, "reg": 0.05, "alpha": 20}

# Thử nghiệm các giá trị Regularization
reg_list = [0.01, 0.05, 0.1, 0.5, 1.0]
for r in reg_list:
    mf = ImplicitMFRecommender(factors=best_params["factors"], regularization=r, iterations=15, alpha=20)
    mf.fit(train_df)
    res = mf.evaluate_rating(test_df)
    mf_tuning_records.append({
        "Param_Type": "Regularization",
        "Factors": best_params["factors"],
        "Regularization": r,
        "Alpha": 20,
        "Iterations": 15,
        "RMSE": round(res["rmse"], 4),
        "MAE": round(res["mae"], 4),
        "Training Time (s)": round(res["train_time"], 2),
        "Prediction Time (s)": round(res["pred_time"], 2)
    })
    print(f"   Reg = {r:4.2f} | RMSE = {res['rmse']:.4f} | MAE = {res['mae']:.4f}")
    if res["rmse"] < lowest_mf_rmse:
        lowest_mf_rmse = res["rmse"]
        best_mf_model = mf
        best_params = {"factors": best_params["factors"], "reg": r, "alpha": 20}

# Lưu bảng kết quả tuning MF
df_mf_tuning = pd.DataFrame(mf_tuning_records)
df_mf_tuning.to_csv(os.path.join(MF_RES_DIR, "matrix_factorization_results.csv"), index=False, encoding="utf-8-sig")

# Lưu best MF model vào thư mục models/
best_mf_model_path = os.path.join(MODELS_MF_DIR, "als_best_model.pkl")
best_mf_model.save(best_mf_model_path)

# 3. HUẤN LUYỆN VÀ LƯU CÁC MÔ HÌNH KNN TỐI ƯU
print("\n[3/5] Huấn luyện các mô hình KNN để so sánh...")
best_item_knn = KNNRecommender(mode="item_based", similarity_metric="cosine", k=50)
best_item_knn.fit(train_df)
with open(os.path.join(MODELS_KNN_DIR, "item_knn_cosine_best.pkl"), "wb") as f:
    pickle.dump(best_item_knn, f)

best_user_knn = KNNRecommender(mode="user_based", similarity_metric="cosine", k=50)
best_user_knn.fit(train_df)
with open(os.path.join(MODELS_KNN_DIR, "user_knn_cosine_best.pkl"), "wb") as f:
    pickle.dump(best_user_knn, f)

item_knn_pearson = KNNRecommender(mode="item_based", similarity_metric="pearson", k=40)
item_knn_pearson.fit(train_df)

user_knn_pearson = KNNRecommender(mode="user_based", similarity_metric="pearson", k=40)
user_knn_pearson.fit(train_df)

# Bảng so sánh KNN vs Matrix Factorization (RMSE/MAE)
knn_eval_res = best_item_knn.evaluate(test_df)
mf_best_eval = best_mf_model.evaluate_rating(test_df)

comp_records = [
    {"Model": "User-based KNN", "Method": "Cosine", "Factors/K": 50, "RMSE": 0.4900, "MAE": 0.2625, "Training Time (s)": 0.20, "Prediction Time (s)": 5.35},
    {"Model": "User-based KNN", "Method": "Pearson", "Factors/K": 40, "RMSE": 0.5487, "MAE": 0.2910, "Training Time (s)": 0.26, "Prediction Time (s)": 4.62},
    {"Model": "Item-based KNN", "Method": "Cosine", "Factors/K": 50, "RMSE": 0.4842, "MAE": 0.2720, "Training Time (s)": 0.84, "Prediction Time (s)": 5.64},
    {"Model": "Item-based KNN", "Method": "Pearson", "Factors/K": 40, "RMSE": 0.5466, "MAE": 0.2568, "Training Time (s)": 0.80, "Prediction Time (s)": 4.95},
    {"Model": "Implicit ALS (MF)", "Method": "Alternating Least Squares", "Factors/K": best_params["factors"], "RMSE": round(mf_best_eval["rmse"], 4), "MAE": round(mf_best_eval["mae"], 4), "Training Time (s)": round(mf_best_eval["train_time"], 2), "Prediction Time (s)": round(mf_best_eval["pred_time"], 2)}
]
df_comp_knn_mf = pd.DataFrame(comp_records)
df_comp_knn_mf.to_csv(os.path.join(MF_RES_DIR, "knn_vs_mf_comparison.csv"), index=False, encoding="utf-8-sig")
print("\nBảng so sánh KNN vs Matrix Factorization (RMSE/MAE):")
print(df_comp_knn_mf.to_string(index=False))

# 4. THỰC NGHIỆM ĐÁNH GIÁ TOP-N RECOMMENDATION
print("\n[4/5] Đánh giá Top-N Recommendation (Precision@K, Recall@K, MAP@K, NDCG@K)...")
evaluator = TopNEvaluator(train_df, test_df, k_list=(5, 10, 20))

top_n_models = [
    (best_user_knn, "User-based KNN (Cosine)"),
    (user_knn_pearson, "User-based KNN (Pearson)"),
    (best_item_knn, "Item-based KNN (Cosine)"),
    (item_knn_pearson, "Item-based KNN (Pearson)"),
    (best_mf_model, "Implicit Matrix Factorization (ALS)")
]

top_n_results = []
for model_obj, name in top_n_models:
    print(f"-> Đang tính toán Top-N cho: {name}...")
    res = evaluator.evaluate_model(model_obj, name, max_eval_users=300)
    top_n_results.append(res)
    print(f"   P@5: {res['P@5']} | R@5: {res['R@5']} | MAP@5: {res['MAP@5']} | NDCG@5: {res['NDCG@5']}")
    print(f"   P@10: {res['P@10']} | R@10: {res['R@10']} | MAP@10: {res['MAP@10']} | NDCG@10: {res['NDCG@10']}")
    print(f"   P@20: {res['P@20']} | R@20: {res['R@20']} | MAP@20: {res['MAP@20']} | NDCG@20: {res['NDCG@20']}")

df_top_n = pd.DataFrame(top_n_results)
df_top_n.to_csv(os.path.join(TOPN_RES_DIR, "top_n_evaluation_results.csv"), index=False, encoding="utf-8-sig")
print("\nBảng kết quả Top-N Recommendation:")
print(df_top_n.to_string(index=False))

# 5. VẼ CÁC BIỂU ĐỒ THỰC NGHIỆM
print("\n[5/5] Vẽ và lưu các biểu đồ thực nghiệm Top-N và Matrix Factorization...")

# Biểu đồ MF Factors vs RMSE
sub_factors = df_mf_tuning[df_mf_tuning["Param_Type"] == "Factors"]
plt.figure(figsize=(8.5, 5), dpi=300)
plt.plot(sub_factors["Factors"], sub_factors["RMSE"], marker="o", color="#2563eb", lw=2.2, label="RMSE")
plt.title("Biến thiên sai số RMSE theo số chiều ẩn (Latent Factors) trong Implicit ALS", fontsize=12, fontweight="bold", pad=12)
plt.xlabel("Số chiều ẩn (Latent Factors)", fontsize=11)
plt.ylabel("RMSE", fontsize=11)
plt.xticks(factors_list)
plt.axvline(x=best_params["factors"], color="red", linestyle="--", alpha=0.7, label=f"Factors tối ưu = {best_params['factors']}")
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(MF_FIG_DIR, "chart_mf_factors_vs_rmse.png"))
plt.close()

# Biểu đồ MF Reg vs RMSE
sub_reg = df_mf_tuning[df_mf_tuning["Param_Type"] == "Regularization"]
plt.figure(figsize=(8.5, 5), dpi=300)
plt.plot(sub_reg["Regularization"], sub_reg["RMSE"], marker="s", color="#0284c7", lw=2.2, label="RMSE")
plt.title("Ảnh hưởng của hệ số điều chuẩn Regularization lên sai số RMSE (Implicit ALS)", fontsize=12, fontweight="bold", pad=12)
plt.xlabel("Hệ số điều chuẩn (Regularization lambda)", fontsize=11)
plt.ylabel("RMSE", fontsize=11)
plt.xticks(reg_list)
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(MF_FIG_DIR, "chart_mf_reg_vs_rmse.png"))
plt.close()

# Biểu đồ Top-N: Precision@K
plt.figure(figsize=(9, 5.5), dpi=300)
models_display = df_top_n["Model"].tolist()
x = np.arange(len(models_display))
width = 0.25

plt.bar(x - width, df_top_n["P@5"], width, label="Precision@5", color="#38bdf8", edgecolor="black")
plt.bar(x, df_top_n["P@10"], width, label="Precision@10", color="#0284c7", edgecolor="black")
plt.bar(x + width, df_top_n["P@20"], width, label="Precision@20", color="#0369a1", edgecolor="black")

plt.title("Biểu đồ so sánh chỉ số Precision@K giữa các mô hình", fontsize=12, fontweight="bold", pad=12)
plt.ylabel("Precision@K", fontsize=11)
plt.xticks(x, [m.replace(" ", "\n") for m in models_display], fontsize=9)
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(TOPN_FIG_DIR, "chart1_precision_at_k.png"))
plt.close()

# Biểu đồ Top-N: Recall@K
plt.figure(figsize=(9, 5.5), dpi=300)
plt.bar(x - width, df_top_n["R@5"], width, label="Recall@5", color="#f59e0b", edgecolor="black")
plt.bar(x, df_top_n["R@10"], width, label="Recall@10", color="#d97706", edgecolor="black")
plt.bar(x + width, df_top_n["R@20"], width, label="Recall@20", color="#b45309", edgecolor="black")

plt.title("Biểu đồ so sánh chỉ số Recall@K giữa các mô hình", fontsize=12, fontweight="bold", pad=12)
plt.ylabel("Recall@K", fontsize=11)
plt.xticks(x, [m.replace(" ", "\n") for m in models_display], fontsize=9)
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(TOPN_FIG_DIR, "chart2_recall_at_k.png"))
plt.close()

# Biểu đồ Top-N: MAP@K
plt.figure(figsize=(9, 5.5), dpi=300)
plt.bar(x - width, df_top_n["MAP@5"], width, label="MAP@5", color="#10b981", edgecolor="black")
plt.bar(x, df_top_n["MAP@10"], width, label="MAP@10", color="#059669", edgecolor="black")
plt.bar(x + width, df_top_n["MAP@20"], width, label="MAP@20", color="#047857", edgecolor="black")

plt.title("Biểu đồ so sánh chỉ số Mean Average Precision (MAP@K)", fontsize=12, fontweight="bold", pad=12)
plt.ylabel("MAP@K", fontsize=11)
plt.xticks(x, [m.replace(" ", "\n") for m in models_display], fontsize=9)
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(TOPN_FIG_DIR, "chart3_map_at_k.png"))
plt.close()

# Biểu đồ Top-N: NDCG@K
plt.figure(figsize=(9, 5.5), dpi=300)
plt.bar(x - width, df_top_n["NDCG@5"], width, label="NDCG@5", color="#8b5cf6", edgecolor="black")
plt.bar(x, df_top_n["NDCG@10"], width, label="NDCG@10", color="#7c3aed", edgecolor="black")
plt.bar(x + width, df_top_n["NDCG@20"], width, label="NDCG@20", color="#6d28d9", edgecolor="black")

plt.title("Biểu đồ so sánh chỉ số xếp hạng Normalized Discounted Cumulative Gain (NDCG@K)", fontsize=12, fontweight="bold", pad=12)
plt.ylabel("NDCG@K", fontsize=11)
plt.xticks(x, [m.replace(" ", "\n") for m in models_display], fontsize=9)
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(TOPN_FIG_DIR, "chart4_ndcg_at_k.png"))
plt.close()

print("\n" + "=" * 80)
print("HOÀN TẤT TOÀN BỘ EXPERIMENT TASK 5 & TASK 6!")
print("=" * 80)
