"""
Kịch bản Grid Search xác định bộ trọng số kết hợp tối ưu cho Hybrid Recommender:
Score_hybrid(u, i) = w_cf * S_cf(u, i) + w_profile * S_profile(u, i) + w_visual * S_visual(i, I_recent)

Thực nghiệm Grid Search trên tập Validation:
- w_cf in [0.3, 0.4, 0.5, 0.6, 0.7]
- w_profile in [0.1, 0.2, 0.3, 0.4]
- w_visual in [0.1, 0.2, 0.3]
sao cho w_cf + w_profile + w_visual = 1.0.

Tiêu chí lựa chọn: Điểm NDCG@10 và MAP@10 cao nhất trên tập Validation.
Lưu kết quả ra CSV và biểu đồ.
"""

import os
import sys
import time
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from hybrid_recommender import HybridRecommender
from top_n_evaluator import precision_at_k, recall_at_k, average_precision_at_k, ndcg_at_k

RES_DIR = os.path.join(BASE_DIR, "experiment_results")
FIG_DIR = os.path.join(BASE_DIR, "figures", "hybrid")
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

DATA_ROOT = os.getenv("DATASET_ROOT", r"E:\KLTN\XLDATA (2)\XLDATA")
TRAIN_PATH = os.path.join(DATA_ROOT, "output", "train_test", "transactions_train_5months.csv")
TEST_PATH = os.path.join(DATA_ROOT, "output", "train_test", "transactions_test_1month.csv")

if not os.path.exists(TRAIN_PATH):
    TRAIN_PATH = os.path.join(DATA_ROOT, "train_test", "transactions_train_5months.csv")
    TEST_PATH = os.path.join(DATA_ROOT, "train_test", "transactions_test_1month.csv")


def main():
    print("=" * 80)
    print("BẮT ĐẦU GRID SEARCH TỐI ƯU TRỌNG SỐ HYBRID RECOMMENDATION TRÊN TẬP VALIDATION")
    print("=" * 80)

    # 1. Chuẩn bị tập Validation từ dữ liệu tương tác
    df_te_raw = pd.read_csv(TEST_PATH, usecols=["customer_id", "article_id"], dtype={"article_id": int})
    df_tr_raw = pd.read_csv(TRAIN_PATH, usecols=["customer_id", "article_id"], dtype={"article_id": int})

    # Lấy tập người dùng có tương tác cả 2 tập và có trong catalog DB
    import sqlite3
    conn = sqlite3.connect(os.path.join(BASE_DIR, "users.db"))
    cat_rows = conn.execute("SELECT article_id FROM products").fetchall()
    conn.close()
    catalog_ids = set(r[0] for r in cat_rows)

    te_filtered = df_te_raw[df_te_raw["article_id"].isin(catalog_ids)]
    tr_users = set(df_tr_raw["customer_id"].unique())
    val_users = [u for u in te_filtered["customer_id"].unique() if u in tr_users]

    val_user_items = te_filtered.groupby("customer_id")["article_id"].apply(set).to_dict()
    eval_users = [u for u in val_users if len(val_user_items.get(u, set())) > 0][:150]

    print(f"-> Tập Validation gồm {len(eval_users)} người dùng lấy mẫu để tinh chỉnh trọng số.")

    # 2. Sinh lưới trọng số (Grid Weight Combinations)
    weight_combinations = []
    cf_weights = [0.30, 0.40, 0.50, 0.60, 0.70]
    prof_weights = [0.10, 0.20, 0.30, 0.40]
    vis_weights = [0.10, 0.20, 0.30]

    for w_c in cf_weights:
        for w_p in prof_weights:
            for w_v in vis_weights:
                if abs((w_c + w_p + w_v) - 1.0) < 1e-5:
                    weight_combinations.append((round(w_c, 2), round(w_p, 2), round(w_v, 2)))

    print(f"-> Tổng số cấu hình trọng số cần thử nghiệm: {len(weight_combinations)} cấu hình.")

    grid_results = []
    hybrid = HybridRecommender()

    for idx, (w_c, w_p, w_v) in enumerate(weight_combinations, 1):
        hybrid.w_cf = w_c
        hybrid.w_profile = w_p
        hybrid.w_visual = w_v

        k = 10
        p_list, r_list, map_list, ndcg_list = [], [], [], []

        for u in eval_users:
            actual = val_user_items[u]
            recs = hybrid.recommend(u, top_n=k, filter_already_seen=True)
            predicted = [r["article_id"] for r in recs]

            p_list.append(precision_at_k(actual, predicted, k))
            r_list.append(recall_at_k(actual, predicted, k))
            map_list.append(average_precision_at_k(actual, predicted, k))
            ndcg_list.append(ndcg_at_k(actual, predicted, k))

        avg_p = round(float(np.mean(p_list)), 4)
        avg_r = round(float(np.mean(r_list)), 4)
        avg_map = round(float(np.mean(map_list)), 4)
        avg_ndcg = round(float(np.mean(ndcg_list)), 4)

        grid_results.append({
            "Config_ID": idx,
            "w_cf": w_c,
            "w_profile": w_p,
            "w_visual": w_v,
            "Precision@10": avg_p,
            "Recall@10": avg_r,
            "MAP@10": avg_map,
            "NDCG@10": avg_ndcg,
        })
        print(f"  [{idx:2d}/{len(weight_combinations)}] (w_cf={w_c:.2f}, w_prof={w_p:.2f}, w_vis={w_v:.2f}) -> NDCG@10: {avg_ndcg:.4f} | MAP@10: {avg_map:.4f} | P@10: {avg_p:.4f}")

    df_grid = pd.DataFrame(grid_results)
    df_grid = df_grid.sort_values(by=["NDCG@10", "MAP@10"], ascending=False).reset_index(drop=True)

    out_csv = os.path.join(RES_DIR, "hybrid_grid_search_results.csv")
    df_grid.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n[✓] Đã lưu bảng kết quả Grid Search vào: {out_csv}")

    best_cfg = df_grid.iloc[0]
    print("\n" + "=" * 80)
    print(f"CẤU HÌNH TRỌNG SỐ HYBRID TỐI ƯU NHẤT TRÊN VALIDATION:")
    print(f"  • w_cf      : {best_cfg['w_cf']:.2f}")
    print(f"  • w_profile : {best_cfg['w_profile']:.2f}")
    print(f"  • w_visual  : {best_cfg['w_visual']:.2f}")
    print(f"  • NDCG@10   : {best_cfg['NDCG@10']:.4f}")
    print(f"  • MAP@10    : {best_cfg['MAP@10']:.4f}")
    print(f"  • P@10      : {best_cfg['Precision@10']:.4f}")
    print("=" * 80)

    # Vẽ biểu đồ kết quả Grid Search
    plt.figure(figsize=(10, 5), dpi=300)
    top_10_grid = df_grid.head(8)
    labels = [f"CF:{r['w_cf']:.1f}\nProf:{r['w_profile']:.1f}\nVis:{r['w_visual']:.1f}" for _, r in top_10_grid.iterrows()]
    x = np.arange(len(labels))
    plt.bar(x - 0.15, top_10_grid["NDCG@10"], width=0.3, label="NDCG@10", color="#6366f1", edgecolor="black")
    plt.bar(x + 0.15, top_10_grid["MAP@10"], width=0.3, label="MAP@10", color="#10b981", edgecolor="black")
    plt.title("Kết quả Grid Search Top các tổ hợp trọng số (Validation Set)", fontsize=12, fontweight="bold", pad=12)
    plt.xticks(x, labels, fontsize=9)
    plt.ylabel("Điểm đánh giá", fontsize=11)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "chart_hybrid_grid_search_weights.png"))
    plt.close()
    print("[✓] Đã tạo biểu đồ tại figures/hybrid/chart_hybrid_grid_search_weights.png")


if __name__ == "__main__":
    main()
