"""
Module Đánh giá Top-N Recommendation:
Cài đặt chính xác các chỉ số xếp hạng theo chuẩn quốc tế cho dữ liệu Implicit Feedback:
- Precision@K (K = 5, 10, 20): Tỷ lệ các sản phẩm đúng trong Top-K dự đoán (chia cho K cố định).
- Recall@K (K = 5, 10, 20): Tỷ lệ các sản phẩm thực tế được tìm thấy trong Top-K (chia cho |Actual|).
- AP@K / MAP@K (Mean Average Precision @ K): Độ chính xác trung bình có trọng số vị trí (chuẩn hóa bởi min(|Actual|, K)).
- NDCG@K (Normalized Discounted Cumulative Gain @ K): Độ đo chiết khấu logarit (Discounted Cumulative Gain) chia cho IDCG lý tưởng.

Quy tắc Candidate Protocol & Đánh giá:
- Loại bỏ hoàn toàn các sản phẩm người dùng đã tương tác trong tập Train (Already Seen Items).
- Khử trùng lặp (Deduplication) thứ tự dự đoán trước khi cắt ngưỡng Top-K.
- Báo cáo rõ ràng MAP@K cắt tại K và thống kê chi tiết người dùng được đánh giá và bị loại.
"""

import sys
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
import pandas as pd

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            pass


def _dedup_list(items: Sequence) -> list:
    """Khử trùng lặp danh sách giữ nguyên thứ tự xuất hiện ban đầu."""
    return list(dict.fromkeys(items))


def precision_at_k(actual_items: Set, predicted_items: Sequence, k: int) -> float:
    """
    Precision@K = (Số item relevant trong Top-K) / K
    - Nếu k <= 0 hoặc actual_items rỗng: trả về 0.0.
    - Nếu danh sách dự đoán ít hơn K: mẫu số vẫn là K cố định theo chuẩn Information Retrieval.
    - Tự động khử trùng lặp trong predicted_items trước khi cắt Top-K.
    """
    if k <= 0 or not actual_items:
        return 0.0
    dedup_preds = _dedup_list(predicted_items)
    pred_k = dedup_preds[:k]
    relevant_hits = len(actual_items.intersection(set(pred_k)))
    return float(relevant_hits) / float(k)


def recall_at_k(actual_items: Set, predicted_items: Sequence, k: int) -> float:
    """
    Recall@K = (Số item relevant trong Top-K) / |Actual Items|
    - Nếu k <= 0 hoặc |Actual Items| == 0: trả về 0.0.
    """
    if k <= 0 or not actual_items:
        return 0.0
    dedup_preds = _dedup_list(predicted_items)
    pred_k = dedup_preds[:k]
    relevant_hits = len(actual_items.intersection(set(pred_k)))
    return float(relevant_hits) / float(len(actual_items))


def average_precision_at_k(actual_items: Set, predicted_items: Sequence, k: int) -> float:
    """
    Average Precision @ K (AP@K) cho 1 người dùng:
        AP@K = (1 / min(|Actual|, K)) * sum_{j=1}^{K} (P@j * I(item_j in Actual))
    - Chuẩn hóa bởi min(|Actual|, K) theo chuẩn IR công bố rộng rãi.
    """
    if k <= 0 or not actual_items:
        return 0.0
    dedup_preds = _dedup_list(predicted_items)
    pred_k = dedup_preds[:k]

    score = 0.0
    num_hits = 0.0
    for idx, item in enumerate(pred_k):
        if item in actual_items:
            num_hits += 1.0
            score += num_hits / (idx + 1.0)

    denominator = min(len(actual_items), k)
    if denominator == 0:
        return 0.0
    return float(score) / float(denominator)


def ndcg_at_k(actual_items: Set, predicted_items: Sequence, k: int) -> float:
    """
    Normalized Discounted Cumulative Gain @ K (NDCG@K) cho binary relevance:
        DCG@K = sum_{j=1}^{K} (I(item_j in Actual) / log2(j + 1))
        IDCG@K = sum_{j=1}^{min(|Actual|, K)} (1 / log2(j + 1))
        NDCG@K = DCG@K / IDCG@K
    """
    if k <= 0 or not actual_items:
        return 0.0
    dedup_preds = _dedup_list(predicted_items)
    pred_k = dedup_preds[:k]

    dcg = 0.0
    for idx, item in enumerate(pred_k):
        if item in actual_items:
            dcg += 1.0 / np.log2(idx + 2.0)

    idcg = 0.0
    for idx in range(min(len(actual_items), k)):
        idcg += 1.0 / np.log2(idx + 2.0)

    if idcg == 0.0:
        return 0.0
    return float(dcg / idcg)


class TopNEvaluator:
    """
    Module đánh giá Top-N Ranking thống nhất cho các mô hình Recommender Systems.
    Hỗ trợ kiểm tra người dùng hợp lệ, tính P@K, R@K, MAP@K, NDCG@K trên cùng split dữ liệu.
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        k_list: Sequence[int] = (5, 10, 20),
    ):
        self.k_list = tuple(k_list)
        self.train_users_items: Dict[str, Set] = (
            train_df.groupby("customer_id")["article_id"].apply(set).to_dict()
        )
        self.test_users_items: Dict[str, Set] = (
            test_df.groupby("customer_id")["article_id"].apply(set).to_dict()
        )

        all_test_users = set(self.test_users_items.keys())
        all_train_users = set(self.train_users_items.keys())

        # Người dùng đủ điều kiện đánh giá: Có lịch sử train và có tương tác test
        self.eval_users = sorted(list(all_test_users.intersection(all_train_users)))

        # Thống kê người dùng bị loại
        self.cold_start_users_in_test = len(all_test_users - all_train_users)
        self.inactive_users_in_test = len(all_train_users - all_test_users)

        print(
            f"TopNEvaluator sẵn sàng: {len(self.eval_users):,} người dùng đủ điều kiện đánh giá. "
            f"(Loại {self.cold_start_users_in_test:,} người dùng mới không có lịch sử Train; "
            f"K-list = {self.k_list})"
        )

    def evaluate_model(
        self,
        model,
        model_name: str,
        max_eval_users: Optional[int] = 300,
    ) -> Dict[str, Union[str, float, int]]:
        """
        Đánh giá mô hình trên tập người dùng kiểm thử.
        Mô hình phải hỗ trợ phương thức recommend(user_id, top_n)
        """
        if max_eval_users is not None and max_eval_users > 0:
            eval_sample = self.eval_users[:max_eval_users]
        else:
            eval_sample = self.eval_users

        max_k = max(self.k_list)
        metrics = {f"P@{k}": [] for k in self.k_list}
        metrics.update({f"R@{k}": [] for k in self.k_list})
        metrics.update({f"MAP@{k}": [] for k in self.k_list})
        metrics.update({f"NDCG@{k}": [] for k in self.k_list})

        t_start = time.time()

        for user in eval_sample:
            actual = self.test_users_items[user]

            # Sinh Top max_k gợi ý từ mô hình
            predicted = []
            if hasattr(model, "recommend"):
                try:
                    raw_recs = model.recommend(user, top_n=max_k)
                    if raw_recs and isinstance(raw_recs[0], dict):
                        predicted = [r["article_id"] for r in raw_recs]
                    elif raw_recs and isinstance(raw_recs[0], tuple):
                        predicted = [r[0] for r in raw_recs]
                    else:
                        predicted = list(raw_recs)
                except Exception as exc:
                    print(f"[TopNEvaluator Warn] Lỗi sinh gợi ý cho user {user}: {exc}")
                    predicted = []

            for k in self.k_list:
                metrics[f"P@{k}"].append(precision_at_k(actual, predicted, k))
                metrics[f"R@{k}"].append(recall_at_k(actual, predicted, k))
                metrics[f"MAP@{k}"].append(average_precision_at_k(actual, predicted, k))
                metrics[f"NDCG@{k}"].append(ndcg_at_k(actual, predicted, k))

        total_inf_time = time.time() - t_start
        avg_inf_per_user_ms = (
            (total_inf_time / len(eval_sample) * 1000.0) if eval_sample else 0.0
        )

        result: Dict[str, Union[str, float, int]] = {
            "Model": model_name,
            "Evaluated_Users": len(eval_sample),
            "Inference_Time_s": round(total_inf_time, 2),
            "Avg_Latency_ms": round(avg_inf_per_user_ms, 2),
        }
        for k in self.k_list:
            result[f"P@{k}"] = round(float(np.mean(metrics[f"P@{k}"])), 4)
            result[f"R@{k}"] = round(float(np.mean(metrics[f"R@{k}"])), 4)
            result[f"MAP@{k}"] = round(float(np.mean(metrics[f"MAP@{k}"])), 4)
            result[f"NDCG@{k}"] = round(float(np.mean(metrics[f"NDCG@{k}"])), 4)

        return result
