"""
Mô hình Collaborative Filtering KNN: User-based & Item-based.
Hỗ trợ đầy đủ độ đo Cosine Similarity và Pearson Correlation.
Đánh giá độ chính xác bằng RMSE, MAE và hỗ trợ sinh gợi ý Top-N.
"""

import sys
import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")


class KNNRecommender:
    def __init__(self, mode="user_based", similarity_metric="cosine", k=20, min_common=1):
        """
        mode: 'user_based' hoặc 'item_based'
        similarity_metric: 'cosine' hoặc 'pearson'
        k: số lượng láng giềng gần nhất (neighbors)
        """
        assert mode in ["user_based", "item_based"], f"Invalid mode: {mode}"
        assert similarity_metric in ["cosine", "pearson"], f"Invalid metric: {similarity_metric}"

        self.mode = mode
        self.similarity_metric = similarity_metric
        self.k = k
        self.min_common = min_common

        self.user_to_idx = {}
        self.idx_to_user = {}
        self.item_to_idx = {}
        self.idx_to_item = {}

        self.user_means = None
        self.item_means = None
        self.global_mean = 3.0

        self.R = None          # Ma trận User-Item gốc
        self.sim_matrix = None # Ma trận tương đồng (User-User hoặc Item-Item)
        self.train_time = 0.0

    def fit(self, train_df: pd.DataFrame):
        """
        train_df: DataFrame với các cột ['customer_id', 'article_id', 'rating']
        """
        start_t = time.time()
        print(f"--- [KNN] Huấn luyện {self.mode.upper()} ({self.similarity_metric.upper()}, K={self.k}) ---")

        # 1. Tạo mapping Index
        unique_users = train_df["customer_id"].unique()
        unique_items = train_df["article_id"].unique()

        self.user_to_idx = {u: idx for idx, u in enumerate(unique_users)}
        self.idx_to_user = {idx: u for idx, u in enumerate(unique_users)}
        self.item_to_idx = {i: idx for idx, i in enumerate(unique_items)}
        self.idx_to_item = {idx: i for idx, i in enumerate(unique_items)}

        n_users = len(unique_users)
        n_items = len(unique_items)

        # 2. Xây dựng User-Item Rating Matrix
        rows = train_df["customer_id"].map(self.user_to_idx).values
        cols = train_df["article_id"].map(self.item_to_idx).values
        data = train_df["rating"].values.astype(np.float32)

        self.R = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
        R_dense = self.R.toarray()

        self.global_mean = float(np.mean(data))

        # Tính mean rating của từng user (chỉ tính trên các item đã tương tác)
        mask = R_dense > 0
        user_sums = np.sum(R_dense, axis=1)
        user_counts = np.sum(mask, axis=1)
        self.user_means = np.zeros(n_users, dtype=np.float32)
        nonzero_users = user_counts > 0
        self.user_means[nonzero_users] = user_sums[nonzero_users] / user_counts[nonzero_users]

        # Tính mean rating của từng item
        item_sums = np.sum(R_dense, axis=0)
        item_counts = np.sum(mask, axis=0)
        self.item_means = np.zeros(n_items, dtype=np.float32)
        nonzero_items = item_counts > 0
        self.item_means[nonzero_items] = item_sums[nonzero_items] / item_counts[nonzero_items]

        # 3. Tính toán ma trận độ đo tương đồng
        if self.mode == "user_based":
            if self.similarity_metric == "cosine":
                # Cosine Similarity giữa các User
                # cos(u, v) = (u . v) / (||u|| * ||v||)
                norms = np.linalg.norm(R_dense, axis=1, keepdims=True)
                norms[norms == 0] = 1e-9
                normalized_R = R_dense / norms
                self.sim_matrix = np.dot(normalized_R, normalized_R.T)

            elif self.similarity_metric == "pearson":
                # Pearson Correlation giữa các User (Mean-centered)
                # Centering rating của user bằng cách trừ user_mean
                centered_R = np.zeros_like(R_dense)
                for u in range(n_users):
                    if user_counts[u] > 0:
                        centered_R[u, mask[u]] = R_dense[u, mask[u]] - self.user_means[u]

                norms = np.linalg.norm(centered_R, axis=1, keepdims=True)
                norms[norms == 0] = 1e-9
                normalized_centered = centered_R / norms
                self.sim_matrix = np.dot(normalized_centered, normalized_centered.T)

            # Đặt đường chéo chính bằng 0 (không lấy chính nó làm neighbor)
            np.fill_diagonal(self.sim_matrix, 0.0)

        elif self.mode == "item_based":
            R_item = R_dense.T  # Shape: (n_items, n_users)
            mask_item = R_item > 0

            if self.similarity_metric == "cosine":
                # Cosine Similarity giữa các Item
                norms = np.linalg.norm(R_item, axis=1, keepdims=True)
                norms[norms == 0] = 1e-9
                normalized_item = R_item / norms
                self.sim_matrix = np.dot(normalized_item, normalized_item.T)

            elif self.similarity_metric == "pearson":
                # Adjusted Cosine / Pearson Correlation giữa các Item
                # Trừ điểm trung bình của User để loại bỏ độ khắt khe chấm điểm
                centered_R_item = np.zeros_like(R_item)
                for u in range(n_users):
                    if user_counts[u] > 0:
                        items_rated = np.where(R_dense[u] > 0)[0]
                        centered_R_item[items_rated, u] = R_dense[u, items_rated] - self.user_means[u]

                norms = np.linalg.norm(centered_R_item, axis=1, keepdims=True)
                norms[norms == 0] = 1e-9
                normalized_centered_item = centered_R_item / norms
                self.sim_matrix = np.dot(normalized_centered_item, normalized_centered_item.T)

            np.fill_diagonal(self.sim_matrix, 0.0)

        self.train_time = time.time() - start_t
        print(f"-> Huấn luyện xong trong {self.train_time:.2f}s!")
        return self

    def predict_rating(self, user_id: str, article_id) -> float:
        """
        Dự đoán rating của user cho một article cụ thể.
        Sử dụng công thức Resnick's kết hợp độ lệch trung bình.
        """
        u_idx = self.user_to_idx.get(user_id)
        i_idx = self.item_to_idx.get(article_id)

        # Cold-start fallback
        if u_idx is None and i_idx is None:
            return self.global_mean
        if u_idx is None:
            return float(self.item_means[i_idx]) if self.item_means[i_idx] > 0 else self.global_mean
        if i_idx is None:
            return float(self.user_means[u_idx]) if self.user_means[u_idx] > 0 else self.global_mean

        if self.mode == "user_based":
            # Lấy tất cả user đã tương tác với item i_idx
            # Dùng ma trận R
            rated_users = self.R[:, i_idx].nonzero()[0]
            if len(rated_users) == 0:
                return float(self.user_means[u_idx]) if self.user_means[u_idx] > 0 else self.global_mean

            # Lấy similarity của user u_idx với các user này
            sims = self.sim_matrix[u_idx, rated_users]

            # Lọc lấy K láng giềng có similarity cao nhất (và dương)
            top_k_order = np.argsort(sims)[::-1][:self.k]
            top_neighbors = rated_users[top_k_order]
            top_sims = sims[top_k_order]

            # Giữ lại các neighbor có sim > 0
            pos_mask = top_sims > 0
            if not np.any(pos_mask):
                return float(self.user_means[u_idx]) if self.user_means[u_idx] > 0 else self.global_mean

            valid_neighbors = top_neighbors[pos_mask]
            valid_sims = top_sims[pos_mask]

            ratings = np.array([self.R[v, i_idx] for v in valid_neighbors])

            if self.similarity_metric == "pearson":
                # Resnick formula: r_hat = mean_u + sum(sim * (r_v - mean_v)) / sum(|sim|)
                diffs = ratings - self.user_means[valid_neighbors]
                sim_sum = np.sum(np.abs(valid_sims))
                pred = self.user_means[u_idx] + (np.sum(valid_sims * diffs) / sim_sum) if sim_sum > 0 else self.user_means[u_idx]
            else:
                # Weighted average: sum(sim * r_v) / sum(sim)
                sim_sum = np.sum(valid_sims)
                pred = np.sum(valid_sims * ratings) / sim_sum if sim_sum > 0 else self.user_means[u_idx]

            return float(np.clip(pred, 1.0, 5.0))

        elif self.mode == "item_based":
            # Lấy tất cả item mà user u_idx đã từng đánh giá/mua
            rated_items = self.R[u_idx, :].nonzero()[1]
            if len(rated_items) == 0:
                return float(self.item_means[i_idx]) if self.item_means[i_idx] > 0 else self.global_mean

            # Lấy similarity giữa item i_idx và các item user đã đánh giá
            sims = self.sim_matrix[i_idx, rated_items]

            # Lấy K láng giềng tương đồng nhất
            top_k_order = np.argsort(sims)[::-1][:self.k]
            top_neighbors = rated_items[top_k_order]
            top_sims = sims[top_k_order]

            pos_mask = top_sims > 0
            if not np.any(pos_mask):
                return float(self.item_means[i_idx]) if self.item_means[i_idx] > 0 else self.global_mean

            valid_items = top_neighbors[pos_mask]
            valid_sims = top_sims[pos_mask]

            ratings = np.array([self.R[u_idx, j] for j in valid_items])
            sim_sum = np.sum(valid_sims)
            pred = np.sum(valid_sims * ratings) / sim_sum if sim_sum > 0 else self.item_means[i_idx]

            return float(np.clip(pred, 1.0, 5.0))

    def evaluate(self, test_df: pd.DataFrame):
        """Đánh giá RMSE và MAE trên tập dữ liệu kiểm thử test_df."""
        start_t = time.time()
        y_true = []
        y_pred = []

        for _, row in test_df.iterrows():
            u = row["customer_id"]
            i = row["article_id"]
            r_true = float(row["rating"])
            r_pred = self.predict_rating(u, i)

            y_true.append(r_true)
            y_pred.append(r_pred)

        pred_time = time.time() - start_t
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))

        return {
            "mode": self.mode,
            "metric": self.similarity_metric,
            "k": self.k,
            "rmse": rmse,
            "mae": mae,
            "train_time": self.train_time,
            "pred_time": pred_time,
            "sample_count": len(y_true)
        }

    def recommend(self, user_id: str, top_n=10, candidate_items=None):
        """Sinh danh sách Top-N gợi ý cho user_id kèm điểm số dự đoán."""
        u_idx = self.user_to_idx.get(user_id)
        if u_idx is None:
            return []

        # Các item user đã mua trong train
        known_items = set(self.R[u_idx, :].nonzero()[1])

        # Vectorized fast path for all items
        if candidate_items is None:
            if self.mode == "user_based":
                u_mean = float(self.user_means[u_idx])
                sims = self.sim_matrix[u_idx].copy()
                sims[u_idx] = 0.0
                top_k_users = np.argsort(sims)[::-1][:self.k]
                top_sims = sims[top_k_users]
                pos_mask = top_sims > 0
                if not np.any(pos_mask):
                    scores_all = self.item_means.copy()
                else:
                    valid_users = top_k_users[pos_mask]
                    valid_sims = top_sims[pos_mask]
                    R_dense = self.R[valid_users].toarray()
                    mask_rated = (R_dense > 0)
                    weights = valid_sims[:, None] * mask_rated
                    weight_sums = np.sum(weights, axis=0)

                    if self.similarity_metric == "pearson":
                        diffs = np.zeros_like(R_dense)
                        for idx_row, v in enumerate(valid_users):
                            diffs[idx_row, mask_rated[idx_row]] = R_dense[idx_row, mask_rated[idx_row]] - self.user_means[v]
                        weighted_diffs = np.sum(valid_sims[:, None] * diffs, axis=0)
                        scores_all = np.where(weight_sums > 0, u_mean + weighted_diffs / weight_sums, u_mean)
                    else:
                        weighted_r = np.sum(valid_sims[:, None] * R_dense, axis=0)
                        scores_all = np.where(weight_sums > 0, weighted_r / weight_sums, u_mean)

                scores_all[list(known_items)] = -np.inf
                top_indices = np.argsort(scores_all)[::-1][:top_n]
                return [(self.idx_to_item[i], float(scores_all[i])) for i in top_indices if not np.isneginf(scores_all[i])]

            elif self.mode == "item_based":
                rated_items = self.R[u_idx, :].nonzero()[1]
                if len(rated_items) == 0:
                    scores_all = self.item_means.copy()
                else:
                    r_user = self.R[u_idx, rated_items].toarray().flatten()
                    sub_sims = self.sim_matrix[:, rated_items].copy()
                    if len(rated_items) > self.k:
                        part_idx = np.argpartition(sub_sims, -self.k, axis=1)[:, -self.k:]
                        mask_top = np.zeros_like(sub_sims, dtype=bool)
                        np.put_along_axis(mask_top, part_idx, True, axis=1)
                        sub_sims = np.where(mask_top & (sub_sims > 0), sub_sims, 0.0)
                    else:
                        sub_sims = np.where(sub_sims > 0, sub_sims, 0.0)

                    sum_sims = np.sum(sub_sims, axis=1)
                    weighted_ratings = np.dot(sub_sims, r_user)
                    scores_all = np.where(sum_sims > 0, weighted_ratings / sum_sims, self.item_means)

                scores_all[list(known_items)] = -np.inf
                top_indices = np.argsort(scores_all)[::-1][:top_n]
                return [(self.idx_to_item[i], float(scores_all[i])) for i in top_indices if not np.isneginf(scores_all[i])]

        # Fallback if candidate_items is explicitly specified
        candidates = [i for i in candidate_items if self.item_to_idx.get(i) not in known_items]
        scores = []
        for art in candidates:
            score = self.predict_rating(user_id, art)
            scores.append((art, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]

