"""
Mô hình Implicit Matrix Factorization (ALS) cho Hệ thống Gợi ý Thời trang H&M.
Dựa trên thuật toán Alternating Least Squares (Hu, Koren, Volinsky 2008).
Xử lý dữ liệu phản hồi ngầm định (Implicit Feedback) với ma trận Confidence C_ui = 1 + alpha * R_ui.
Hỗ trợ:
- Dự đoán điểm rating phục vụ đo lường RMSE, MAE
- Sinh danh sách Top-N gợi ý loại bỏ các sản phẩm đã tương tác
- Tìm kiếm sản phẩm tương tự dựa trên Item Latent Factors
- Lưu và tải mô hình phục vụ Backend API
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import mean_squared_error, mean_absolute_error
import implicit

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")


class ImplicitMFRecommender:
    def __init__(self, factors=40, regularization=0.05, iterations=15, alpha=20, random_state=42):
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.random_state = random_state

        self.user_to_idx = {}
        self.idx_to_user = {}
        self.item_to_idx = {}
        self.idx_to_item = {}

        self.model = None
        self.user_factors = None
        self.item_factors = None
        self.train_matrix = None # CSR (users, items)

        self.calib_slope = 1.0
        self.calib_intercept = 2.5
        self.global_mean_rating = 3.0
        self.train_time = 0.0

    def fit(self, train_df: pd.DataFrame):
        """
        train_df: DataFrame gồm ['customer_id', 'article_id', 'rating' hoặc 'freq']
        """
        start_t = time.time()
        print(f"--- [MF/ALS] Huấn luyện Implicit ALS (Factors={self.factors}, Reg={self.regularization}, Alpha={self.alpha}, Iter={self.iterations}) ---")

        # 1. Tạo mapping Index
        unique_users = train_df["customer_id"].unique()
        unique_items = train_df["article_id"].unique()

        self.user_to_idx = {u: idx for idx, u in enumerate(unique_users)}
        self.idx_to_user = {idx: u for idx, u in enumerate(unique_users)}
        self.item_to_idx = {i: idx for idx, i in enumerate(unique_items)}
        self.idx_to_item = {idx: i for idx, i in enumerate(unique_items)}

        n_users = len(unique_users)
        n_items = len(unique_items)

        # 2. Xây dựng Confidence Matrix: C_ui = 1 + alpha * interaction
        rows = train_df["customer_id"].map(self.user_to_idx).values
        cols = train_df["article_id"].map(self.item_to_idx).values
        
        # Nếu có cột rating thì chuẩn hóa, nếu có freq thì lấy freq
        if "freq" in train_df.columns:
            counts = train_df["freq"].values.astype(np.float32)
        elif "rating" in train_df.columns:
            counts = train_df["rating"].values.astype(np.float32)
        else:
            counts = np.ones(len(train_df), dtype=np.float32)

        # Confidence: c_ui = 1 + alpha * count
        confidence = 1.0 + self.alpha * counts

        self.train_matrix = sp.csr_matrix((confidence, (rows, cols)), shape=(n_users, n_items))
        self.global_mean_rating = float(np.mean(counts))

        # 3. Khởi tạo và Huấn luyện mô hình Alternating Least Squares
        self.model = implicit.als.AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.random_state
        )

        # implicit nhận user_items CSR matrix
        self.model.fit(self.train_matrix, show_progress=False)

        # Trích xuất latent factors dạng numpy array
        self.user_factors = np.array(self.model.user_factors)
        self.item_factors = np.array(self.model.item_factors)

        # 4. Hiệu chuẩn tuyến tính (Calibration) từ dot-product sang thang rating thực tế
        # Lấy mẫu các tương tác huấn luyện để khớp hàm hồi quy: rating ~ a * dot_product + b
        sample_indices = np.random.choice(len(rows), size=min(10000, len(rows)), replace=False)
        sample_dots = np.sum(self.user_factors[rows[sample_indices]] * self.item_factors[cols[sample_indices]], axis=1)
        sample_ratings = counts[sample_indices]

        # Fit least squares slope and intercept
        var_dot = np.var(sample_dots)
        if var_dot > 1e-6:
            self.calib_slope = float(np.cov(sample_dots, sample_ratings)[0, 1] / var_dot)
            self.calib_intercept = float(np.mean(sample_ratings) - self.calib_slope * np.mean(sample_dots))
        else:
            self.calib_slope = 1.0
            self.calib_intercept = self.global_mean_rating

        self.train_time = time.time() - start_t
        print(f"-> Huấn luyện xong trong {self.train_time:.2f}s! (Factors shape: Users {self.user_factors.shape}, Items {self.item_factors.shape})")
        return self

    def predict_score(self, user_id: str, article_id) -> float:
        """Dự đoán raw latent score: x_u^T * y_i"""
        u_idx = self.user_to_idx.get(user_id)
        i_idx = self.item_to_idx.get(article_id)
        if u_idx is None or i_idx is None:
            return 0.0
        return float(np.dot(self.user_factors[u_idx], self.item_factors[i_idx]))

    def predict_rating(self, user_id: str, article_id) -> float:
        """Dự đoán điểm quy đổi sang thang rating [1.0, 5.0] phục vụ so sánh RMSE/MAE"""
        u_idx = self.user_to_idx.get(user_id)
        i_idx = self.item_to_idx.get(article_id)
        if u_idx is None and i_idx is None:
            return self.global_mean_rating
        if u_idx is None or i_idx is None:
            return self.global_mean_rating

        raw_dot = np.dot(self.user_factors[u_idx], self.item_factors[i_idx])
        pred_rating = self.calib_slope * raw_dot + self.calib_intercept
        return float(np.clip(pred_rating, 1.0, 5.0))

    def evaluate_rating(self, test_df: pd.DataFrame):
        """Đo lường RMSE và MAE trên tập kiểm thử test_df."""
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
            "model": "Implicit ALS",
            "factors": self.factors,
            "regularization": self.regularization,
            "iterations": self.iterations,
            "alpha": self.alpha,
            "rmse": rmse,
            "mae": mae,
            "train_time": self.train_time,
            "pred_time": pred_time,
            "sample_count": len(y_true)
        }

    def recommend(self, user_id: str, top_n=10, filter_already_liked_items=True):
        """
        Sinh Top-N gợi ý cho user_id.
        Tự động loại bỏ các sản phẩm user đã tương tác trong tập train.
        """
        u_idx = self.user_to_idx.get(user_id)
        if u_idx is None:
            return []

        # Tính điểm số cho toàn bộ item
        scores = np.dot(self.item_factors, self.user_factors[u_idx])

        if filter_already_liked_items and self.train_matrix is not None:
            seen_items = self.train_matrix[u_idx].indices
            scores[seen_items] = -np.inf

        top_indices = np.argsort(scores)[::-1][:top_n]
        recs = []
        for rank, idx in enumerate(top_indices, 1):
            if np.isneginf(scores[idx]):
                break
            recs.append({
                "rank": rank,
                "article_id": self.idx_to_item[idx],
                "score": float(scores[idx])
            })
        return recs

    def get_similar_items(self, article_id, top_n=10):
        """Tìm Top-N sản phẩm tương tự dựa trên Cosine similarity của Item Latent Factors."""
        i_idx = self.item_to_idx.get(article_id)
        if i_idx is None:
            return []

        item_vec = self.item_factors[i_idx]
        norms = np.linalg.norm(self.item_factors, axis=1)
        norms[norms == 0] = 1e-9
        item_norm = np.linalg.norm(item_vec)
        if item_norm == 0:
            item_norm = 1e-9

        # Cosine similarity
        sims = np.dot(self.item_factors, item_vec) / (norms * item_norm)
        sims[i_idx] = -1.0 # Bỏ qua chính nó

        top_indices = np.argsort(sims)[::-1][:top_n]
        results = []
        for rank, idx in enumerate(top_indices, 1):
            results.append({
                "rank": rank,
                "article_id": self.idx_to_item[idx],
                "similarity": float(sims[idx])
            })
        return results

    def save(self, filepath: str):
        """Lưu toàn bộ mô hình và các mapping ra file pickle."""
        data = {
            "factors": self.factors,
            "regularization": self.regularization,
            "iterations": self.iterations,
            "alpha": self.alpha,
            "user_to_idx": self.user_to_idx,
            "idx_to_user": self.idx_to_user,
            "item_to_idx": self.item_to_idx,
            "idx_to_item": self.idx_to_item,
            "user_factors": self.user_factors,
            "item_factors": self.item_factors,
            "calib_slope": self.calib_slope,
            "calib_intercept": self.calib_intercept,
            "global_mean_rating": self.global_mean_rating,
            "train_time": self.train_time
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        print(f"-> Đã lưu mô hình MF/ALS vào: {filepath}")

    @classmethod
    def load(cls, filepath: str):
        """Tải mô hình đã lưu từ file pickle."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        obj = cls(
            factors=data["factors"],
            regularization=data["regularization"],
            iterations=data["iterations"],
            alpha=data["alpha"]
        )
        obj.user_to_idx = data["user_to_idx"]
        obj.idx_to_user = data["idx_to_user"]
        obj.item_to_idx = data["item_to_idx"]
        obj.idx_to_item = data["idx_to_item"]
        obj.user_factors = data["user_factors"]
        obj.item_factors = data["item_factors"]
        obj.calib_slope = data["calib_slope"]
        obj.calib_intercept = data["calib_intercept"]
        obj.global_mean_rating = data["global_mean_rating"]
        obj.train_time = data["train_time"]
        return obj
