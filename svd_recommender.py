"""
Mô hình SVD Baseline (PureSVD) cho Hệ thống Gợi ý Thời trang H&M.
Dựa trên phương pháp phân rã ma trận giá trị kỳ dị (Singular Value Decomposition)
trên ma trận tương tác ngầm định (Implicit Interaction Matrix) theo chuẩn Cremonesi et al. (RecSys 2010).

Phương pháp:
- Xây dựng ma trận tương tác thưa R (Users x Items) từ dữ liệu giao dịch H&M (tần suất hoặc tương tác nhị phân).
- Phân rã giá trị kỳ dị rút gọn (Truncated SVD / PureSVD):
    R ≈ U_k * Σ_k * V_k^T
  Trong đó:
    - k là số chiều nhân tố ẩn (Latent Factors).
    - P = U_k * Σ_k biểu diễn đặc trưng ẩn của người dùng (User Latent Factors).
    - Q = V_k biểu diễn đặc trưng ẩn của sản phẩm (Item Latent Factors).
- Điểm số dự đoán tương tác:
    Score(u, i) = P_u · Q_i^T
- Sinh danh sách gợi ý Top-N bằng cách sắp xếp theo Score(u, i) giảm dần,
  tự động loại bỏ các sản phẩm người dùng đã tương tác trong tập huấn luyện (Already Seen Items).

Đặc điểm và Hạn chế:
- Ưu điểm: Thuật toán kinh điển, có nghiệm giải tích tối ưu theo chuẩn Frobenius norm,
  huấn luyện nhanh, không cần khởi tạo ngẫu nhiên hay tối ưu hóa hàm mất mát phi lồi.
- Hạn chế: Coi tất cả các ô chưa quan sát (unobserved entries) là 0 (negative feedback)
  mà không có cơ chế trọng số tin cậy (Confidence Weighting) như Implicit ALS,
  dễ bị sai lệch khi ma trận tương tác cực kỳ thưa (sparsity > 99.5%).
"""

import os
import sys
import time
import pickle
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            pass


class SVDBaselineRecommender:
    """Mô hình gợi ý PureSVD Baseline trên ma trận tương tác ngầm định."""

    def __init__(self, n_factors: int = 40, random_state: int = 42):
        self.n_factors = n_factors
        self.random_state = random_state

        self.user_to_idx: Dict[str, int] = {}
        self.idx_to_user: Dict[int, str] = {}
        self.item_to_idx: Dict[Union[int, str], int] = {}
        self.idx_to_item: Dict[int, Union[int, str]] = {}

        self.user_factors: Optional[np.ndarray] = None  # Shape: (n_users, n_factors)
        self.item_factors: Optional[np.ndarray] = None  # Shape: (n_items, n_factors)
        self.singular_values: Optional[np.ndarray] = None  # Shape: (n_factors,)
        self.train_matrix: Optional[sp.csr_matrix] = None  # CSR (n_users, n_items)

        self.train_time: float = 0.0

    def fit(self, train_df: pd.DataFrame):
        """
        Huấn luyện mô hình PureSVD từ DataFrame tương tác.
        train_df phải chứa các cột: ['customer_id', 'article_id'] và tùy chọn ['freq' hoặc 'rating'].
        """
        start_t = time.time()
        print(f"--- [SVD Baseline] Huấn luyện PureSVD (Factors={self.n_factors}) ---")

        # 1. Tạo mapping Index hai chiều
        unique_users = train_df["customer_id"].unique()
        unique_items = train_df["article_id"].unique()

        self.user_to_idx = {u: idx for idx, u in enumerate(unique_users)}
        self.idx_to_user = {idx: u for idx, u in enumerate(unique_users)}
        self.item_to_idx = {i: idx for idx, i in enumerate(unique_items)}
        self.idx_to_item = {idx: i for idx, i in enumerate(unique_items)}

        n_users = len(unique_users)
        n_items = len(unique_items)

        # 2. Xây dựng ma trận tương tác thưa (Interaction Matrix)
        rows = train_df["customer_id"].map(self.user_to_idx).values
        cols = train_df["article_id"].map(self.item_to_idx).values

        if "freq" in train_df.columns:
            values = train_df["freq"].values.astype(np.float32)
        elif "rating" in train_df.columns:
            values = train_df["rating"].values.astype(np.float32)
        else:
            values = np.ones(len(train_df), dtype=np.float32)

        self.train_matrix = sp.csr_matrix(
            (values, (rows, cols)), shape=(n_users, n_items), dtype=np.float32
        )

        # 3. Phân rã ma trận giá trị kỳ dị rút gọn (Truncated SVD)
        # scipy svds yêu cầu k < min(n_users, n_items)
        k = min(self.n_factors, min(n_users, n_items) - 1)
        if k < 1:
            k = 1

        # Sử dụng scipy.sparse.linalg.svds
        # random_state được truyền qua np.random.RandomState
        rng = np.random.RandomState(self.random_state)
        # v0 làm vector khởi tạo ngẫu nhiên để đảm bảo tính tái lập (reproducibility)
        v0 = rng.randn(min(n_users, n_items))

        u, s, vt = svds(self.train_matrix, k=k, v0=v0)

        # svds sắp xếp giá trị kỳ dị theo thứ tự tăng dần, ta đảo ngược lại theo thứ tự giảm dần
        order = np.argsort(s)[::-1]
        u = u[:, order]
        s = s[order]
        vt = vt[order, :]

        self.singular_values = s
        # User representations: P = U * diag(S)
        self.user_factors = u * s  # shape: (n_users, k)
        # Item representations: Q = V = vt.T
        self.item_factors = vt.T  # shape: (n_items, k)

        self.train_time = time.time() - start_t
        print(
            f"-> SVD hoàn tất trong {self.train_time:.2f}s! "
            f"(Users {self.user_factors.shape}, Items {self.item_factors.shape}, S={s[:3].round(2)}...)"
        )
        return self

    def predict_score(self, user_id: str, article_id: Union[int, str]) -> float:
        """Dự đoán raw interaction score: P_u · Q_i^T."""
        u_idx = self.user_to_idx.get(user_id)
        i_idx = self.item_to_idx.get(article_id)
        if u_idx is None or i_idx is None:
            return 0.0
        return float(np.dot(self.user_factors[u_idx], self.item_factors[i_idx]))

    def recommend(
        self,
        user_id: str,
        top_n: int = 10,
        filter_already_liked_items: bool = True,
        candidate_items: Optional[List[Union[int, str]]] = None,
    ) -> List[dict]:
        """
        Sinh danh sách Top-N gợi ý cho user_id.
        Tự động loại bỏ các sản phẩm đã tương tác trong tập huấn luyện (nếu filter_already_liked_items=True).
        Trả về danh sách dict: [{'rank': 1, 'article_id': ..., 'score': ...}, ...]
        """
        u_idx = self.user_to_idx.get(user_id)
        if u_idx is None:
            return []

        # Vector hóa tính toàn bộ điểm số: scores = item_factors · user_factors[u]
        scores = np.dot(self.item_factors, self.user_factors[u_idx])

        # Loại bỏ các item đã xem/tương tác trong tập train
        if filter_already_liked_items and self.train_matrix is not None:
            seen_items = self.train_matrix[u_idx].indices
            scores[seen_items] = -np.inf

        # Nếu có danh sách candidate cụ thể
        if candidate_items is not None:
            mask = np.zeros(len(scores), dtype=bool)
            for item in candidate_items:
                idx = self.item_to_idx.get(item)
                if idx is not None:
                    mask[idx] = True
            scores[~mask] = -np.inf

        top_indices = np.argsort(scores)[::-1][:top_n]
        recs = []
        for rank, idx in enumerate(top_indices, 1):
            if np.isneginf(scores[idx]):
                break
            recs.append({
                "rank": rank,
                "article_id": self.idx_to_item[idx],
                "score": float(scores[idx]),
            })
        return recs

    def get_similar_items(self, article_id: Union[int, str], top_n: int = 10) -> List[dict]:
        """Tìm Top-N sản phẩm tương tự dựa trên Cosine similarity trong không gian ẩn SVD."""
        i_idx = self.item_to_idx.get(article_id)
        if i_idx is None or self.item_factors is None:
            return []

        item_vec = self.item_factors[i_idx]
        norms = np.linalg.norm(self.item_factors, axis=1)
        norms[norms == 0] = 1e-9
        item_norm = np.linalg.norm(item_vec)
        if item_norm == 0:
            item_norm = 1e-9

        sims = np.dot(self.item_factors, item_vec) / (norms * item_norm)
        sims[i_idx] = -1.0  # Loại trừ chính nó

        top_indices = np.argsort(sims)[::-1][:top_n]
        results = []
        for rank, idx in enumerate(top_indices, 1):
            results.append({
                "rank": rank,
                "article_id": self.idx_to_item[idx],
                "similarity": float(sims[idx]),
            })
        return results

    def save(self, filepath: str):
        """Lưu trữ mô hình ra đĩa (pickle)."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        print(f"[SVD Baseline] Đã lưu mô hình thành công tại: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "SVDBaselineRecommender":
        """Nạp mô hình đã lưu từ đĩa."""
        with open(filepath, "rb") as f:
            model = pickle.load(f)
        return model
