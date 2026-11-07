"""
Mô hình Baseline: Content-Based Filtering cho Hệ thống Gợi ý Thời trang H&M.
Áp dụng TF-IDF trên metadata sản phẩm và độ đo Cosine Similarity.
"""

import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error


class ContentBasedBaseline:
    def __init__(self, max_features=5000):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
        self.tfidf_matrix = None
        self.article_to_idx = {}
        self.idx_to_article = {}
        self.user_profiles = {}
        self.global_mean_rating = 3.0
        self.train_time = 0.0

    def fit(self, articles_df: pd.DataFrame, train_interactions: pd.DataFrame):
        """
        articles_df: chứa article_id và các trường mô tả
        train_interactions: customer_id, article_id, rating
        """
        start_t = time.time()
        print("--- [BASELINE] Huấn luyện mô hình Content-Based Filtering ---")

        # 1. Chuẩn hóa metadata bài viết
        df_items = articles_df.copy()
        for col in ["prod_name", "product_type_name", "product_group_name", 
                    "colour_group_name", "department_name", "detail_desc"]:
            if col in df_items.columns:
                df_items[col] = df_items[col].fillna("").astype(str)
            else:
                df_items[col] = ""

        # Ghép thành chuỗi văn bản mô tả đặc trưng (Item Profile)
        df_items["combined_text"] = (
            df_items["prod_name"] + " " +
            df_items["product_type_name"] + " " +
            df_items["product_group_name"] + " " +
            df_items["colour_group_name"] + " " +
            df_items["department_name"] + " " +
            df_items["detail_desc"]
        )

        df_items = df_items.drop_duplicates(subset=["article_id"]).reset_index(drop=True)
        self.article_to_idx = {int(art): idx for idx, art in enumerate(df_items["article_id"])}
        self.idx_to_article = {idx: int(art) for idx, art in enumerate(df_items["article_id"])}

        # 2. Tính TF-IDF
        print(f"-> Trích xuất TF-IDF cho {len(df_items):,} sản phẩm...")
        self.tfidf_matrix = self.vectorizer.fit_transform(df_items["combined_text"])

        # 3. Tính user profiles từ lịch sử tương tác
        print("-> Xây dựng User Profiles từ lịch sử tương tác...")
        self.global_mean_rating = float(train_interactions["rating"].mean())

        # Gom các item tương tác theo từng user
        user_grouped = train_interactions.groupby("customer_id")
        for user_id, group in user_grouped:
            valid_indices = []
            weights = []
            for _, row in group.iterrows():
                art = int(row["article_id"])
                if art in self.article_to_idx:
                    valid_indices.append(self.article_to_idx[art])
                    weights.append(float(row["rating"]))

            if valid_indices:
                weights = np.array(weights).reshape(-1, 1)
                item_vecs = self.tfidf_matrix[valid_indices].toarray()
                # User profile vector là trung bình có trọng số theo rating
                user_vec = np.sum(item_vecs * weights, axis=0) / (np.sum(weights) + 1e-9)
                # Chuẩn hóa L2 norm
                norm = np.linalg.norm(user_vec)
                if norm > 0:
                    user_vec = user_vec / norm
                self.user_profiles[user_id] = user_vec

        self.train_time = time.time() - start_t
        print(f"-> Huấn luyện Baseline hoàn tất trong {self.train_time:.2f}s!")
        return self

    def predict_rating(self, user_id: str, article_id: int) -> float:
        """Dự đoán điểm tương quan của người dùng với sản phẩm (thang 1-5)."""
        art_idx = self.article_to_idx.get(int(article_id))
        user_vec = self.user_profiles.get(user_id)

        if art_idx is None or user_vec is None:
            return self.global_mean_rating

        item_vec = self.tfidf_matrix[art_idx].toarray().flatten()
        dot_val = float(np.dot(user_vec, item_vec))
        sim = max(0.0, min(1.0, dot_val))
        # Map Cosine similarity [0, 1] sang thang rating [1.0, 5.0]
        pred_rating = 1.0 + 4.0 * sim
        return float(np.clip(pred_rating, 1.0, 5.0))

    def recommend(self, user_id: str, top_n=10, exclude_known=True, known_items=None):
        """Sinh Top-N gợi ý cho user."""
        user_vec = self.user_profiles.get(user_id)
        if user_vec is None:
            # Fallback phổ biến
            return []

        # Tính similarity giữa user vector và toàn bộ item TF-IDF matrix
        scores = self.tfidf_matrix.dot(user_vec)
        top_indices = np.argsort(scores)[::-1]

        recs = []
        known_set = set(known_items or [])
        for idx in top_indices:
            art = self.idx_to_article[idx]
            if exclude_known and art in known_set:
                continue
            recs.append((art, float(scores[idx])))
            if len(recs) >= top_n:
                break
        return recs

    def recommend_similar_items(self, article_id: int, top_n=10):
        """Tìm Top-N sản phẩm tương tự dựa trên Cosine similarity nội dung."""
        art_idx = self.article_to_idx.get(int(article_id))
        if art_idx is None:
            return []
        item_vec = self.tfidf_matrix[art_idx]
        sims = cosine_similarity(item_vec, self.tfidf_matrix).flatten()
        sims[art_idx] = -1.0 # Bỏ qua chính nó
        top_indices = np.argsort(sims)[::-1][:top_n]
        return [(self.idx_to_article[idx], float(sims[idx])) for idx in top_indices]

    def evaluate(self, test_df: pd.DataFrame):
        """Đánh giá RMSE và MAE trên tập test."""
        start_t = time.time()
        y_true = []
        y_pred = []
        for _, row in test_df.iterrows():
            u = row["customer_id"]
            i = int(row["article_id"])
            r_true = float(row["rating"])
            r_pred = self.predict_rating(u, i)
            y_true.append(r_true)
            y_pred.append(r_pred)

        pred_time = time.time() - start_t
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        return {
            "rmse": rmse,
            "mae": mae,
            "train_time": self.train_time,
            "pred_time": pred_time,
            "sample_count": len(y_true)
        }
