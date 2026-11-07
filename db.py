"""SQLite: tài khoản, lịch sử chat và conversation CRUD."""

import functools
import json
import re
import sqlite3
from typing import Optional, List, Dict, Any, Tuple, Union
import streamlit as st

from style_profile import normalize_style_preferences

DEFAULT_CHAT_GREETING = (
    "Chào bạn! Mình là Trợ lý Phong Cách. "
    "Bạn có thể hỏi mình về phối đồ, màu sắc, phong cách "
    "hoặc trang phục phù hợp theo từng dịp."
)


import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "users.db")

_DB_INITIALIZED = False


def init_db(conn=None):
    """Khởi tạo cấu trúc các bảng CSDL SQLite một lần duy nhất."""
    global _DB_INITIALIZED
    if _DB_INITIALIZED:
        return

    close_after = False
    if conn is None:
        conn = sqlite3.connect(DB_PATH, timeout=30.0, check_same_thread=False)
        conn.execute("PRAGMA journal_mode = MEMORY")
        conn.execute("PRAGMA temp_store = MEMORY")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA busy_timeout = 30000")
        close_after = True

    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password TEXT,
                role TEXT,
                full_name TEXT,
                dob TEXT,
                gender TEXT,
                email TEXT,
                phone TEXT,
                occupation TEXT,
                needs TEXT,
                usage_freq TEXT,
                is_verified INTEGER DEFAULT 0
            )
        """)

        # Nếu users.db cũ thiếu cột thì tự bổ sung
        existing_cols = [
            row[1]
            for row in conn.execute("PRAGMA table_info(users)").fetchall()
        ]

        profile_columns = {
            "full_name": "TEXT",
            "dob": "TEXT",
            "gender": "TEXT",
            "email": "TEXT",
            "phone": "TEXT",
            "occupation": "TEXT",
            "needs": "TEXT",
            "usage_freq": "TEXT",
            "is_verified": "INTEGER DEFAULT 0",
        }

        for col, col_type in profile_columns.items():
            if col not in existing_cols:
                conn.execute(f"ALTER TABLE users ADD COLUMN {col} {col_type}")

        # Bảng user_profiles phục vụ lưu trữ hồ sơ người dùng (Task 8)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                username TEXT PRIMARY KEY,
                occupation TEXT,
                usage_need TEXT,
                usage_frequency TEXT,
                style_preferences TEXT NOT NULL DEFAULT '{}',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
            )
        """)

        existing_profile_cols = {
            row[1] for row in conn.execute("PRAGMA table_info(user_profiles)")
        }
        if "style_preferences" not in existing_profile_cols:
            conn.execute(
                "ALTER TABLE user_profiles ADD COLUMN "
                "style_preferences TEXT NOT NULL DEFAULT '{}'"
            )

        # Bảng email_otps phục vụ xác minh Email OTP bảo mật (Task 7)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS email_otps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT,
                otp_hash TEXT,
                created_at REAL,
                expires_at REAL,
                attempts INTEGER DEFAULT 0,
                is_verified INTEGER DEFAULT 0
            )
        """)
        # Lịch sử chatbot theo từng tài khoản người dùng
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chat_history_username_id
            ON chat_history(username, id)
        """)

        # ========================================================
        # CHAT THREADS - mỗi tài khoản có nhiều cuộc trò chuyện
        # ========================================================
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                title TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chat_conversations_user_updated
            ON chat_conversations(username, updated_at DESC)
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                username TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(conversation_id)
                    REFERENCES chat_conversations(id)
                    ON DELETE CASCADE
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chat_messages_conversation
            ON chat_messages(conversation_id, id)
        """)

        # ========================================================
        # SẢN PHẨM, TƯƠNG TÁC VÀ GIỎ HÀNG THỜI TRANG H&M
        # ========================================================
        conn.execute("""
            CREATE TABLE IF NOT EXISTS products (
                article_id INTEGER PRIMARY KEY,
                product_code INTEGER,
                prod_name TEXT,
                product_type_name TEXT,
                product_group_name TEXT,
                colour_group_name TEXT,
                department_name TEXT,
                price REAL DEFAULT 0.05,
                image_url TEXT,
                detail_desc TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                article_id INTEGER NOT NULL,
                interaction_type TEXT NOT NULL,
                rating_score REAL DEFAULT 1.0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(article_id) REFERENCES products(article_id)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS cart (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                article_id INTEGER NOT NULL,
                quantity INTEGER DEFAULT 1,
                added_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(article_id) REFERENCES products(article_id)
            )
        """)

        # ========================================================
        # ĐƠN HÀNG VÀ CHI TIẾT ĐƠN HÀNG (ORDERS & ORDER_ITEMS)
        # ========================================================
        conn.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_code TEXT UNIQUE NOT NULL,
                username TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'Chờ xác nhận',
                subtotal_vnd INTEGER NOT NULL DEFAULT 0,
                shipping_fee_vnd INTEGER NOT NULL DEFAULT 0,
                discount_vnd INTEGER NOT NULL DEFAULT 0,
                total_vnd INTEGER NOT NULL DEFAULT 0,
                recipient_name TEXT,
                recipient_phone TEXT,
                shipping_address TEXT,
                payment_method TEXT DEFAULT 'COD',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(username) REFERENCES users(username)
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_orders_username
            ON orders(username, created_at DESC)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_orders_status
            ON orders(status, created_at DESC)
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS order_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id INTEGER NOT NULL,
                article_id INTEGER NOT NULL,
                prod_name TEXT NOT NULL,
                shop_name TEXT DEFAULT 'Routine Official Store',
                color TEXT DEFAULT 'Mặc định',
                size TEXT DEFAULT 'M',
                unit_price_vnd INTEGER NOT NULL DEFAULT 0,
                quantity INTEGER NOT NULL DEFAULT 1,
                line_total_vnd INTEGER NOT NULL DEFAULT 0,
                image_url TEXT,
                FOREIGN KEY(order_id) REFERENCES orders(id) ON DELETE CASCADE,
                FOREIGN KEY(article_id) REFERENCES products(article_id)
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_order_items_order
            ON order_items(order_id)
        """)

        # Bảng product_reviews: Lưu đánh giá sao thực tế của khách hàng (1..5 sao)
        # Tách biệt hoàn toàn với bảng interactions (vốn lưu trọng số hành vi ngầm định view/cart/like/purchase)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS product_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                article_id INTEGER NOT NULL,
                username TEXT NOT NULL,
                rating INTEGER NOT NULL CHECK(rating >= 1 AND rating <= 5),
                comment TEXT DEFAULT '',
                status TEXT NOT NULL DEFAULT 'approved',
                order_code TEXT DEFAULT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(article_id) REFERENCES products(article_id)
            )
        """)

        # Các chỉ mục tối ưu hóa truy vấn thực tế (loại bỏ full table scan)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_product_reviews_article_status
            ON product_reviews(article_id, status, rating)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_product_reviews_user_article
            ON product_reviews(username, article_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_interactions_user_id
            ON interactions(username, id DESC)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_interactions_article
            ON interactions(article_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_cart_user_article
            ON cart(username, article_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_products_group
            ON products(product_group_name)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_products_type
            ON products(product_type_name)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_products_price
            ON products(price)
        """)

        conn.commit()
        _DB_INITIALIZED = True
    except Exception as e:
        print(f"[DB Warning] init_db: {e}")
    finally:
        if close_after and conn:
            conn.close()


def get_db_connection():
    """
    Trả về kết nối SQLite an toàn với đường dẫn tuyệt đối DB_PATH.
    Được tối ưu cho ổ đĩa ngoài FAT32:
    - journal_mode = MEMORY & temp_store = MEMORY: Gộp chung trong 1 lần thực thi để giảm round-trip.
    - timeout = 30.0 trên sqlite3.connect đã thiết lập busy_timeout = 30000ms ở tầng C driver.
    """
    conn = sqlite3.connect(DB_PATH, timeout=30.0, check_same_thread=False)
    conn.execute("PRAGMA journal_mode = MEMORY;")

    if not _DB_INITIALIZED:
        init_db(conn)

    return conn


def load_chat_history(username: str, limit: int = 100):
    """Đọc lịch sử chat đã lưu của đúng tài khoản."""
    if not username:
        return [{"role": "assistant", "content": DEFAULT_CHAT_GREETING}]

    conn = get_db_connection()
    try:
        rows = conn.execute(
            """
            SELECT role, content
            FROM (
                SELECT id, role, content
                FROM chat_history
                WHERE username = ?
                ORDER BY id DESC
                LIMIT ?
            )
            ORDER BY id ASC
            """,
            (username, limit),
        ).fetchall()
    finally:
        conn.close()

    history = [
        {"role": role, "content": content}
        for role, content in rows
    ]

    if not history:
        history = [
            {"role": "assistant", "content": DEFAULT_CHAT_GREETING}
        ]

    return history


def save_chat_message(username: str, role: str, content: str):
    """Lưu từng message vào SQLite."""
    if not username or role not in ("user", "assistant"):
        return

    content = str(content or "").strip()
    if not content:
        return

    conn = get_db_connection()
    try:
        conn.execute(
            """
            INSERT INTO chat_history(username, role, content)
            VALUES (?, ?, ?)
            """,
            (username, role, content),
        )
        conn.commit()
    finally:
        conn.close()


def clear_chat_history_db(username: str):
    """Xóa toàn bộ lịch sử chatbot của đúng user."""
    if not username:
        return

    conn = get_db_connection()
    try:
        conn.execute(
            "DELETE FROM chat_history WHERE username = ?",
            (username,),
        )
        conn.commit()
    finally:
        conn.close()


def _conversation_title_from_prompt(prompt: str) -> str:
    text = re.sub(r"\s+", " ", str(prompt or "")).strip()
    if not text:
        return "Cuộc trò chuyện mới"

    if len(text) > 44:
        return text[:44].rstrip() + "…"

    return text


def create_chat_conversation(username: str, title: str) -> int:
    conn = get_db_connection()
    try:
        cur = conn.execute(
            """
            INSERT INTO chat_conversations(username, title)
            VALUES (?, ?)
            """,
            (
                username,
                title or "Cuộc trò chuyện mới",
            ),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def get_chat_conversations(username: str, limit: int = 30):
    if not username:
        return []

    conn = get_db_connection()
    try:
        rows = conn.execute(
            """
            SELECT id, title, created_at, updated_at
            FROM chat_conversations
            WHERE username = ?
            ORDER BY updated_at DESC, id DESC
            LIMIT ?
            """,
            (username, limit),
        ).fetchall()
    finally:
        conn.close()

    return [
        {
            "id": int(row[0]),
            "title": row[1],
            "created_at": row[2],
            "updated_at": row[3],
        }
        for row in rows
    ]


def load_conversation_messages(
    username: str,
    conversation_id: int,
    limit: int = 200,
):
    if not username or not conversation_id:
        return [
            {
                "role": "assistant",
                "content": DEFAULT_CHAT_GREETING,
            }
        ]

    conn = get_db_connection()
    try:
        rows = conn.execute(
            """
            SELECT role, content
            FROM (
                SELECT id, role, content
                FROM chat_messages
                WHERE username = ?
                  AND conversation_id = ?
                ORDER BY id DESC
                LIMIT ?
            )
            ORDER BY id ASC
            """,
            (
                username,
                int(conversation_id),
                int(limit),
            ),
        ).fetchall()
    finally:
        conn.close()

    history = [
        {
            "role": role,
            "content": content,
        }
        for role, content in rows
    ]

    if not history:
        history = [
            {
                "role": "assistant",
                "content": DEFAULT_CHAT_GREETING,
            }
        ]

    return history


def save_conversation_message(
    username: str,
    conversation_id: int,
    role: str,
    content: str,
):
    if (
        not username
        or not conversation_id
        or role not in ("user", "assistant")
    ):
        return

    content = str(content or "").strip()

    if not content:
        return

    conn = get_db_connection()
    try:
        conn.execute(
            """
            INSERT INTO chat_messages(
                conversation_id,
                username,
                role,
                content
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                int(conversation_id),
                username,
                role,
                content,
            ),
        )

        conn.execute(
            """
            UPDATE chat_conversations
            SET updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
              AND username = ?
            """,
            (
                int(conversation_id),
                username,
            ),
        )

        conn.commit()
    finally:
        conn.close()


def save_conversation_turn(
    username: str,
    conversation_id: int,
    user_content: str,
    assistant_content: str,
) -> bool:
    """Persist a complete reply atomically, only in its owner's conversation."""
    user_content = str(user_content or "").strip()
    assistant_content = str(assistant_content or "").strip()
    if not username or not conversation_id or not user_content or not assistant_content:
        return False

    conn = get_db_connection()
    try:
        with conn:
            # Acquire the write lock before checking ownership/deletion so a
            # concurrent delete cannot leave an orphaned turn behind.
            conn.execute("BEGIN IMMEDIATE")
            owner = conn.execute(
                "SELECT 1 FROM chat_conversations WHERE id = ? AND username = ?",
                (int(conversation_id), username),
            ).fetchone()
            if owner is None:
                return False
            conn.executemany(
                """
                INSERT INTO chat_messages(conversation_id, username, role, content)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (int(conversation_id), username, "user", user_content),
                    (int(conversation_id), username, "assistant", assistant_content),
                ],
            )
            conn.execute(
                "UPDATE chat_conversations SET updated_at = CURRENT_TIMESTAMP "
                "WHERE id = ? AND username = ?",
                (int(conversation_id), username),
            )
        return True
    finally:
        conn.close()


def delete_chat_conversation(username: str, conversation_id: int):
    if not username or not conversation_id:
        return

    conn = get_db_connection()
    try:
        # Xóa message trước để tương thích SQLite cũ
        conn.execute(
            """
            DELETE FROM chat_messages
            WHERE username = ?
              AND conversation_id = ?
            """,
            (
                username,
                int(conversation_id),
            ),
        )

        conn.execute(
            """
            DELETE FROM chat_conversations
            WHERE username = ?
              AND id = ?
            """,
            (
                username,
                int(conversation_id),
            ),
        )

        conn.commit()
    finally:
        conn.close()


def migrate_legacy_chat_history(username: str):
    """
    Chuyển dữ liệu bảng chat_history cũ sang hệ conversation mới.

    Quy tắc:
    - Nếu chưa có conversation mới:
      mỗi câu hỏi của user + các câu trả lời assistant kế tiếp
      sẽ thành một cuộc trò chuyện riêng.
    - Nếu trước đây đã bị gom thành đúng 1 conversation
      có tên "Lịch sử trước đây", tự tách lại một lần.
    """
    if not username:
        return

    conn = get_db_connection()
    try:
        conversations = conn.execute(
            """
            SELECT id, title
            FROM chat_conversations
            WHERE username = ?
            ORDER BY id ASC
            """,
            (username,),
        ).fetchall()

        legacy_rows = conn.execute(
            """
            SELECT role, content
            FROM chat_history
            WHERE username = ?
            ORDER BY id ASC
            """,
            (username,),
        ).fetchall()

        if not legacy_rows:
            return

        # Nếu user đã có conversation thật ngoài legacy thì không migrate lại.
        real_conversations = [
            row
            for row in conversations
            if row[1] != "Lịch sử trước đây"
        ]

        if real_conversations:
            return

        # Nếu phiên bản cũ đã tạo 1 thread "Lịch sử trước đây",
        # xóa thread đó để tách lại thành nhiều thread.
        for conv_id, title in conversations:
            if title == "Lịch sử trước đây":
                conn.execute(
                    """
                    DELETE FROM chat_messages
                    WHERE username = ?
                      AND conversation_id = ?
                    """,
                    (username, int(conv_id)),
                )

                conn.execute(
                    """
                    DELETE FROM chat_conversations
                    WHERE username = ?
                      AND id = ?
                    """,
                    (username, int(conv_id)),
                )

        conn.commit()
    finally:
        conn.close()

    # Gom dữ liệu cũ theo từng lượt user.
    groups = []
    current = []

    for role, content in legacy_rows:
        if role == "user":
            if current:
                groups.append(current)

            current = [(role, content)]
        else:
            if not current:
                # greeting/assistant trước câu hỏi đầu tiên:
                # bỏ qua vì mỗi thread mới đã có greeting riêng.
                continue

            current.append((role, content))

    if current:
        groups.append(current)

    # Nếu lịch sử cũ không có user turn thì không tạo thread.
    for group in groups:
        first_user = next(
            (
                content
                for role, content in group
                if role == "user"
            ),
            "",
        )

        if not first_user:
            continue

        conv_id = create_chat_conversation(
            username,
            _conversation_title_from_prompt(first_user),
        )

        for role, content in group:
            save_conversation_message(
                username,
                conv_id,
                role,
                content,
            )


# ========================================================
# CÁC THAO TÁC CRUD CHO SẢN PHẨM, GIỎ HÀNG VÀ TƯƠNG TÁC
# ========================================================

def populate_initial_products_if_empty(csv_path: str, limit: int = 500):
    """Nạp một số sản phẩm ban đầu từ articles.csv vào bảng SQLite products."""
    conn = get_db_connection()
    try:
        count = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        if count > 0:
            return

        import os
        import pandas as pd
        if not os.path.exists(csv_path):
            return

        df = pd.read_csv(csv_path, nrows=limit)
        # Đọc giá từ transactions_train.csv nếu có
        trans_csv = os.path.join(os.path.dirname(csv_path), "transactions_train.csv")
        price_map = {}
        if os.path.exists(trans_csv):
            try:
                for chunk in pd.read_csv(trans_csv, usecols=["t_dat", "article_id", "price"], chunksize=500000):
                    m = chunk[chunk["article_id"].isin(df["article_id"])]
                    if not m.empty:
                        m_sorted = m.sort_values(by=["article_id", "t_dat"], ascending=[True, False])
                        for art_id_val, grp in m_sorted.groupby("article_id"):
                            if art_id_val not in price_map:
                                price_map[art_id_val] = float(grp["price"].iloc[0])
            except Exception:
                pass

        rows_to_insert = []
        for _, row in df.iterrows():
            art_id = int(row.get("article_id", 0))
            art_str = f"{art_id:010d}"
            subfolder = art_str[:3]
            prod_name = str(row.get("prod_name", "")).strip()
            type_name = str(row.get("product_type_name", "")).strip()
            group_name = str(row.get("product_group_name", "")).strip()
            color = str(row.get("colour_group_name", "")).strip()
            dept = str(row.get("department_name", "")).strip()
            price = price_map.get(art_id, 0.025)
            img = f"output/images/{subfolder}/{art_str}.jpg"
            desc = str(row.get("detail_desc", "")).strip()

            rows_to_insert.append((art_id, int(row.get("product_code", 0)), prod_name, type_name, group_name, color, dept, price, img, desc))

        conn.executemany("""
            INSERT OR IGNORE INTO products
            (article_id, product_code, prod_name, product_type_name, product_group_name, colour_group_name, department_name, price, image_url, detail_desc)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows_to_insert)
        conn.commit()
    finally:
        conn.close()


VALID_REVIEW_STATUSES = ("approved", "active", "pending", "rejected", "hidden", "deleted")


def validate_review_rating(rating: Any) -> Optional[int]:
    """
    Xác thực điểm đánh giá sản phẩm thuộc miền giá trị nguyên 1..5.
    Từ chối None, bool, chuỗi không hợp lệ, số thực lẻ hoặc giá trị ngoài [1, 5].
    """
    if rating is None or isinstance(rating, bool):
        return None
    if isinstance(rating, int):
        return rating if 1 <= rating <= 5 else None
    if isinstance(rating, float):
        if rating.is_integer() and 1 <= int(rating) <= 5:
            return int(rating)
        return None
    if isinstance(rating, str):
        s = rating.strip()
        if not s:
            return None
        try:
            f = float(s)
            if f.is_integer() and 1 <= int(f) <= 5:
                return int(f)
        except ValueError:
            return None
    return None


def _invalidate_product_rating_caches():
    """Làm mới các bộ đệm danh mục khi đánh giá sản phẩm được thêm, sửa hoặc xóa."""
    try:
        if hasattr(get_product_rating_stats, "cache_clear"):
            get_product_rating_stats.cache_clear()
        import sys
        if "ui.product_view" in sys.modules:
            pv = sys.modules["ui.product_view"]
            if hasattr(pv, "_get_cached_catalog") and hasattr(pv._get_cached_catalog, "clear"):
                pv._get_cached_catalog.clear()
        if "chatbot.product_retriever" in sys.modules:
            pr = sys.modules["chatbot.product_retriever"]
            if hasattr(pr, "ProductRetriever"):
                pr.ProductRetriever._catalog_cache = None
    except Exception:
        pass


@functools.lru_cache(maxsize=2048)
def get_product_rating_stats(article_id: int) -> dict:
    """
    Tính điểm trung bình (rating_average) và số lượng đánh giá (rating_count)
    từ đúng tập đánh giá hợp lệ (status='approved' và rating trong [1, 5]).
    Nếu chưa có đánh giá hợp lệ: trả rating_average = None, rating_count = 0.
    Giữ nguyên độ chính xác số thực khi tính toán; chỉ làm tròn lúc hiển thị.
    """
    try:
        art_id = int(article_id)
    except (TypeError, ValueError):
        return {"rating_average": None, "rating_count": 0}

    conn = get_db_connection()
    try:
        row = conn.execute(
            """
            SELECT COUNT(*) AS cnt, AVG(CAST(rating AS REAL)) AS avg_rating
            FROM product_reviews
            WHERE article_id = ? AND status IN ('approved', 'active') AND rating >= 1 AND rating <= 5
            """,
            (art_id,),
        ).fetchone()
        cnt = int(row[0] or 0) if row else 0
        avg_val = float(row[1]) if (row and cnt > 0 and row[1] is not None) else None
        return {
            "rating_average": avg_val,
            "rating_count": cnt,
        }
    finally:
        conn.close()


def get_products_rating_stats_batch(article_ids: Optional[list] = None) -> dict[int, dict]:
    """Lấy thống kê đánh giá hợp lệ theo lô cho danh sách mã sản phẩm trong 1 truy vấn."""
    conn = get_db_connection()
    try:
        if article_ids is not None:
            ids = list(dict.fromkeys(int(x) for x in article_ids if x is not None))
            if not ids:
                return {}
            placeholders = ",".join("?" for _ in ids)
            rows = conn.execute(
                f"""
                SELECT article_id, COUNT(*) AS cnt, AVG(CAST(rating AS REAL)) AS avg_rating
                FROM product_reviews
                WHERE status IN ('approved', 'active') AND rating >= 1 AND rating <= 5
                  AND article_id IN ({placeholders})
                GROUP BY article_id
                """,
                ids,
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT article_id, COUNT(*) AS cnt, AVG(CAST(rating AS REAL)) AS avg_rating
                FROM product_reviews
                WHERE status IN ('approved', 'active') AND rating >= 1 AND rating <= 5
                GROUP BY article_id
                """
            ).fetchall()

        stats_map: dict[int, dict] = {}
        for r in rows:
            cnt = int(r[1] or 0)
            if cnt > 0 and r[2] is not None:
                stats_map[int(r[0])] = {
                    "rating_average": float(r[2]),
                    "rating_count": cnt,
                }
        return stats_map
    finally:
        conn.close()


def add_product_review(
    username: str,
    article_id: int,
    rating: Any,
    comment: str = "",
    status: str = "approved",
    order_code: Optional[str] = None,
    upsert_by_user: bool = False,
) -> tuple[bool, str, Optional[int]]:
    """
    Thêm (hoặc cập nhật nếu upsert_by_user=True) đánh giá sản phẩm của khách hàng.
    Kiểm tra chặt miền giá trị 1..5 và trạng thái hợp lệ.
    """
    uname = str(username or "").strip()
    if not uname:
        return False, "Tên người dùng không được để trống.", None
    try:
        art_id = int(article_id)
        if art_id <= 0:
            return False, "Mã sản phẩm không hợp lệ.", None
    except (TypeError, ValueError):
        return False, "Mã sản phẩm không hợp lệ.", None

    valid_rating = validate_review_rating(rating)
    if valid_rating is None:
        return False, "Điểm đánh giá không hợp lệ (chỉ chấp nhận số nguyên từ 1 đến 5 sao).", None

    clean_status = str(status or "approved").strip().lower()
    if clean_status not in VALID_REVIEW_STATUSES:
        return False, f"Trạng thái đánh giá '{status}' không hợp lệ.", None

    clean_comment = str(comment or "").strip()
    clean_order = str(order_code).strip() if order_code else None

    conn = get_db_connection()
    try:
        existing_id = None
        if upsert_by_user:
            if clean_order:
                row = conn.execute(
                    "SELECT id FROM product_reviews WHERE username = ? AND article_id = ? AND order_code = ? ORDER BY id DESC LIMIT 1",
                    (uname, art_id, clean_order),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT id FROM product_reviews WHERE username = ? AND article_id = ? ORDER BY id DESC LIMIT 1",
                    (uname, art_id),
                ).fetchone()
            if row:
                existing_id = int(row[0])

        if existing_id is not None:
            conn.execute(
                """
                UPDATE product_reviews
                SET rating = ?, comment = ?, status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (valid_rating, clean_comment, clean_status, existing_id),
            )
            review_id = existing_id
        else:
            cur = conn.execute(
                """
                INSERT INTO product_reviews (article_id, username, rating, comment, status, order_code)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (art_id, uname, valid_rating, clean_comment, clean_status, clean_order),
            )
            review_id = int(cur.lastrowid)

        conn.commit()
    finally:
        conn.close()

    _invalidate_product_rating_caches()
    return True, "Đã ghi nhận đánh giá sản phẩm thành công!", review_id


def submit_product_review(
    username: str,
    article_id: int,
    rating: Any,
    comment: str = "",
    status: str = "approved",
    order_code: Optional[str] = None,
    upsert_by_user: bool = True,
) -> tuple[bool, str, Optional[int]]:
    """Gửi đánh giá sản phẩm (mặc định cập nhật bản ghi hiện có của cùng người dùng & sản phẩm để chống trùng lặp)."""
    return add_product_review(
        username=username,
        article_id=article_id,
        rating=rating,
        comment=comment,
        status=status,
        order_code=order_code,
        upsert_by_user=upsert_by_user,
    )


def update_product_review(
    review_id: int,
    rating: Any = None,
    comment: Optional[str] = None,
    status: Optional[str] = None,
) -> tuple[bool, str]:
    """Cập nhật nội dung, số sao hoặc trạng thái duyệt của một đánh giá."""
    conn = get_db_connection()
    try:
        row = conn.execute("SELECT id, rating, comment, status FROM product_reviews WHERE id = ?", (int(review_id),)).fetchone()
        if not row:
            return False, "Không tìm thấy đánh giá cần cập nhật."

        new_rating = row[1]
        if rating is not None:
            checked = validate_review_rating(rating)
            if checked is None:
                return False, "Điểm đánh giá không hợp lệ (chỉ chấp nhận từ 1 đến 5 sao)."
            new_rating = checked

        new_comment = row[2] if comment is None else str(comment).strip()
        new_status = row[3]
        if status is not None:
            clean_st = str(status).strip().lower()
            if clean_st not in VALID_REVIEW_STATUSES:
                return False, f"Trạng thái '{status}' không hợp lệ."
            new_status = clean_st

        conn.execute(
            """
            UPDATE product_reviews
            SET rating = ?, comment = ?, status = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (new_rating, new_comment, new_status, int(review_id)),
        )
        conn.commit()
    finally:
        conn.close()

    _invalidate_product_rating_caches()
    return True, "Đã cập nhật đánh giá thành công."


def delete_product_review(
    review_id: Optional[int] = None,
    username: Optional[str] = None,
    article_id: Optional[int] = None,
    hard_delete: bool = True,
) -> bool:
    """Xóa (hoặc ẩn với status='deleted' nếu hard_delete=False) đánh giá theo ID hoặc (username, article_id) và làm mới thống kê."""
    conn = get_db_connection()
    try:
        if review_id is not None:
            if hard_delete:
                cur = conn.execute("DELETE FROM product_reviews WHERE id = ?", (int(review_id),))
            else:
                cur = conn.execute(
                    "UPDATE product_reviews SET status = 'deleted', updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (int(review_id),),
                )
        elif username and article_id is not None:
            if hard_delete:
                cur = conn.execute(
                    "DELETE FROM product_reviews WHERE username = ? AND article_id = ?",
                    (str(username).strip(), int(article_id)),
                )
            else:
                cur = conn.execute(
                    "UPDATE product_reviews SET status = 'deleted', updated_at = CURRENT_TIMESTAMP WHERE username = ? AND article_id = ?",
                    (str(username).strip(), int(article_id)),
                )
        elif article_id is not None:
            cur = conn.execute("DELETE FROM product_reviews WHERE article_id = ?", (int(article_id),))
        else:
            return False
        conn.commit()
        deleted = cur.rowcount > 0
    finally:
        conn.close()

    if deleted:
        _invalidate_product_rating_caches()
    return deleted


def get_product_reviews(
    article_id: int,
    approved_only: bool = True,
    limit: int = 50,
    include_inactive: Optional[bool] = None,
) -> list[dict]:
    """Lấy danh sách đánh giá của một sản phẩm."""
    if include_inactive is not None:
        approved_only = not bool(include_inactive)
    conn = get_db_connection()
    try:
        if approved_only:
            rows = conn.execute(
                """
                SELECT id, article_id, username, rating, comment, status, order_code, created_at, updated_at
                FROM product_reviews
                WHERE article_id = ? AND status IN ('approved', 'active') AND rating >= 1 AND rating <= 5
                ORDER BY id DESC LIMIT ?
                """,
                (int(article_id), int(limit)),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id, article_id, username, rating, comment, status, order_code, created_at, updated_at
                FROM product_reviews
                WHERE article_id = ?
                ORDER BY id DESC LIMIT ?
                """,
                (int(article_id), int(limit)),
            ).fetchall()
        cols = ["id", "article_id", "username", "rating", "comment", "status", "order_code", "created_at", "updated_at"]
        return [dict(zip(cols, r)) for r in rows]
    finally:
        conn.close()


def get_all_products(limit: int = 40, category: str = None, search: str = None):
    """Lấy danh sách sản phẩm hiển thị trên giao diện Grid View kèm thống kê đánh giá thực tế."""
    conn = get_db_connection()
    try:
        query = """
            SELECT p.article_id, p.prod_name, p.product_type_name, p.product_group_name,
                   p.colour_group_name, p.price, p.image_url, p.detail_desc,
                   rv.rating_average, COALESCE(rv.rating_count, 0) AS rating_count
            FROM products p
            LEFT JOIN (
                SELECT article_id,
                       AVG(CAST(rating AS REAL)) AS rating_average,
                       COUNT(*) AS rating_count
                FROM product_reviews
                WHERE status IN ('approved', 'active') AND rating >= 1 AND rating <= 5
                GROUP BY article_id
            ) rv ON p.article_id = rv.article_id
            WHERE 1=1
        """
        params = []

        if category and category != "Tất cả":
            query += " AND p.product_group_name = ?"
            params.append(category)

        if search:
            query += " AND (p.prod_name LIKE ? OR p.detail_desc LIKE ?)"
            params.extend([f"%{search}%", f"%{search}%"])

        query += " ORDER BY p.article_id DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        results = []
        for r in rows:
            cnt = int(r[9] or 0)
            avg_val = float(r[8]) if (cnt > 0 and r[8] is not None) else None
            results.append({
                "article_id": r[0],
                "prod_name": r[1],
                "product_type_name": r[2],
                "product_group_name": r[3],
                "colour_group_name": r[4],
                "price": r[5],
                "image_url": r[6],
                "detail_desc": r[7],
                "rating_average": avg_val,
                "rating_count": cnt,
            })
        return results
    finally:
        conn.close()


def get_products_by_ids(article_ids):
    """Read current product metadata and valid rating stats in one query, independent of model caches."""
    ids = list(dict.fromkeys(int(article_id) for article_id in article_ids if article_id is not None))
    if not ids:
        return {}
    conn = get_db_connection()
    try:
        placeholders = ",".join("?" for _ in ids)
        rows = conn.execute(
            f"""
            SELECT p.article_id, p.prod_name, p.product_type_name, p.product_group_name,
                   p.colour_group_name, p.price, p.image_url, p.detail_desc,
                   rv.rating_average, COALESCE(rv.rating_count, 0) AS rating_count
            FROM products p
            LEFT JOIN (
                SELECT article_id,
                       AVG(CAST(rating AS REAL)) AS rating_average,
                       COUNT(*) AS rating_count
                FROM product_reviews
                WHERE status IN ('approved', 'active') AND rating >= 1 AND rating <= 5
                GROUP BY article_id
            ) rv ON p.article_id = rv.article_id
            WHERE p.article_id IN ({placeholders})
            """,
            ids,
        ).fetchall()
        result = {}
        for r in rows:
            cnt = int(r[9] or 0)
            avg_val = float(r[8]) if (cnt > 0 and r[8] is not None) else None
            result[r[0]] = {
                "article_id": r[0],
                "prod_name": r[1],
                "product_type_name": r[2],
                "product_group_name": r[3],
                "colour_group_name": r[4],
                "price": r[5],
                "image_url": r[6],
                "detail_desc": r[7],
                "rating_average": avg_val,
                "rating_count": cnt,
            }
        return result
    finally:
        conn.close()


def get_product_by_id(article_id: int):
    """Lấy chi tiết một sản phẩm kèm thống kê đánh giá hợp lệ."""
    conn = get_db_connection()
    try:
        row = conn.execute(
            """
            SELECT p.article_id, p.prod_name, p.product_type_name, p.product_group_name,
                   p.colour_group_name, p.price, p.image_url, p.detail_desc,
                   rv.rating_average, COALESCE(rv.rating_count, 0) AS rating_count
            FROM products p
            LEFT JOIN (
                SELECT article_id,
                       AVG(CAST(rating AS REAL)) AS rating_average,
                       COUNT(*) AS rating_count
                FROM product_reviews
                WHERE article_id = ? AND status IN ('approved', 'active') AND rating >= 1 AND rating <= 5
                GROUP BY article_id
            ) rv ON p.article_id = rv.article_id
            WHERE p.article_id = ?
            """,
            (article_id, article_id),
        ).fetchone()
        if not row:
            return None
        cnt = int(row[9] or 0)
        avg_val = float(row[8]) if (cnt > 0 and row[8] is not None) else None
        return {
            "article_id": row[0],
            "prod_name": row[1],
            "product_type_name": row[2],
            "product_group_name": row[3],
            "colour_group_name": row[4],
            "price": row[5],
            "image_url": row[6],
            "detail_desc": row[7],
            "rating_average": avg_val,
            "rating_count": cnt,
        }
    finally:
        conn.close()


def record_interaction(username: str, article_id: int, interaction_type: str, score: float = 1.0):
    """Ghi nhận hành vi của người dùng (xem, thích, thêm giỏ, mua, đánh giá)."""
    if not username or not article_id:
        return
    conn = get_db_connection()
    try:
        conn.execute("""
            INSERT INTO interactions (username, article_id, interaction_type, rating_score)
            VALUES (?, ?, ?, ?)
        """, (username, article_id, interaction_type, score))
        conn.commit()
    finally:
        conn.close()
    try:
        import sys
        if "hybrid_recommender" in sys.modules:
            hr = getattr(sys.modules["hybrid_recommender"], "hybrid_recommender", None)
            if hr is not None and hasattr(hr, "_user_interactions_cache"):
                hr._user_interactions_cache.pop(username, None)
    except Exception:
        pass


def add_to_cart(username: str, article_id: int, quantity: int = 1) -> bool:
    """Thêm sản phẩm vào giỏ hàng."""
    if not username or not article_id:
        return False
    conn = get_db_connection()
    try:
        existing = conn.execute("SELECT id, quantity FROM cart WHERE username = ? AND article_id = ?", (username, article_id)).fetchone()
        if existing:
            conn.execute("UPDATE cart SET quantity = quantity + ? WHERE id = ?", (quantity, existing[0]))
        else:
            conn.execute("INSERT INTO cart (username, article_id, quantity) VALUES (?, ?, ?)", (username, article_id, quantity))
        conn.commit()
    except Exception:
        return False
    finally:
        conn.close()
    record_interaction(username, article_id, "cart", 2.0)
    return True


def get_user_cart(username: str):
    """Lấy các sản phẩm trong giỏ hàng kèm thông tin chi tiết."""
    if not username:
        return []
    conn = get_db_connection()
    try:
        rows = conn.execute("""
            SELECT c.id, c.article_id, c.quantity, p.prod_name, p.price, p.product_group_name, p.image_url
            FROM cart c
            JOIN products p ON c.article_id = p.article_id
            WHERE c.username = ?
            ORDER BY c.added_at DESC
        """, (username,)).fetchall()
        cols = ["cart_id", "article_id", "quantity", "prod_name", "price", "product_group_name", "image_url"]
        return [dict(zip(cols, r)) for r in rows]
    finally:
        conn.close()


def remove_from_cart(username: str, cart_id: int):
    """Xóa mục khỏi giỏ hàng."""
    conn = get_db_connection()
    try:
        conn.execute("DELETE FROM cart WHERE id = ? AND username = ?", (cart_id, username))
        conn.commit()
    finally:
        conn.close()


def update_cart_quantity(username: str, cart_id: int, quantity: int):
    """Cập nhật số lượng sản phẩm trong giỏ hàng. Nếu số lượng <= 0, xóa mục khỏi giỏ."""
    if quantity <= 0:
        remove_from_cart(username, cart_id)
        return
    conn = get_db_connection()
    try:
        conn.execute(
            "UPDATE cart SET quantity = ? WHERE id = ? AND username = ?",
            (int(quantity), int(cart_id), username),
        )
        conn.commit()
    finally:
        conn.close()


def delete_cart_item_by_id(cart_id: int):
    """Xóa mục khỏi giỏ hàng theo ID mục (hỗ trợ REST API & Admin)."""
    conn = get_db_connection()
    try:
        conn.execute("DELETE FROM cart WHERE id = ?", (int(cart_id),))
        conn.commit()
    finally:
        conn.close()


def clear_user_cart(username: str):
    """Xóa toàn bộ sản phẩm trong giỏ hàng của người dùng."""
    if not username:
        return
    conn = get_db_connection()
    try:
        conn.execute("DELETE FROM cart WHERE username = ?", (username,))
        conn.commit()
    finally:
        conn.close()


def get_cart_item_by_id(cart_id: int):
    """Lấy thông tin chi tiết một mục trong giỏ hàng."""
    conn = get_db_connection()
    try:
        row = conn.execute("""
            SELECT c.id, c.username, c.article_id, c.quantity, c.added_at,
                   p.prod_name, p.price, p.product_group_name, p.image_url
            FROM cart c
            JOIN products p ON c.article_id = p.article_id
            WHERE c.id = ?
        """, (int(cart_id),)).fetchone()
        if not row:
            return None
        cols = ["cart_id", "username", "article_id", "quantity", "added_at",
                "prod_name", "price", "product_group_name", "image_url"]
        return dict(zip(cols, row))
    finally:
        conn.close()


def get_all_carts():
    """Lấy danh sách toàn bộ các mục giỏ hàng trong hệ thống (dành cho Admin WinForms)."""
    conn = get_db_connection()
    try:
        rows = conn.execute("""
            SELECT c.id, c.username, c.article_id, c.quantity, c.added_at,
                   p.prod_name, p.price, p.product_group_name, p.image_url
            FROM cart c
            LEFT JOIN products p ON c.article_id = p.article_id
            ORDER BY c.added_at DESC
        """, ()).fetchall()
        cols = ["cart_id", "username", "article_id", "quantity", "added_at",
                "prod_name", "price", "product_group_name", "image_url"]
        return [dict(zip(cols, r)) for r in rows]
    finally:
        conn.close()


def get_user_chatbot_context(username: str) -> dict:
    """Thu thập thông tin bối cảnh của người dùng đăng nhập để cấp cho Chatbot AI tư vấn thời trang."""
    if not username:
        return {
            "username": "Khách hàng",
            "full_name": "Quý khách",
            "occupation": "Khách hàng",
            "usage_need": "Thời trang hàng ngày",
            "usage_frequency": "Thường xuyên",
            "cart_items": [],
            "cart_count": 0
        }

    conn = get_db_connection()
    full_name = username
    occupation = "Học sinh / Sinh viên"
    usage_need = "Dạo phố / Đời thường"
    usage_freq = "Hàng ngày"
    cart_items = []

    try:
        u_row = conn.execute("SELECT full_name, occupation, needs, usage_freq FROM users WHERE username = ?", (username,)).fetchone()
        if u_row:
            full_name = u_row[0] or username
            occupation = u_row[1] or occupation
            usage_need = u_row[2] or usage_need
            usage_freq = u_row[3] or usage_freq

        p_row = conn.execute("SELECT occupation, usage_need, usage_frequency FROM user_profiles WHERE username = ?", (username,)).fetchone()
        if p_row:
            occupation = p_row[0] or occupation
            usage_need = p_row[1] or usage_need
            usage_freq = p_row[2] or usage_freq

        c_rows = conn.execute("""
            SELECT c.article_id, c.quantity, p.prod_name, p.product_group_name, p.price
            FROM cart c
            JOIN products p ON c.article_id = p.article_id
            WHERE c.username = ?
            LIMIT 5
        """, (username,)).fetchall()
        for cr in c_rows:
            cart_items.append({
                "article_id": cr[0],
                "quantity": cr[1],
                "prod_name": cr[2],
                "product_group_name": cr[3],
                "price": cr[4]
            })
    except Exception:
        pass
    finally:
        conn.close()

    return {
        "username": username,
        "full_name": full_name,
        "occupation": occupation,
        "usage_need": usage_need,
        "usage_frequency": usage_freq,
        "cart_items": cart_items,
        "cart_count": len(cart_items)
    }



def get_user_profile(username: str):
    """Lấy thông tin hồ sơ người dùng từ bảng user_profiles (hoặc fallback từ users)."""
    if not username:
        return None
    conn = get_db_connection()
    try:
        row = conn.execute("""
            SELECT occupation, usage_need, usage_frequency, updated_at, style_preferences
            FROM user_profiles WHERE username = ?
        """, (username,)).fetchone()
        if row:
            try:
                preferences = normalize_style_preferences(json.loads(row[4] or "{}"))
            except (ValueError, TypeError):
                preferences = normalize_style_preferences()
            return {
                "username": username,
                "occupation": row[0],
                "usage_need": row[1],
                "usage_frequency": row[2],
                "updated_at": row[3],
                "style_preferences": preferences,
            }
        # Fallback từ bảng users nếu chưa có trong user_profiles
        user_row = conn.execute("""
            SELECT occupation, needs, usage_freq FROM users WHERE username = ?
        """, (username,)).fetchone()
        if user_row:
            return {
                "username": username,
                "occupation": user_row[0] or "",
                "usage_need": user_row[1] or "",
                "usage_frequency": user_row[2] or "",
                "updated_at": None,
                "style_preferences": normalize_style_preferences(),
            }
        return None
    finally:
        conn.close()


def save_user_profile(
    username: str, occupation: str, usage_need: str, usage_frequency: str,
    style_preferences: dict = None,
):
    """Save both profile tables atomically; legacy callers preserve new details.

    Pass an empty dictionary to clear the optional preferences. Invalid details
    raise ValueError before any write; an unknown username returns False.
    """
    if not username:
        return False
    preferences_json = None
    if style_preferences is not None:
        preferences_json = json.dumps(
            normalize_style_preferences(style_preferences), ensure_ascii=False,
            allow_nan=False,
        )
    conn = get_db_connection()
    try:
        with conn:
            updated = conn.execute("""
                UPDATE users SET occupation = ?, needs = ?, usage_freq = ? WHERE username = ?
            """, (occupation, usage_need, usage_frequency, username))
            if updated.rowcount == 0:
                return False

            conn.execute("""
                INSERT INTO user_profiles (
                    username, occupation, usage_need, usage_frequency,
                    style_preferences, updated_at
                )
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(username) DO UPDATE SET
                    occupation = excluded.occupation,
                    usage_need = excluded.usage_need,
                    usage_frequency = excluded.usage_frequency,
                    style_preferences = CASE WHEN ? IS NULL
                        THEN user_profiles.style_preferences
                        ELSE excluded.style_preferences END,
                    updated_at = CURRENT_TIMESTAMP
            """, (
                username, occupation, usage_need, usage_frequency,
                preferences_json or "{}", preferences_json,
            ))
        return True
    finally:
        conn.close()


# ==============================================================================
# QUẢN LÝ ĐƠN HÀNG VÀ ĐỒNG BỘ WEB - WINFORMS (SECTIONS 4 & 5)
# ==============================================================================

VALID_ORDER_STATUSES = [
    "Chờ xác nhận",
    "Đang xử lý",
    "Đang giao",
    "Hoàn thành",
    "Đã hủy",
]

ALLOWED_STATUS_TRANSITIONS = {
    "Chờ xác nhận": ["Đang xử lý", "Đang giao", "Đã hủy"],
    "Đang xử lý": ["Đang giao", "Đã hủy"],
    "Đang giao": ["Hoàn thành", "Đã hủy"],
    "Hoàn thành": [],
    "Đã hủy": [],
}


def db_price_to_vnd(raw_price) -> int:
    """Chuyển đổi giá số thực H&M trong CSDL sang VNĐ nguyên bản chính xác."""
    try:
        p = float(raw_price or 0.03)
    except (TypeError, ValueError):
        p = 0.03
    if p <= 0:
        p = 0.03
    if p < 5.0:
        return max(99000, int(round(p * 25000000, -3)))
    if p < 2000:
        return int(round(p * 25000, -3))
    return int(round(p, -3))


def create_order_from_cart(
    username: str,
    recipient_name: str,
    recipient_phone: str,
    shipping_address: str,
    payment_method: str = "Thanh toán khi nhận hàng (COD)",
    discount_voucher: int = 0,
    discount_vnd: int = 0,
    shipping_fee_vnd: int = 0,
) -> tuple[bool, str, Optional[dict]]:
    """
    Tạo đơn hàng mới từ giỏ hàng thực tế của người dùng:
    - Kiểm tra giỏ hàng có sản phẩm.
    - Đọc đơn giá và thông tin tin cậy từ bảng products.
    - Lưu snapshot đơn giá tại thời điểm đặt (không đổi khi catalog thay đổi giá).
    - Tạo đơn và ghi nhận từng order_item trong transaction.
    - Xóa các dòng đã đặt khỏi giỏ hàng.
    - Ghi nhận tương tác purchase để hệ thống gợi ý học được.
    """
    import random
    import time

    if not username:
        return False, "Vui lòng đăng nhập để đặt hàng.", None

    if not str(recipient_name or "").strip():
        return False, "Vui lòng nhập tên người nhận.", None
    if not str(recipient_phone or "").strip():
        return False, "Vui lòng nhập số điện thoại người nhận.", None
    if not str(shipping_address or "").strip():
        return False, "Vui lòng nhập địa chỉ giao hàng.", None

    conn = get_db_connection()
    try:
        with conn:
            # 1. Đọc các sản phẩm trong giỏ hàng kèm giá tin cậy từ products
            cart_rows = conn.execute("""
                SELECT c.id, c.article_id, c.quantity,
                       p.prod_name, p.price, p.product_group_name, p.colour_group_name, p.image_url
                FROM cart c
                JOIN products p ON c.article_id = p.article_id
                WHERE c.username = ?
            """, (username,)).fetchall()

            if not cart_rows:
                return False, "Giỏ hàng của bạn đang trống, không thể tạo đơn hàng.", None

            subtotal_vnd = 0
            items_to_create = []

            for cr in cart_rows:
                c_id, art_id, qty, p_name, raw_price, p_grp, p_col, img_url = cr
                qty = max(1, int(qty or 1))
                unit_vnd = db_price_to_vnd(raw_price)
                line_vnd = unit_vnd * qty
                subtotal_vnd += line_vnd

                items_to_create.append({
                    "cart_id": c_id,
                    "article_id": art_id,
                    "prod_name": p_name or f"H&M #{art_id}",
                    "shop_name": "Routine Official Store",
                    "color": p_col or "Mặc định",
                    "size": "M",
                    "unit_price_vnd": unit_vnd,
                    "quantity": qty,
                    "line_total_vnd": line_vnd,
                    "image_url": img_url or f"/api/products/{art_id}/image",
                })

            shipping_fee = max(0, int(shipping_fee_vnd or 0))
            voucher_val = max(0, int(discount_voucher or discount_vnd or 0))
            final_total = max(0, subtotal_vnd + shipping_fee - voucher_val)

            # 2. Sinh mã đơn hàng duy nhất: ORD-YYYYMMDD-RAND
            import datetime
            now_dt = datetime.datetime.now()
            order_code = f"ORD-{now_dt.strftime('%Y%m%d%H%M')}-{random.randint(100, 999)}"

            # 3. Tạo bản ghi đơn hàng (orders)
            cur = conn.execute("""
                INSERT INTO orders (
                    order_code, username, status, subtotal_vnd,
                    shipping_fee_vnd, discount_vnd, total_vnd,
                    recipient_name, recipient_phone, shipping_address,
                    payment_method, created_at, updated_at
                )
                VALUES (?, ?, 'Chờ xác nhận', ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """, (
                order_code,
                username,
                subtotal_vnd,
                shipping_fee,
                voucher_val,
                final_total,
                str(recipient_name).strip(),
                str(recipient_phone).strip(),
                str(shipping_address).strip(),
                str(payment_method).strip(),
            ))
            order_db_id = cur.lastrowid

            # 4. Lưu từng snapshot sản phẩm (order_items)
            for it in items_to_create:
                conn.execute("""
                    INSERT INTO order_items (
                        order_id, article_id, prod_name, shop_name,
                        color, size, unit_price_vnd, quantity,
                        line_total_vnd, image_url
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    order_db_id,
                    it["article_id"],
                    it["prod_name"],
                    it["shop_name"],
                    it["color"],
                    it["size"],
                    it["unit_price_vnd"],
                    it["quantity"],
                    it["line_total_vnd"],
                    it["image_url"],
                ))

                # Ghi nhận tương tác mua hàng để học
                conn.execute("""
                    INSERT INTO interactions (username, article_id, interaction_type, rating_score, created_at)
                    VALUES (?, ?, 'purchase', 5.0, CURRENT_TIMESTAMP)
                """, (username, it["article_id"]))

            # 5. Xóa các mặt hàng đã đặt khỏi giỏ
            conn.execute("DELETE FROM cart WHERE username = ?", (username,))

        order_data = {
            "order_id": order_db_id,
            "order_code": order_code,
            "username": username,
            "status": "Chờ xác nhận",
            "subtotal_vnd": subtotal_vnd,
            "shipping_fee_vnd": shipping_fee,
            "discount_vnd": voucher_val,
            "total_vnd": final_total,
            "recipient_name": recipient_name,
            "recipient_phone": recipient_phone,
            "shipping_address": shipping_address,
            "payment_method": payment_method,
            "items_count": len(items_to_create),
            "items": items_to_create,
        }
        return True, f"Đặt hàng thành công! Mã đơn: #{order_code}", order_data

    except Exception as e:
        return False, f"Lỗi tạo đơn hàng: {str(e)}", None
    finally:
        conn.close()


def get_user_orders(username: str) -> list[dict]:
    """Lấy danh sách các đơn hàng của một người dùng kèm danh sách sản phẩm snapshot (Tối ưu 2 truy vấn, loại bỏ N+1)."""
    if not username:
        return []
    conn = get_db_connection()
    try:
        order_rows = conn.execute("""
            SELECT id, order_code, username, status, subtotal_vnd,
                   shipping_fee_vnd, discount_vnd, total_vnd,
                   recipient_name, recipient_phone, shipping_address,
                   payment_method, created_at, updated_at
            FROM orders
            WHERE username = ?
            ORDER BY id DESC
        """, (username,)).fetchall()

        if not order_rows:
            return []

        order_ids = [int(o[0]) for o in order_rows]
        placeholders = ",".join("?" for _ in order_ids)
        all_item_rows = conn.execute(
            f"""
            SELECT id, order_id, article_id, prod_name, shop_name, color, size,
                   unit_price_vnd, quantity, line_total_vnd, image_url
            FROM order_items
            WHERE order_id IN ({placeholders})
            ORDER BY id ASC
            """,
            order_ids,
        ).fetchall()

        items_by_order: dict[int, list[dict]] = {oid: [] for oid in order_ids}
        for it in all_item_rows:
            oid = int(it[1])
            if oid in items_by_order:
                items_by_order[oid].append({
                    "id": it[0],
                    "article_id": it[2],
                    "prod_name": it[3],
                    "shop_name": it[4],
                    "color": it[5],
                    "size": it[6],
                    "unit_price_vnd": it[7],
                    "quantity": it[8],
                    "line_total_vnd": it[9],
                    "image_url": it[10],
                })

        orders_list = []
        for o in order_rows:
            o_id = int(o[0])
            orders_list.append({
                "id": o_id,
                "order_code": o[1],
                "username": o[2],
                "status": o[3],
                "subtotal_vnd": o[4],
                "shipping_fee_vnd": o[5],
                "discount_vnd": o[6],
                "total_vnd": o[7],
                "recipient_name": o[8],
                "recipient_phone": o[9],
                "shipping_address": o[10],
                "payment_method": o[11],
                "created_at": str(o[12]),
                "updated_at": str(o[13]),
                "items": items_by_order.get(o_id, []),
            })
        return orders_list
    finally:
        conn.close()


def cancel_user_order(username: str, order_code: str) -> tuple[bool, str]:
    """Khách hàng hủy đơn hàng (chỉ cho phép khi đơn ở trạng thái 'Chờ xác nhận')."""
    if not username or not order_code:
        return False, "Thiếu thông tin người dùng hoặc mã đơn."
    conn = get_db_connection()
    try:
        row = conn.execute("""
            SELECT id, status FROM orders WHERE order_code = ? AND username = ?
        """, (order_code, username)).fetchone()
        if not row:
            return False, "Không tìm thấy đơn hàng hoặc bạn không có quyền hủy đơn này."

        order_id, curr_status = row
        if curr_status != "Chờ xác nhận":
            return False, f"Không thể hủy đơn hàng vì đơn đã ở trạng thái '{curr_status}'."

        conn.execute("""
            UPDATE orders
            SET status = 'Đã hủy', updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (order_id,))
        conn.commit()
        return True, f"Đã hủy đơn hàng #{order_code} thành công."
    finally:
        conn.close()


def get_all_admin_orders(status_filter: Optional[str] = None) -> list[dict]:
    """Lấy toàn bộ danh sách đơn hàng phục vụ WinForms Admin Desktop."""
    conn = get_db_connection()
    try:
        query = """
            SELECT o.id, o.order_code, o.username, o.status, o.subtotal_vnd,
                   o.total_vnd, o.recipient_name, o.recipient_phone, o.shipping_address,
                   o.created_at, o.updated_at,
                   COUNT(oi.id) as item_count,
                   SUM(oi.quantity) as total_qty
            FROM orders o
            LEFT JOIN order_items oi ON o.id = oi.order_id
        """
        params = []
        if status_filter and status_filter.strip() and status_filter.strip() != "Tất cả":
            query += " WHERE o.status = ?"
            params.append(status_filter.strip())

        query += " GROUP BY o.id ORDER BY o.id DESC"
        rows = conn.execute(query, params).fetchall()

        result = []
        for r in rows:
            result.append({
                "id": r[0],
                "order_code": r[1],
                "username": r[2],
                "status": r[3],
                "subtotal_vnd": r[4],
                "total_vnd": r[5],
                "recipient_name": r[6] or "",
                "recipient_phone": r[7] or "",
                "shipping_address": r[8] or "",
                "created_at": str(r[9]),
                "updated_at": str(r[10]),
                "items_count": r[11] or 0,
                "total_quantity": r[12] or 0,
                "formatted_total": f"{r[5]:,} ₫".replace(",", "."),
            })
        return result
    finally:
        conn.close()


def get_admin_order_details(order_id_or_code: Any) -> Optional[dict]:
    """Lấy thông tin chi tiết đầy đủ của một đơn hàng kèm snapshot items."""
    conn = get_db_connection()
    try:
        is_id = isinstance(order_id_or_code, int) or (str(order_id_or_code).isdigit() and not str(order_id_or_code).startswith("ORD"))
        if is_id:
            row = conn.execute("""
                SELECT id, order_code, username, status, subtotal_vnd,
                       shipping_fee_vnd, discount_vnd, total_vnd,
                       recipient_name, recipient_phone, shipping_address,
                       payment_method, created_at, updated_at
                FROM orders WHERE id = ?
            """, (int(order_id_or_code),)).fetchone()
        else:
            row = conn.execute("""
                SELECT id, order_code, username, status, subtotal_vnd,
                       shipping_fee_vnd, discount_vnd, total_vnd,
                       recipient_name, recipient_phone, shipping_address,
                       payment_method, created_at, updated_at
                FROM orders WHERE order_code = ?
            """, (str(order_id_or_code),)).fetchone()

        if not row:
            return None

        order_id = row[0]
        items_rows = conn.execute("""
            SELECT id, article_id, prod_name, shop_name, color, size,
                   unit_price_vnd, quantity, line_total_vnd, image_url
            FROM order_items WHERE order_id = ? ORDER BY id ASC
        """, (order_id,)).fetchall()

        items = []
        for it in items_rows:
            items.append({
                "id": it[0],
                "article_id": it[1],
                "prod_name": it[2],
                "shop_name": it[3],
                "color": it[4],
                "size": it[5],
                "unit_price_vnd": it[6],
                "quantity": it[7],
                "line_total_vnd": it[8],
                "image_url": it[9],
                "formatted_unit_price": f"{it[6]:,} ₫".replace(",", "."),
                "formatted_line_total": f"{it[8]:,} ₫".replace(",", "."),
            })

        return {
            "id": row[0],
            "order_code": row[1],
            "username": row[2],
            "status": row[3],
            "subtotal_vnd": row[4],
            "shipping_fee_vnd": row[5],
            "discount_vnd": row[6],
            "total_vnd": row[7],
            "formatted_total": f"{row[7]:,} ₫".replace(",", "."),
            "recipient_name": row[8],
            "recipient_phone": row[9],
            "shipping_address": row[10],
            "payment_method": row[11],
            "created_at": str(row[12]),
            "updated_at": str(row[13]),
            "items": items,
        }
    finally:
        conn.close()


def update_order_status_admin(order_id_or_code: Any, new_status: str) -> tuple[bool, str]:
    """
    Cập nhật trạng thái đơn hàng theo các chuyển đổi hợp lệ:
    - Chờ xác nhận -> Đang xử lý / Đang giao / Đã hủy
    - Đang xử lý -> Đang giao / Đã hủy
    - Đang giao -> Hoàn thành / Đã hủy
    - Hoàn thành hoặc Đã hủy -> Không thể đổi lại
    """
    if new_status not in VALID_ORDER_STATUSES:
        return False, f"Trạng thái '{new_status}' không hợp lệ. Các trạng thái hợp lệ: {', '.join(VALID_ORDER_STATUSES)}."

    conn = get_db_connection()
    try:
        is_id = isinstance(order_id_or_code, int) or (str(order_id_or_code).isdigit() and not str(order_id_or_code).startswith("ORD"))
        if is_id:
            row = conn.execute("SELECT id, order_code, status FROM orders WHERE id = ?", (int(order_id_or_code),)).fetchone()
        else:
            row = conn.execute("SELECT id, order_code, status FROM orders WHERE order_code = ?", (str(order_id_or_code),)).fetchone()

        if not row:
            return False, f"Không tìm thấy đơn hàng {order_id_or_code}."

        order_id, order_code, curr_status = row
        if curr_status == new_status:
            return True, f"Đơn hàng #{order_code} đã ở trạng thái '{new_status}'."

        allowed = ALLOWED_STATUS_TRANSITIONS.get(curr_status, [])
        if new_status not in allowed:
            return False, f"Chuyển trạng thái không hợp lệ: Không thể chuyển từ '{curr_status}' sang '{new_status}'."

        conn.execute("""
            UPDATE orders
            SET status = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (new_status, order_id))
        conn.commit()
        return True, f"Cập nhật trạng thái đơn hàng #{order_code} thành công sang '{new_status}'!"
    finally:
        conn.close()


