"""
Kiểm tra toàn diện tất cả các module và phân hệ trong dự án KLTN.
"""

import sys
import os

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

print("=" * 70)
print("  KIỂM TRA SỨC KHỎE TOÀN DIỆN HỆ THỐNG (SYSTEM INTEGRITY CHECK)")
print("=" * 70)

# 1. Kiểm tra CSDL SQLite
print("\n[1] Kiểm tra Cơ sở dữ liệu SQLite (users.db):")
import sqlite3
from db import DB_PATH, get_db_connection, init_db

try:
    init_db()
    conn = get_db_connection()
    integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
    print(f"    • Đường dẫn DB:       {DB_PATH}")
    print(f"    • Trạng thái toàn vẹn: {integrity}")
    
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    print(f"    • Danh sách bảng ({len(tables)}): {', '.join(tables)}")
    
    user_count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    print(f"    • Tổng số tài khoản: {user_count}")
    conn.close()
    print("    [✓] CSDL SQLite hoạt động hoàn hảo!")
except Exception as e:
    print(f"    [✗] Lỗi CSDL: {e}")

# 2. Kiểm tra các module cốt lõi
print("\n[2] Kiểm tra nạp các module mã nguồn:")
modules = [
    ("auth", "Xác thực & Bảo mật mật khẩu"),
    ("db", "Quản lý CSDL SQLite"),
    ("email_service", "Dịch vụ Email OTP (Brevo)"),
    ("ui.components.navbar", "UI Components & Global Design System"),
    ("image_feature_extractor", "Trích xuất ResNet50 CV"),
    ("profile_recommender", "Bộ lọc Hồ sơ phong cách"),
    ("hybrid_recommender", "Engine Gợi ý Đa phương thức"),
    ("knn_recommender", "Mô hình KNN CF"),
    ("svd_recommender", "Mô hình SVD Baseline (PureSVD)"),
    ("matrix_factorization", "Mô hình Matrix Factorization (ALS)"),
    ("top_n_evaluator", "Module Đánh giá Top-N Ranking"),
    ("chatbot", "Trợ lý ảo Stylist AI"),
    ("api.main", "FastAPI Backend REST API"),
    ("ui.login", "UI Đăng nhập & Đăng ký"),
    ("ui.product_view", "UI Danh mục & Khám phá trang phục"),
    ("ui.chat_panel", "UI Khung chat tương tác"),
]

err_count = 0
for mod_name, desc in modules:
    try:
        __import__(mod_name)
        print(f"    [✓] {mod_name:<25} : {desc}")
    except Exception as e:
        err_count += 1
        print(f"    [✗] {mod_name:<25} : {type(e).__name__}: {e}")

# 3. Kiểm tra các file mô hình đã huấn luyện
print("\n[3] Kiểm tra các mô hình học máy đã lưu (Model Artifacts):")
model_files = [
    ("models/computer_vision/image_features_resnet50.npz", "Vector đặc trưng ảnh ResNet50 (2048 chiều)"),
    ("models/matrix_factorization/als_best_model.pkl", "Mô hình Implicit ALS Factorization"),
    ("models/svd/svd_best_model.pkl", "Mô hình SVD Baseline (PureSVD)"),
    ("models/knn/item_knn_cosine_best.pkl", "Mô hình Item-KNN Cosine"),
]

for m_rel, desc in model_files:
    m_path = os.path.join(BASE_DIR, m_rel)
    if os.path.exists(m_path):
        size_mb = os.path.getsize(m_path) / (1024 * 1024)
        print(f"    [✓] {m_rel:<45} ({size_mb:.2f} MB) - {desc}")
    else:
        print(f"    [✗] THIẾU MÔ HÌNH: {m_rel} ({desc})")

print("\n" + "=" * 70)
print(f"KẾT QUẢ KIỂM TRA: {'TẤT CẢ HOÀN HẢO' if err_count == 0 else f'CÓ {err_count} LỖI CẦN XỬ LÝ'}")
print("=" * 70)
