"""
Sinh các sơ đồ kiến trúc và luồng xử lý (Architecture & Workflow Diagrams)
Đảm bảo độ phân giải cao (300 DPI), phong cách chuyên nghiệp, nhãn tiếng Việt rõ ràng.
"""

import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "figures")
MF_FIG_DIR = os.path.join(FIG_DIR, "matrix_factorization")
ARCH_FIG_DIR = os.path.join(FIG_DIR, "architecture")
TOPN_FIG_DIR = os.path.join(FIG_DIR, "top_n")
AUTH_FIG_DIR = os.path.join(FIG_DIR, "authentication")

for d in [MF_FIG_DIR, ARCH_FIG_DIR, TOPN_FIG_DIR, AUTH_FIG_DIR]:
    os.makedirs(d, exist_ok=True)

def draw_box(ax, xy, width, height, text, bg_color="#e0f2fe", edge_color="#0284c7", fontsize=10, text_color="#0f172a", bold=True):
    rect = patches.FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.1,rounding_size=0.15",
        facecolor=bg_color,
        edgecolor=edge_color,
        linewidth=1.8
    )
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text,
            ha="center", va="center", fontsize=fontsize, color=text_color,
            weight=weight, wrap=True)
    return rect

def draw_arrow(ax, start, end, label=None, color="#475569"):
    ax.annotate(
        "", xy=end, xytext=start,
        arrowprops=dict(arrowstyle="->", color=color, lw=2.0, mutation_scale=15)
    )
    if label:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2 + 0.03
        ax.text(mid_x, mid_y, label, ha="center", va="bottom", fontsize=8.5, color=color, style="italic")

# 1. SƠ ĐỒ KIẾN TRÚC MATRIX FACTORIZATION (R ≈ P * Q^T)
def gen_mf_architecture():
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    draw_box(ax, (0.5, 1.2), 2.2, 2.6, "Ma trận Tương tác R\n(User - Item)\nSparse Matrix\n[m x n]", "#dbeafe", "#1d4ed8", 11)
    ax.text(3.1, 2.5, "≈", fontsize=28, ha="center", va="center", weight="bold", color="#1e293b")
    draw_box(ax, (3.6, 1.2), 1.8, 2.6, "Ma trận Ẩn\nNgười dùng P\n(User Factors)\n[m x k]", "#fef3c7", "#d97706", 11)
    ax.text(5.8, 2.5, "×", fontsize=24, ha="center", va="center", weight="bold", color="#1e293b")
    draw_box(ax, (6.3, 1.6), 2.8, 1.8, "Ma trận Ẩn Sản phẩm Q^T\n(Item Factors Transposed)\n[k x n]", "#dcfce7", "#15803d", 11)

    ax.text(5.0, 4.4, "Kiến trúc Phân rã Ma trận (Matrix Factorization)", fontsize=13, weight="bold", ha="center")
    ax.text(5.0, 0.4, "Dự đoán sở thích: r̂_ui = x_u^T · y_i  |  Không gian đặc trưng ẩn k nhỏ hơn min(m, n)", fontsize=10, ha="center", style="italic", color="#334155")

    plt.tight_layout()
    plt.savefig(os.path.join(MF_FIG_DIR, "mf_architecture.png"), bbox_inches="tight")
    plt.close()

# 2. SƠ ĐỒ QUY TRÌNH HUẤN LUYỆN IMPLICIT ALS WORKFLOW
def gen_als_workflow():
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=300)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.5)
    ax.axis("off")

    draw_box(ax, (0.4, 1.5), 1.8, 1.5, "Tập dữ liệu H&M\nImplicit Feedback\n(Lịch sử mua)", "#e0e7ff", "#4338ca", 9.5)
    draw_arrow(ax, (2.2, 2.25), (2.8, 2.25))

    draw_box(ax, (2.8, 1.5), 2.0, 1.5, "Tính ma trận Tin cậy\nc_ui = 1 + α·r_ui\np_ui = 1 (nếu r_ui>0)", "#dbeafe", "#2563eb", 9.5)
    draw_arrow(ax, (4.8, 2.25), (5.4, 2.25))

    draw_box(ax, (5.4, 2.4), 2.2, 1.4, "Bước 1: Cố định Q\nTối ưu hóa P\n(User Factors)", "#fef3c7", "#d97706", 9.5)
    draw_box(ax, (5.4, 0.6), 2.2, 1.4, "Bước 2: Cố định P\nTối ưu hóa Q\n(Item Factors)", "#dcfce7", "#15803d", 9.5)

    draw_arrow(ax, (6.5, 2.4), (6.5, 2.0), label="Cập nhật")
    draw_arrow(ax, (7.6, 1.3), (8.1, 1.3))
    draw_arrow(ax, (7.6, 3.1), (8.1, 3.1))

    draw_box(ax, (8.1, 1.3), 2.4, 1.9, "Hội tụ (15 vòng lặp)\nXuất P & Q tối ưu\nLưu als_best_model.pkl", "#f1f5f9", "#475569", 10)

    ax.text(5.5, 4.1, "Quy trình Huấn luyện Mô hình Tương tác Ẩn ALS (Alternating Least Squares)", fontsize=12, weight="bold", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(MF_FIG_DIR, "als_workflow.png"), bbox_inches="tight")
    plt.close()

# 3. SƠ ĐỒ KIẾN TRÚC REST API TỔNG THỂ
def gen_api_architecture():
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6)
    ax.axis("off")

    draw_box(ax, (0.5, 3.8), 2.2, 1.4, "Web Frontend\n(Streamlit / React)\nKhách hàng", "#e0f2fe", "#0284c7", 10)
    draw_box(ax, (0.5, 1.2), 2.2, 1.4, "WinForms Client\n(C# / Desktop)\nQuản trị viên Admin", "#fce7f3", "#db2777", 10)

    draw_box(ax, (4.0, 1.0), 3.0, 4.4, "FastAPI Backend Server\n\n- Authentication (JWT / Bcrypt)\n- /api/recommendations/{u}\n- /api/items/{id}/similar\n- /api/model/status\n- Pydantic Validation & CORS", "#ede9fe", "#7c3aed", 10.5)

    draw_box(ax, (8.2, 3.6), 2.3, 1.7, "Preloaded ML Models\n\n- als_best_model.pkl\n- knn_best_model.pkl\n(Fast Vector Inference)", "#dcfce7", "#16a34a", 9.5)
    draw_box(ax, (8.2, 1.0), 2.3, 1.7, "Database (SQLite)\n\n- users & profiles\n- products & catalog\n- interactions / carts", "#ffedd5", "#ea580c", 9.5)

    draw_arrow(ax, (2.7, 4.5), (4.0, 4.5), "JSON / REST")
    draw_arrow(ax, (2.7, 1.9), (4.0, 1.9), "HTTP Request")

    draw_arrow(ax, (7.0, 4.5), (8.2, 4.5), "Cosine / Latent Dot")
    draw_arrow(ax, (7.0, 2.0), (8.2, 2.0), "SQL Query / Auth")

    ax.text(5.5, 5.7, "Kiến trúc Hệ thống Backend REST API và Tích hợp Đa nền tảng", fontsize=12.5, weight="bold", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(ARCH_FIG_DIR, "api_architecture.png"), bbox_inches="tight")
    plt.close()

# 4. SƠ ĐỒ LUỒNG WEB -> API -> MODEL
def gen_web_api_flow():
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.5)
    ax.axis("off")

    draw_box(ax, (0.5, 1.5), 1.8, 1.5, "1. Khách hàng\nTruy cập trang chủ\nhoặc xem chi tiết", "#e0f2fe", "#0284c7", 9.5)
    draw_arrow(ax, (2.3, 2.25), (2.9, 2.25), "GET /api/recs")

    draw_box(ax, (2.9, 1.5), 2.0, 1.5, "2. Backend API\nKiểm tra User ID\nxác thực tham số", "#ede9fe", "#7c3aed", 9.5)
    draw_arrow(ax, (4.9, 2.25), (5.5, 2.25), "Query ALS")

    draw_box(ax, (5.5, 1.5), 2.0, 1.5, "3. ALS Model\nTính x_u^T · Y\nLoại bỏ đã mua", "#dcfce7", "#16a34a", 9.5)
    draw_arrow(ax, (7.5, 2.25), (8.1, 2.25), "Top-N items")

    draw_box(ax, (8.1, 1.5), 1.5, 1.5, "4. Render UI\nHiển thị Grid\nSản phẩm gợi ý", "#fef3c7", "#d97706", 9.5)

    ax.text(5.0, 4.0, "Luồng Tương tác Dữ liệu: Web Frontend -> REST API -> Khuyến nghị ALS", fontsize=12, weight="bold", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(ARCH_FIG_DIR, "web_api_model_flow.png"), bbox_inches="tight")
    plt.close()

# 5. SƠ ĐỒ LUỒNG WINFORMS ADMIN -> API
def gen_winforms_api_flow():
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.5)
    ax.axis("off")

    draw_box(ax, (0.5, 1.5), 1.8, 1.5, "1. Admin WinForms\nĐiều khiển tham số\nK, Mode, Threshold", "#fce7f3", "#db2777", 9.5)
    draw_arrow(ax, (2.3, 2.25), (2.9, 2.25), "HTTP GET/POST")

    draw_box(ax, (2.9, 1.5), 2.0, 1.5, "2. API Controller\n/api/model/status\n/api/recommendations", "#ede9fe", "#7c3aed", 9.5)
    draw_arrow(ax, (4.9, 2.25), (5.5, 2.25), "Kiểm tra KPI")

    draw_box(ax, (5.5, 1.5), 2.0, 1.5, "3. Model & Cache\nTrả về RMSE, MAE,\nThời gian tính toán", "#dbeafe", "#1d4ed8", 9.5)
    draw_arrow(ax, (7.5, 2.25), (8.1, 2.25), "JSON Data")

    draw_box(ax, (8.1, 1.5), 1.5, 1.5, "4. Admin UI\nCập nhật Charts\nBảng KPI thực tế", "#f1f5f9", "#475569", 9.5)

    ax.text(5.0, 4.0, "Luồng Quản trị Giám sát: WinForms Desktop -> REST API -> Trực quan hóa KPI", fontsize=12, weight="bold", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(ARCH_FIG_DIR, "winforms_api_model_flow.png"), bbox_inches="tight")
    plt.close()

# 6. SƠ ĐỒ ĐÁNH GIÁ TOP-N WORKFLOW
def gen_top_n_workflow():
    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=300)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.8)
    ax.axis("off")

    draw_box(ax, (0.4, 1.6), 2.0, 1.6, "1. Dữ liệu Test\n(Customer, Article)\nGround Truth tương tác", "#e0e7ff", "#4338ca", 9.5)
    draw_arrow(ax, (2.4, 2.4), (3.0, 2.4))

    draw_box(ax, (3.0, 1.6), 2.2, 1.6, "2. Bộ lọc Đã Mua\n(Already Seen Items)\nLoại bỏ item tập Train", "#fee2e2", "#dc2626", 9.5)
    draw_arrow(ax, (5.2, 2.4), (5.8, 2.4))

    draw_box(ax, (5.8, 1.6), 2.3, 1.6, "3. Sinh Top-N Ranking\nSắp xếp điểm dự đoán\nChọn Top 5, 10, 20", "#dcfce7", "#16a34a", 9.5)
    draw_arrow(ax, (8.1, 2.4), (8.7, 2.4))

    draw_box(ax, (8.7, 1.6), 2.0, 1.6, "4. Tính 4 Chỉ số\n- Precision@K\n- Recall@K\n- MAP@K\n- NDCG@K", "#fef3c7", "#d97706", 9.5)

    ax.text(5.5, 4.2, "Quy trình Đánh giá Xếp hạng Khuyến nghị Thực tế (Top-N Evaluation Workflow)", fontsize=12, weight="bold", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(TOPN_FIG_DIR, "top_n_workflow.png"), bbox_inches="tight")
    plt.close()

# 7. SƠ ĐỒ LUỒNG XÁC THỰC BẢO MẬT (AUTH FLOW)
def gen_auth_flow():
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    draw_box(ax, (0.5, 2.8), 2.0, 1.4, "Người dùng\nNhập Mật khẩu\n+ Thông tin đăng nhập", "#e0f2fe", "#0284c7", 9.5)
    draw_arrow(ax, (2.5, 3.5), (3.3, 3.5), "POST /api/login")

    draw_box(ax, (3.3, 2.8), 2.2, 1.4, "Kiểm tra SQLite\nLấy Hash Bcrypt\n(users table)", "#ede9fe", "#7c3aed", 9.5)
    draw_arrow(ax, (5.5, 3.5), (6.3, 3.5), "bcrypt.checkpw")

    draw_box(ax, (6.3, 3.0), 1.8, 1.2, "Thành công\nTạo Phiên/Token\nCho phép truy cập", "#dcfce7", "#16a34a", 9.5)
    draw_box(ax, (6.3, 1.0), 1.8, 1.2, "Thất bại\nSai Mật khẩu\nTừ chối 401", "#fee2e2", "#dc2626", 9.5)

    draw_arrow(ax, (5.5, 3.2), (6.3, 1.6), "Khác Hash")
    draw_box(ax, (8.5, 2.0), 1.2, 1.6, "Phân quyền\n- Customer\n- Admin", "#fef3c7", "#d97706", 9.5)
    draw_arrow(ax, (8.1, 3.6), (8.5, 3.2))

    ax.text(5.0, 4.6, "Quy trình Xác thực & Bảo mật Mật khẩu Chuẩn Bcrypt (Authentication Flow)", fontsize=12, weight="bold", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(AUTH_FIG_DIR, "auth_flow.png"), bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    print("Bắt đầu sinh 7 sơ đồ kiến trúc và quy trình...")
    gen_mf_architecture()
    gen_als_workflow()
    gen_api_architecture()
    gen_web_api_flow()
    gen_winforms_api_flow()
    gen_top_n_workflow()
    gen_auth_flow()
    print("-> Đã tạo thành công toàn bộ 7 sơ đồ độ phân giải cao tại các thư mục figures/!")
