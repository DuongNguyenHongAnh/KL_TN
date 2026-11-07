"""
Sinh 4 sơ đồ kiến trúc và quy trình nghiệp vụ cho Task 7 & Task 8
Độ phân giải cao (300 DPI), chuẩn phong cách học thuật, nhãn tiếng Việt rõ ràng.

1. figures/architecture/system_architecture_task7_8.png
2. figures/computer_vision/image_feature_pipeline.png
3. figures/authentication/email_otp_flow.png
4. figures/architecture/personalized_recommendation_flow.png
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
ARCH_DIR = os.path.join(FIG_DIR, "architecture")
CV_DIR = os.path.join(FIG_DIR, "computer_vision")
AUTH_DIR = os.path.join(FIG_DIR, "authentication")

for d in [ARCH_DIR, CV_DIR, AUTH_DIR]:
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
        mid_y = (start[1] + end[1]) / 2 + 0.04
        ax.text(mid_x, mid_y, label, ha="center", va="bottom", fontsize=8.5, color=color, style="italic")

# SƠ ĐỒ 1: KIẾN TRÚC TỔNG THỂ HỆ THỐNG ĐA PHƯƠNG THỨC (TASK 7 & 8)
def gen_overall_architecture():
    fig, ax = plt.subplots(figsize=(11.5, 6.5), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    # Layer 1: Client
    draw_box(ax, (0.5, 4.2), 2.2, 1.6, "Web Frontend\n(Streamlit)\n- Grid View (Ảnh thật)\n- Hybrid Recommendation\n- Chatbot Gemini AI", "#e0f2fe", "#0284c7", 9.5)
    draw_box(ax, (0.5, 1.2), 2.2, 1.6, "WinForms Desktop\n(C# Admin Client)\n- Quản trị viên Dashboard\n- Giám sát KPI & Slider\n- Điều khiển tham số", "#fce7f3", "#db2777", 9.5)

    # Layer 2: API Gateway
    draw_box(ax, (3.8, 1.0), 3.0, 4.8, "FastAPI Backend Server\n\n- Auth: Bcrypt & Email OTP\n- /api/recommendations (Hybrid)\n- /api/products/similar-image\n- /api/users/profile (CRUD)\n- /api/model/status\n- Pydantic Schemas & CORS", "#ede9fe", "#7c3aed", 10)

    # Layer 3: Recommendation Engine Services
    draw_box(ax, (8.0, 4.2), 3.5, 1.6, "Hybrid Recommendation Engine\n\n- Implicit ALS (x_u^T · y_i)\n- Profile Recommender (Match)\n- ResNet50 Visual Similarity", "#dcfce7", "#16a34a", 9.5)
    draw_box(ax, (8.0, 1.2), 3.5, 1.6, "Data & Storage Layer\n\n- SQLite (users, profiles, products)\n- image_features_resnet50.npz\n- 34,472 H&M Product Images", "#ffedd5", "#ea580c", 9.5)

    # Connectors
    draw_arrow(ax, (2.7, 5.0), (3.8, 5.0), "HTTP / JSON")
    draw_arrow(ax, (2.7, 2.0), (3.8, 2.0), "REST API")

    draw_arrow(ax, (6.8, 5.0), (8.0, 5.0), "Inference")
    draw_arrow(ax, (6.8, 2.0), (8.0, 2.0), "SQL / Vectors")

    draw_arrow(ax, (9.75, 4.2), (9.75, 2.8), "Nạp Model & Cache")

    ax.text(6.0, 6.1, "Sơ đồ Kiến trúc Tổng thể Hệ thống Gợi ý Đa phương thức (Multimodal Architecture)", fontsize=12.5, weight="bold", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(ARCH_DIR, "system_architecture_task7_8.png"), bbox_inches="tight")
    plt.close()

# SƠ ĐỒ 2: QUY TRÌNH TRÍCH XUẤT ĐẶC TRƯNG ẢNH RESNET50 TRANSFER LEARNING
def gen_image_pipeline():
    fig, ax = plt.subplots(figsize=(11.5, 4.8), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.8)
    ax.axis("off")

    draw_box(ax, (0.4, 1.6), 1.8, 1.6, "1. Ảnh Sản phẩm\nH&M Dataset\n(34,472 ảnh .jpg)", "#e0e7ff", "#4338ca", 9.5)
    draw_arrow(ax, (2.2, 2.4), (2.8, 2.4))

    draw_box(ax, (2.8, 1.6), 2.2, 1.6, "2. Tiền xử lý Ảnh\n- Resize 224x224\n- RGB Conversion\n- ImageNet Norm", "#dbeafe", "#2563eb", 9.5)
    draw_arrow(ax, (5.0, 2.4), (5.6, 2.4))

    draw_box(ax, (5.6, 1.6), 2.4, 1.6, "3. Pretrained ResNet50\n- Conv Backbone\n- Global Avg Pooling\n- Bỏ lớp phân loại FC", "#fef3c7", "#d97706", 9.5)
    draw_arrow(ax, (8.0, 2.4), (8.6, 2.4))

    draw_box(ax, (8.6, 2.5), 3.0, 1.6, "4. Vector Đặc trưng 2048-D\n- Chuẩn hóa L2 Norm\n- Lưu image_features.npz", "#dcfce7", "#15803d", 9.5)
    draw_box(ax, (8.6, 0.5), 3.0, 1.5, "5. Cosine Similarity\nTìm Top-K đồ tương tự thị giác\nsim(A, B) = v_A · v_B", "#f1f5f9", "#475569", 9.5)

    draw_arrow(ax, (10.1, 2.5), (10.1, 2.0))

    ax.text(6.0, 4.2, "Quy trình Trích xuất Đặc trưng Ảnh bằng ResNet50 Transfer Learning", fontsize=12.5, weight="bold", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(CV_DIR, "image_feature_pipeline.png"), bbox_inches="tight")
    plt.close()

# SƠ ĐỒ 3: QUY TRÌNH BẢO MẬT VÀ XÁC MINH EMAIL OTP
def gen_auth_otp_flow():
    fig, ax = plt.subplots(figsize=(11.5, 4.8), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.8)
    ax.axis("off")

    draw_box(ax, (0.4, 1.6), 1.8, 1.6, "1. Đăng ký\n- Username, Email\n- Mật khẩu mạnh\n- Hồ sơ phong cách", "#e0f2fe", "#0284c7", 9.5)
    draw_arrow(ax, (2.2, 2.4), (2.8, 2.4))

    draw_box(ax, (2.8, 1.6), 2.2, 1.6, "2. Kiểm tra & Băm\n- Regex Password\n- Băm Bcrypt mật khẩu\n- Trạng thái chưa verify", "#ede9fe", "#7c3aed", 9.5)
    draw_arrow(ax, (5.0, 2.4), (5.6, 2.4))

    draw_box(ax, (5.6, 1.6), 2.2, 1.6, "3. Cấp mã OTP\n- Sinh 6 số ngẫu nhiên\n- Hạn dùng 5 phút\n- Gửi qua EmailService", "#fef3c7", "#d97706", 9.5)
    draw_arrow(ax, (7.8, 2.4), (8.4, 2.4))

    draw_box(ax, (8.4, 2.5), 3.0, 1.6, "4. Xác minh OTP\n- Nhập mã 6 chữ số\n- Giới hạn tối đa 5 lần\n- Kích hoạt is_verified = 1", "#dcfce7", "#16a34a", 9.5)
    draw_box(ax, (8.4, 0.5), 3.0, 1.5, "5. Đăng nhập Thành công\nCấp phát Bearer Token\nTruy cập hệ thống đầy đủ", "#f8fafc", "#334155", 9.5)

    draw_arrow(ax, (9.9, 2.5), (9.9, 2.0))

    ax.text(6.0, 4.2, "Quy trình Xác thực Tài khoản và Bảo mật Email OTP (Chống Spam Tài khoản Ảo)", fontsize=12.5, weight="bold", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(AUTH_DIR, "email_otp_flow.png"), bbox_inches="tight")
    plt.close()

# SƠ ĐỒ 4: LUỒNG GỢI Ý CÁ NHÂN HÓA HYBRID RECOMMENDATION
def gen_personalized_recommendation_flow():
    fig, ax = plt.subplots(figsize=(11.5, 5.5), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    draw_box(ax, (0.5, 2.0), 2.2, 2.0, "Thông tin Người dùng\n\n- user_id & Lịch sử mua\n- Nghề nghiệp (Occupation)\n- Nhu cầu (Usage Need)\n- Tần suất (Frequency)", "#e0f2fe", "#0284c7", 9.5)

    draw_arrow(ax, (2.7, 3.4), (3.8, 4.0), "Lịch sử mua")
    draw_arrow(ax, (2.7, 2.8), (3.8, 2.6), "Hồ sơ cá nhân")
    draw_arrow(ax, (2.7, 2.2), (3.8, 1.2), "Đồ xem gần đây")

    draw_box(ax, (3.8, 3.4), 3.0, 1.5, "1. Lọc cộng tác CF\nImplicit ALS (Factors=20)\nĐiểm số S_cf", "#ede9fe", "#7c3aed", 9.5)
    draw_box(ax, (3.8, 1.8), 3.0, 1.4, "2. Bộ lọc Hồ sơ Profile\nÁnh xạ nhóm hàng/từ khóa\nĐiểm số S_profile", "#fef3c7", "#d97706", 9.5)
    draw_box(ax, (3.8, 0.4), 3.0, 1.2, "3. ResNet50 Visual Sim\nTương đồng thị giác ảnh\nĐiểm số S_visual", "#dbeafe", "#1d4ed8", 9.5)

    draw_arrow(ax, (6.8, 4.15), (7.8, 3.0))
    draw_arrow(ax, (6.8, 2.5), (7.8, 2.7))
    draw_arrow(ax, (6.8, 1.0), (7.8, 2.4))

    draw_box(ax, (7.8, 1.8), 3.7, 1.8, "Bộ chấm điểm Hybrid & Ranking\n\nw_cf·S_cf + w_prof·S_prof + w_vis·S_vis\n- Loại bỏ sản phẩm đã mua\n- Sắp xếp thứ tự ưu tiên Top-N", "#dcfce7", "#16a34a", 10)

    draw_arrow(ax, (9.65, 1.8), (9.65, 1.2))
    draw_box(ax, (7.8, 0.3), 3.7, 0.9, "Hiển thị Grid View (Web & WinForms)\nẢnh sản phẩm + Tên + Danh mục + Điểm Rank", "#f1f5f9", "#475569", 9.5)

    ax.text(6.0, 5.1, "Quy trình Gợi ý Cá nhân hóa Đa phương thức (Personalized Hybrid Flow)", fontsize=12.5, weight="bold", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(ARCH_DIR, "personalized_recommendation_flow.png"), bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    print("Bắt đầu sinh 4 sơ đồ kiến trúc và quy trình Task 7 & Task 8...")
    gen_overall_architecture()
    gen_image_pipeline()
    gen_auth_otp_flow()
    gen_personalized_recommendation_flow()
    print("-> Đã tạo thành công 4 sơ đồ độ phân giải cao 300 DPI!")
