import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import importlib

# Luôn luôn đưa thư mục gốc của dự án vào đầu sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
    importlib.invalidate_caches()

import streamlit as st

# set_page_config phải chạy trước các lệnh Streamlit khác
st.set_page_config(
    page_title="Thời trang dành cho bạn",
    page_icon=":material/checkroom:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from auth import clear_user_session, init_session, is_authenticated
from ui.chat_panel import inside_placeholder
from ui.login import login_page
from ui.styles import load_styles


def main():
    load_styles()
    init_session()

    # 1. Kiểm tra trạng thái phiên đăng nhập trước khi render giao diện chính
    if not is_authenticated():
        # Xóa sạch trạng thái nếu phiên không hợp lệ hoặc hết hạn
        if st.session_state.get("logged_in", False):
            clear_user_session()
        # 2. Render màn hình xác thực và dừng nhánh xử lý ngay lập tức
        login_page()
        st.stop()

    # 3. Chỉ render giao diện ứng dụng chính khi phiên đăng nhập hợp lệ
    inside_placeholder()


if __name__ == "__main__":
    main()
