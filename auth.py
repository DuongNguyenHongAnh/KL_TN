"""Xác thực người dùng, Email OTP và validation đăng ký.

Bản Email-only:
- Không hỗ trợ OTP qua số điện thoại/SMS.
- OTP Email random 6 số.
- OTP chỉ được lưu sau khi EmailService gửi thành công.
- OTP plaintext không được trả về UI.
"""

import random
import re
import sqlite3
import bcrypt
import streamlit as st

from db import DEFAULT_CHAT_GREETING, get_db_connection

OCCUPATION_OPTIONS = (
    "Học sinh / Sinh viên",
    "Nhân viên văn phòng",
    "Kinh doanh / Bán hàng",
    "Giáo viên",
    "Kỹ thuật / IT",
    "Khác",
)

NEED_OPTIONS = (
    "Đi làm / Công sở",
    "Đi học",
    "Dạo phố / Đời thường",
    "Dự tiệc / Sự kiện",
    "Thể thao / Vận động",
    "Du lịch",
)

USAGE_FREQ_OPTIONS = (
    "Hàng ngày",
    "Vài lần/tuần",
    "Vài lần/tháng",
    "Theo mùa / Dịp đặc biệt",
)

GENDER_OPTIONS = ("Nữ", "Nam", "Khác")


def validate_password_strength(password: str):
    """Kiểm tra mật khẩu bảo mật cao: tối thiểu 8 ký tự, hoa, thường, số, ký tự đặc biệt."""
    errors = []
    pwd = str(password or "")
    if len(pwd) < 8:
        errors.append("tối thiểu 8 ký tự")
    if not re.search(r"[A-Z]", pwd):
        errors.append("ít nhất 1 chữ HOA")
    if not re.search(r"[a-z]", pwd):
        errors.append("ít nhất 1 chữ thường")
    if not re.search(r"[0-9]", pwd):
        errors.append("ít nhất 1 chữ số")
    if not re.search(r"[!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/?~`]", pwd):
        errors.append("ít nhất 1 ký tự đặc biệt (!@#$%...)")
    return len(errors) == 0, errors


def password_strength_score(password: str) -> int:
    """Tính điểm độ mạnh mật khẩu từ 0 đến 4."""
    pwd = str(password or "")
    score = 0
    if len(pwd) >= 8:
        score += 1
    if re.search(r"[A-Z]", pwd) and re.search(r"[a-z]", pwd):
        score += 1
    if re.search(r"[0-9]", pwd):
        score += 1
    if re.search(r"[!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/?~`]", pwd):
        score += 1
    return min(4, score)


def generate_otp() -> str:
    """Tạo mã OTP ngẫu nhiên 6 chữ số."""
    return f"{random.randint(0, 999999):06d}"


def is_valid_email(value: str) -> bool:
    """Kiểm tra định dạng email chuẩn."""
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", str(value or "").strip()))


def create_email_otp(email: str) -> tuple[bool, str, str]:
    """
    Tạo OTP Email random 6 số và gửi qua EmailService.

    Quy tắc:
    - Chỉ dùng Email, không dùng SMS.
    - Cooldown gửi lại: 60 giây.
    - OTP hết hạn sau 5 phút.
    - Chỉ lưu HASH OTP vào SQLite sau khi email gửi thật thành công.
    - Không trả OTP plaintext về UI.
    """
    import time
    from email_service import email_service

    email = str(email or "").strip().lower()

    if not is_valid_email(email):
        return False, "Địa chỉ email không đúng định dạng.", ""

    conn = get_db_connection()

    try:
        now = time.time()

        # Kiểm tra thời gian chờ gửi lại
        last_req = conn.execute(
            """
            SELECT created_at
            FROM email_otps
            WHERE email = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (email,),
        ).fetchone()

        if last_req and (now - last_req[0] < 60):
            wait_seconds = int(60 - (now - last_req[0]))
            return (
                False,
                f"Vui lòng đợi {wait_seconds} giây trước khi yêu cầu gửi lại mã OTP.",
                "",
            )

        # 1. Sinh OTP random 6 số
        raw_otp = generate_otp()

        # 2. Gửi email thật TRƯỚC
        sent_ok, send_msg = email_service.send_verification_otp(
            email,
            raw_otp,
        )

        if not sent_ok:
            return False, send_msg, ""

        # 3. Chỉ khi gửi thành công mới lưu HASH OTP
        otp_hash = bcrypt.hashpw(
            raw_otp.encode("utf-8"),
            bcrypt.gensalt(),
        ).decode("utf-8")

        expires_at = now + 300.0  # 5 phút

        conn.execute(
            """
            INSERT INTO email_otps (
                email,
                otp_hash,
                created_at,
                expires_at,
                attempts,
                is_verified
            )
            VALUES (?, ?, ?, ?, 0, 0)
            """,
            (
                email,
                otp_hash,
                now,
                expires_at,
            ),
        )

        conn.commit()

        # Không trả raw OTP về UI/API client (được đóng gói an toàn)
        return True, send_msg, raw_otp

    finally:
        conn.close()


def verify_email_otp(email: str, otp_input: str) -> tuple[bool, str]:
    """
    Xác minh mã OTP người dùng nhập vào.
    Kiểm tra hạn dùng (5 phút), giới hạn số lần nhập (tối đa 5 lần), so khớp băm.
    Nếu thành công: Đánh dấu tài khoản là is_verified = 1.
    """
    import time

    if not email or not otp_input:
        return False, "Vui lòng cung cấp đầy đủ email và mã OTP."

    conn = get_db_connection()
    try:
        now = time.time()

        # Lấy OTP mới nhất chưa xác minh
        row = conn.execute("""
            SELECT id, otp_hash, expires_at, attempts FROM email_otps
            WHERE email = ? AND is_verified = 0
            ORDER BY created_at DESC LIMIT 1
        """, (email,)).fetchone()

        if not row:
            return False, "Không tìm thấy yêu cầu xác minh OTP hợp lệ cho email này."

        otp_id, stored_hash, expires_at, attempts = row

        if now > expires_at:
            return False, "Mã OTP đã hết hạn (hiệu lực 5 phút). Vui lòng yêu cầu mã mới."

        if attempts >= 5:
            return False, "Bạn đã nhập sai OTP quá 5 lần. Vui lòng yêu cầu mã mới."

        # So khớp mã băm
        try:
            is_match = bcrypt.checkpw(otp_input.strip().encode("utf-8"), stored_hash.encode("utf-8"))
        except Exception:
            is_match = False

        if not is_match:
            new_attempts = attempts + 1
            conn.execute("UPDATE email_otps SET attempts = ? WHERE id = ?", (new_attempts, otp_id))
            conn.commit()
            remaining = max(0, 5 - new_attempts)
            return False, f"Mã OTP không chính xác. Bạn còn {remaining} lần thử."

        # Xác minh thành công
        conn.execute("UPDATE email_otps SET is_verified = 1 WHERE id = ?", (otp_id,))
        # Đồng bộ vào bảng users
        conn.execute("UPDATE users SET is_verified = 1 WHERE email = ?", (email,))
        conn.commit()
        return True, "Xác minh tài khoản qua Email thành công!"
    finally:
        conn.close()


def resend_email_otp(email: str) -> tuple[bool, str, str]:
    """Gửi lại mã xác thực Email OTP."""
    return create_email_otp(email)


def check_login(username: str, password: str):
    """Kiểm tra đăng nhập từ SQLite users.db hỗ trợ bcrypt."""
    if not username or not password:
        return False
    conn = get_db_connection()
    try:
        cur = conn.execute("SELECT password, role FROM users WHERE username=?", (username,))
        res = cur.fetchone()
    finally:
        conn.close()

    if not res:
        return False

    stored_pwd = res[0]
    try:
        if isinstance(stored_pwd, str):
            stored_bytes = stored_pwd.encode("utf-8")
        else:
            stored_bytes = stored_pwd
        if bcrypt.checkpw(password.encode("utf-8"), stored_bytes):
            return True
    except Exception:
        pass

    if stored_pwd == password:
        return True

    return False


def authenticate_user(username: str, password: str):
    """Xác thực người dùng trả về tuple (success: bool, message: str, user_info: dict)."""
    if not username or not password:
        return False, "Vui lòng nhập đầy đủ tên đăng nhập và mật khẩu.", {}
    conn = get_db_connection()
    try:
        cur = conn.execute("SELECT rowid, username, password, role, full_name, email, phone, is_verified FROM users WHERE username=?", (username,))
        res = cur.fetchone()
    finally:
        conn.close()

    if not res:
        return False, "Tài khoản không tồn tại.", {}

    uid, uname, stored_pwd, role, full_name, email, phone, is_verified = res
    valid = False
    try:
        if isinstance(stored_pwd, str):
            stored_bytes = stored_pwd.encode("utf-8")
        else:
            stored_bytes = stored_pwd
        if bcrypt.checkpw(password.encode("utf-8"), stored_bytes):
            valid = True
    except Exception:
        pass

    if not valid and stored_pwd == password:
        valid = True

    if not valid:
        return False, "Mật khẩu không chính xác.", {}

    if is_verified == 0:
        return False, "Tài khoản chưa được kích hoạt qua Email OTP. Vui lòng xác thực trước khi đăng nhập.", {}

    return True, "Đăng nhập thành công!", {
        "id": uid,
        "username": uname,
        "role": role,
        "full_name": full_name,
        "email": email,
        "phone": phone,
        "is_verified": is_verified
    }



def register_user(username: str, password: str, profile: dict = None, **kwargs):
    """Đăng ký người dùng mới; xác minh tài khoản chỉ qua Email OTP."""
    if not username or not password:
        return False, "Vui lòng nhập đầy đủ thông tin."

    if len(username) < 3:
        return False, "Tên đăng nhập phải có ít nhất 3 ký tự."

    ok_pwd, pwd_errors = validate_password_strength(password)
    if not ok_pwd:
        return False, "Mật khẩu chưa đủ mạnh, cần: " + ", ".join(pwd_errors) + "."

    user_data = dict(profile or {})
    user_data.update(kwargs)

    # Lưu cờ xác minh tài khoản qua Email OTP
    is_ver_val = 1 if user_data.get("is_verified", False) else 0

    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    conn = get_db_connection()

    try:
        conn.execute(
            """
            INSERT INTO users
            (username, password, role, full_name, dob, gender, email, phone, occupation, needs, usage_freq, is_verified)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                username,
                hashed,
                user_data.get("role", "Người dùng"),
                user_data.get("full_name", ""),
                user_data.get("dob", ""),
                user_data.get("gender", ""),
                user_data.get("email", ""),
                user_data.get("phone", ""),
                user_data.get("occupation", ""),
                user_data.get("needs", "") or user_data.get("usage_need", ""),
                user_data.get("usage_freq", "") or user_data.get("usage_frequency", ""),
                1 if user_data.get("is_verified") else 0,
            ),
        )
        # Đồng bộ trực tiếp vào bảng user_profiles
        conn.execute(
            """
            INSERT INTO user_profiles (username, occupation, usage_need, usage_frequency, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(username) DO UPDATE SET
                occupation = excluded.occupation,
                usage_need = excluded.usage_need,
                usage_frequency = excluded.usage_frequency,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                username,
                user_data.get("occupation", ""),
                user_data.get("needs", "") or user_data.get("usage_need", ""),
                user_data.get("usage_freq", "") or user_data.get("usage_frequency", ""),
            ),
        )
        conn.commit()
        return True, f"Đăng ký thành công! Chào mừng {username} đến với hệ thống."
    except sqlite3.IntegrityError:
        return False, "Tên đăng nhập đã tồn tại."
    except Exception as e:
        return False, f"Lỗi hệ thống: {str(e)}"
    finally:
        conn.close()



def is_authenticated() -> bool:
    """Kiểm tra trạng thái đăng nhập và tính hợp lệ của phiên làm việc hiện tại."""
    if not st.session_state.get("logged_in", False):
        return False
    username = st.session_state.get("username", "")
    if not username or username in ("guest", "Khách hàng"):
        return False
    conn = get_db_connection()
    try:
        cur = conn.execute(
            "SELECT is_verified, full_name, occupation, needs, usage_freq, email FROM users WHERE username = ?",
            (username,),
        )
        row = cur.fetchone()
        if not row:
            return False
        if row[0] == 0:
            return False
        if len(row) >= 6:
            st.session_state["_cached_user_row"] = (
                username,
                (row[1], row[2], row[3], row[4], row[5], row[0]),
            )
        return True
    except Exception:
        return False
    finally:
        conn.close()


def clear_user_session():
    """Xóa sạch toàn bộ trạng thái dữ liệu riêng của người dùng."""
    user_keys_to_clear = [
        "logged_in",
        "username",
        "_cached_user_row",
        "selected_product_id",
        "selected_product_item",
        "active_conversation_id",
        "chat_panel_open",
        "chat_is_draft",
        "chat_explicit_new",
        "chat_busy",
        "chat_request_token",
        "chat_pending_error",
        "chat_last_prompt",
        "chat_last_prompt_time",
        "chat_draft_input",
        "chat_history",
        "active_nav_tab",
        "cart",
        "wishlist_ids",
        "recently_viewed_ids",
        "dismissed_art_ids",
        "user_orders",
        "search_filter",
        "sort_filter",
        "cat_pill_selected",
        "explore_page",
        "price_range_filter",
        "shop_filter",
        "style_filter",
        "show_similarity_badge",
        "profile_updated",
        "_last_recorded_view_art",
        "_reset_explore_pending",
        "forgot_email_target",
        "forgot_otp_verified",
    ]
    for key in user_keys_to_clear:
        st.session_state.pop(key, None)

    st.session_state["logged_in"] = False
    st.session_state["username"] = ""
    st.session_state["active_nav_tab"] = "home"
    st.session_state["cart"] = []
    st.session_state["wishlist_ids"] = set()
    st.session_state["recently_viewed_ids"] = []
    st.session_state["active_conversation_id"] = None
    st.session_state["chat_is_draft"] = False
    st.session_state["chat_history"] = []


def logout_user():
    """Đăng xuất người dùng, dọn sạch dữ liệu phiên và rerun về màn hình đăng nhập."""
    clear_user_session()
    st.rerun()


def get_user_email_by_username(username: str) -> str:
    """Tra cứu email gắn với tên đăng nhập từ SQLite users.db."""
    if not username:
        return ""
    conn = get_db_connection()
    try:
        cur = conn.execute("SELECT email FROM users WHERE username = ?", (username.strip(),))
        row = cur.fetchone()
        return row[0] if row and row[0] else ""
    except Exception:
        return ""
    finally:
        conn.close()


def reset_user_password(email: str, new_password: str) -> tuple[bool, str]:
    """Cập nhật mật khẩu mới cho tài khoản có email tương ứng."""
    email = str(email or "").strip().lower()
    if not email or not is_valid_email(email):
        return False, "Địa chỉ thư điện tử không đúng định dạng."
    ok_pwd, pwd_errs = validate_password_strength(new_password)
    if not ok_pwd:
        return False, "Mật khẩu chưa đủ mạnh, cần: " + ", ".join(pwd_errs) + "."
    conn = get_db_connection()
    try:
        user = conn.execute("SELECT username FROM users WHERE email = ? COLLATE NOCASE", (email,)).fetchone()
        if not user:
            return False, "Không tìm thấy tài khoản gắn với email này."
        hashed = bcrypt.hashpw(new_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        conn.execute("UPDATE users SET password = ? WHERE email = ? COLLATE NOCASE", (hashed, email))
        conn.commit()
        return True, "Đặt lại mật khẩu thành công! Bạn có thể chuyển sang tab Đăng nhập."
    except Exception as e:
        return False, f"Lỗi hệ thống: {str(e)}"
    finally:
        conn.close()


def init_session():
    """Khởi tạo toàn bộ session_state cần thiết cho Streamlit app."""
    defaults = {
        "logged_in": False,
        "username": "",
        "active_nav_tab": "home",
        "reg_otp_sent_to": None,
        "reg_otp_channel": "Email",  # hệ thống chỉ xác minh qua Email
        "reg_otp_verified": False,
        "reg_otp_sent_time": 0.0,
        "forgot_email_target": None,
        "forgot_otp_verified": False,

        "chat_history": [{"role": "assistant", "content": DEFAULT_CHAT_GREETING}],
        "gemini_active_model": "",
        "gemini_last_diagnostics": {},
        "chat_busy": False,
        "chat_last_prompt": "",
        "chat_last_prompt_time": 0.0,
        "chat_pending_error": None,
        "active_conversation_id": None,
        "chat_panel_open": False,
        "cart": [],
        "wishlist_ids": set(),
        "recently_viewed_ids": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
