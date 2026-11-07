"""Giao diện đăng nhập, đăng ký và xác minh OTP theo phong cách Fashion E-commerce hiện đại (Sections 28, 29, 30)."""

import re
import time
from datetime import date

import streamlit as st

from auth import (
    GENDER_OPTIONS,
    NEED_OPTIONS,
    OCCUPATION_OPTIONS,
    USAGE_FREQ_OPTIONS,
    check_login,
    create_email_otp,
    get_user_email_by_username,
    is_valid_email,
    password_strength_score,
    register_user,
    reset_user_password,
    validate_password_strength,
    verify_email_otp,
)
from ui.chat_panel import prepare_chat_session_for_user
from ui.components.icons import icon_svg
from ui.styles import FASHION_BANNER_BASE64


def required_label(text: str):
    st.markdown(
        f"""
        <div style="
            font-size: 13px;
            font-weight: 600;
            margin-bottom: -0.35rem;
            color: #2D2438;
        ">
            {text} <span style="color:#B91C1C;font-weight:700;">*</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_section_badge(num: str, title: str):
    st.markdown(
        f"""
        <div class="reg-sec-head">
            <span class="reg-sec-num">{num}</span>
            <span class="reg-sec-title">{title}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_password_strength_block(password: str):
    """Hiển thị thang độ mạnh và các điều kiện bảo mật của mật khẩu."""
    pwd = str(password or "")
    score = password_strength_score(pwd)

    labels = [
        "Rất yếu",
        "Yếu",
        "Trung bình",
        "Mạnh",
        "Rất mạnh",
    ]
    colors = [
        "#DC2626",
        "#EA580C",
        "#CA8A04",
        "#15803D",
        "#2D2438",
    ]

    active_bars = score + 1 if pwd else 0
    bar_color = colors[score] if pwd else "#999999"
    status_label = labels[score] if pwd else "Chưa nhập"

    bars_html = "".join(
        f'<div class="pwd-bar-seg" style="background: {bar_color if i < active_bars else "#E2E2DF"};"></div>'
        for i in range(5)
    )

    checks = [
        ("Ít nhất 8 ký tự", len(pwd) >= 8),
        ("Chữ hoa", bool(re.search(r"[A-Z]", pwd))),
        ("Chữ thường", bool(re.search(r"[a-z]", pwd))),
        ("Số", bool(re.search(r"[0-9]", pwd))),
        ("Ký tự đặc biệt", bool(re.search(r"[!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/?~`]", pwd))),
    ]

    checks_html = "".join(
        f'<div class="pwd-check-item {"ok" if passed else "missing"}">'
        f'<span style="display:inline-flex;width:16px;vertical-align:middle;" '
        f'aria-label="{"Đạt" if passed else "Chưa đạt"}">'
        f'{icon_svg("check", size=14) if passed else "—"}</span> {label}'
        f'</div>'
        for label, passed in checks
    )

    st.markdown(
        f"""
        <div class="pwd-meter-wrap">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:12px; font-weight:600; color:#2D2438;">Mật khẩu</span>
                <span style="font-size:12px; font-weight:700; color:{bar_color};">{status_label}</span>
            </div>
            <div class="pwd-bars">{bars_html}</div>
            <div class="pwd-checklist">{checks_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def login_page():
    """
    Giao diện Đăng nhập / Đăng ký Split Screen (Mục 28, 29, 30):
    - Desktop: Cột trái là Fashion Image Editorial, Cột phải là Form Đăng nhập / Đăng ký.
    - Mobile: Hero thu gọn ở trên, Form full width ở dưới.
    """
    with st.container(key="fashion_split_auth"):
        col_visual, col_auth = st.columns([1.05, 1.15], gap="large")

        with col_visual:
            st.markdown(
                f"""
                <div class="fashion-auth-left-panel" style="background: linear-gradient(180deg, rgba(17,17,17,0.38) 0%, rgba(17,17,17,0.82) 100%), url('data:image/jpeg;base64,{FASHION_BANNER_BASE64}') center/cover no-repeat;">
                    <div>
                        <div style="display:inline-flex; align-items:center; gap:8px; background:rgba(255,255,255,0.14); backdrop-filter:blur(8px); padding:6px 14px; border-radius:999px; font-size:11px; font-weight:700; letter-spacing:0.16em; text-transform:uppercase;">
                            FASHION E-COMMERCE
                        </div>
                    </div>
                    <div>
                        <div style="font-size:11px; font-weight:700; letter-spacing:0.22em; color:#C3B1E1; text-transform:uppercase; margin-bottom:10px;">
                            THỜI TRANG · PHONG CÁCH · CẢM HỨNG
                        </div>
                        <div style="font-family:'Playfair Display', serif; font-size:clamp(28px, 3.2vw, 42px); font-weight:700; line-height:1.12; margin-bottom:12px;">
                            THỜI TRANG<br>DÀNH CHO BẠN
                        </div>
                        <div style="font-size:14px; color:rgba(255,255,255,0.88); line-height:1.6; max-width:420px;">
                            Khám phá trang phục và phụ kiện phù hợp với nhu cầu và phong cách cá nhân của bạn mỗi ngày.
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_auth:
            st.markdown(
                """
                <div class="fashion-auth-right-header">
                    <div class="fashion-auth-brand">FASHION</div>
                    <div class="fashion-auth-sub">Thời trang · Phong cách · Cá nhân hóa</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            tab_login, tab_register, tab_forgot = st.tabs(["Đăng nhập", "Đăng ký", "Quên mật khẩu"])

            # ========================================================
            # TAB ĐĂNG NHẬP (MỤC 28)
            # ========================================================
            with tab_login:
                with st.container(border=True):
                    with st.form(
                        "login_form",
                        clear_on_submit=False,
                        enter_to_submit=True,
                    ):
                        login_user = st.text_input(
                            "Tên đăng nhập",
                            key="login_user",
                            placeholder="Nhập tên đăng nhập của bạn",
                        )

                        login_pwd = st.text_input(
                            "Mật khẩu",
                            type="password",
                            key="login_pwd",
                            placeholder="••••••••",
                        )

                        st.markdown('<div style="height: 6px;"></div>', unsafe_allow_html=True)

                        login_submit = st.form_submit_button(
                            "ĐĂNG NHẬP",
                            type="primary",
                            width="stretch",
                        )

                    if login_submit:
                        if not login_user or not login_pwd:
                            st.error("Vui lòng nhập đầy đủ thông tin.")
                        elif check_login(login_user, login_pwd):
                            st.session_state.logged_in = True
                            st.session_state.username = login_user
                            st.session_state.active_nav_tab = "home"
                            st.session_state.pop("user_orders", None)
                            st.session_state.pop("cart", None)
                            st.session_state.pop("wishlist_ids", None)
                            st.session_state.pop("recently_viewed_ids", None)
                            prepare_chat_session_for_user(login_user)
                            st.rerun()
                        else:
                            st.error("Sai tên đăng nhập hoặc mật khẩu.")

            # ========================================================
            # TAB ĐĂNG KÝ (MỤC 29 & 30)
            # ========================================================
            with tab_register:
                step1_done = bool(st.session_state.get("reg_user") and st.session_state.get("reg_fullname"))
                step2_done = bool(st.session_state.get("reg_otp_verified"))
                step3_done = bool(st.session_state.get("reg_needs"))

                st.markdown(
                    f"""
                    <div class="reg-stepper">
                        <span class="reg-step-item active">01 Tài khoản {icon_svg("check", size=14) if step1_done else ''}</span>
                        <div class="reg-step-line"></div>
                        <span class="reg-step-item {'active' if step1_done or step2_done else ''}">02 Xác minh {icon_svg("check", size=14) if step2_done else ''}</span>
                        <div class="reg-step-line"></div>
                        <span class="reg-step-item {'active' if step2_done or step3_done else ''}">03 Phong cách {icon_svg("check", size=14) if step3_done else ''}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                with st.container(border=True):
                    # 01: TÀI KHOẢN
                    _render_section_badge("01", "Tài khoản")

                    c1, c2 = st.columns(2)
                    with c1:
                        required_label("Họ và tên")
                        reg_fullname = st.text_input(
                            "Họ và tên",
                            key="reg_fullname",
                            placeholder="Nguyễn Văn A",
                            label_visibility="collapsed",
                        )

                        required_label("Tên đăng nhập")
                        reg_user = st.text_input(
                            "Tên đăng nhập",
                            key="reg_user",
                            placeholder="Tối thiểu 3 ký tự",
                            label_visibility="collapsed",
                        )

                    with c2:
                        required_label("Ngày sinh")
                        reg_dob = st.date_input(
                            "Ngày sinh",
                            key="reg_dob",
                            min_value=date(1940, 1, 1),
                            max_value=date.today(),
                            value=date(2000, 1, 1),
                            label_visibility="collapsed",
                        )

                        required_label("Giới tính")
                        reg_gender = st.selectbox(
                            "Giới tính",
                            GENDER_OPTIONS,
                            key="reg_gender",
                            label_visibility="collapsed",
                        )

                    st.divider()

                    # 02: BẢO MẬT
                    _render_section_badge("02", "Bảo mật")

                    c3, c4 = st.columns(2)
                    with c3:
                        required_label("Mật khẩu")
                        reg_pwd = st.text_input(
                            "Mật khẩu",
                            type="password",
                            key="reg_pwd",
                            placeholder="Tối thiểu 8 ký tự, Hoa/thường/số/ký tự đặc biệt",
                            label_visibility="collapsed",
                        )

                    with c4:
                        required_label("Xác nhận mật khẩu")
                        reg_pwd2 = st.text_input(
                            "Xác nhận mật khẩu",
                            type="password",
                            key="reg_pwd2",
                            placeholder="Nhập lại mật khẩu",
                            label_visibility="collapsed",
                        )

                    _render_password_strength_block(reg_pwd)
                    if reg_pwd:
                        ok_pwd, pwd_errs = validate_password_strength(reg_pwd)
                        if not ok_pwd:
                            st.caption("Cần bổ sung: " + ", ".join(pwd_errs))

                    st.divider()

                    # 03: XÁC MINH EMAIL
                    _render_section_badge("03", "Xác minh Email")

                    oc1, oc2 = st.columns([3, 1], vertical_alignment="bottom")
                    with oc1:
                        required_label("Địa chỉ Email nhận mã xác minh")
                        reg_contact = st.text_input(
                            "Thư điện tử",
                            key="reg_email",
                            placeholder="Ví dụ: ban@gmail.com",
                            label_visibility="collapsed",
                        )

                    with oc2:
                        send_otp_clicked = st.button(
                            "Gửi mã",
                            key="btn_reg_send_otp",
                            width="stretch",
                        )

                    if (
                        st.session_state.reg_otp_sent_to
                        and not st.session_state.reg_otp_verified
                    ):
                        vc1, vc2 = st.columns([3, 1], vertical_alignment="bottom")
                        with vc1:
                            required_label("Nhập mã xác minh (6 số)")
                            otp_input = st.text_input(
                                "Nhập mã xác minh (6 số)",
                                key="reg_otp_input",
                                max_chars=6,
                                placeholder="Nhập 6 chữ số từ hộp thư",
                                label_visibility="collapsed",
                            )
                        with vc2:
                            verify_clicked = st.button(
                                "Xác nhận",
                                key="btn_reg_verify_otp",
                                width="stretch",
                            )
                    else:
                        otp_input = ""
                        verify_clicked = False

                    if (
                        st.session_state.reg_otp_sent_to
                        and not st.session_state.reg_otp_verified
                    ):
                        st.info(
                            f"Mã xác minh 6 số đã được gửi tới **{st.session_state.reg_otp_sent_to}**. "
                            "Vui lòng kiểm tra Hộp thư đến hoặc Thư rác rồi nhập mã phía trên."
                        )

                    if st.session_state.reg_otp_verified:
                        st.success(
                            f"Đã xác minh Email thành công: **{st.session_state.reg_otp_sent_to}**"
                        )

                    st.divider()

                    # 04: PHONG CÁCH CỦA BẠN
                    _render_section_badge("04", "Phong cách của bạn")
                    st.caption(
                        "Thông tin này giúp hệ thống cá nhân hóa gợi ý trang phục phù hợp riêng cho bạn."
                    )

                    p1, p2 = st.columns(2)
                    with p1:
                        required_label("Nghề nghiệp")
                        reg_occupation = st.selectbox(
                            "Nghề nghiệp",
                            OCCUPATION_OPTIONS,
                            key="reg_occupation",
                            label_visibility="collapsed",
                        )

                    with p2:
                        required_label("Tần suất sử dụng")
                        reg_usage_freq = st.selectbox(
                            "Tần suất / thời gian sử dụng sản phẩm",
                            USAGE_FREQ_OPTIONS,
                            key="reg_usage_freq",
                            label_visibility="collapsed",
                        )

                    required_label("Nhu cầu sử dụng chính (có thể chọn nhiều)")
                    reg_needs = st.multiselect(
                        "Nhu cầu sử dụng chính (có thể chọn nhiều)",
                        NEED_OPTIONS,
                        key="reg_needs",
                        placeholder="Chọn 1 hoặc nhiều nhu cầu",
                        label_visibility="collapsed",
                    )

                    st.markdown('<div style="height: 8px;"></div>', unsafe_allow_html=True)

                    register_submit = st.button(
                        "HOÀN TẤT ĐĂNG KÝ",
                        key="btn_reg_submit_final",
                        type="primary",
                        width="stretch",
                    )

                # XỬ LÝ GỬI MÃ OTP
                if send_otp_clicked:
                    clean_email = str(reg_contact or "").strip().lower()
                    if not clean_email or not is_valid_email(clean_email):
                        st.error("Địa chỉ thư điện tử không đúng định dạng.")
                    else:
                        ok_send, msg_send, _ = create_email_otp(clean_email)
                        if not ok_send:
                            st.error(msg_send)
                        else:
                            st.session_state.reg_otp_sent_to = clean_email
                            st.session_state.reg_otp_channel = "Email"
                            st.session_state.reg_otp_verified = False
                            st.session_state.reg_otp_sent_time = time.time()
                            st.success(msg_send)
                            st.rerun()

                # XỬ LÝ XÁC NHẬN MÃ OTP
                if verify_clicked:
                    clean_input = str(otp_input or "").strip()
                    if not clean_input:
                        st.error("Vui lòng nhập đủ 6 chữ số mã xác minh.")
                    elif len(clean_input) != 6 or not clean_input.isdigit():
                        st.error("Mã xác minh phải gồm đúng 6 chữ số.")
                    else:
                        target_email = st.session_state.reg_otp_sent_to
                        ok_ver, msg_ver = verify_email_otp(target_email, clean_input)
                        if ok_ver:
                            st.session_state.reg_otp_verified = True
                            st.success(msg_ver)
                            st.rerun()
                        else:
                            st.error(msg_ver)

                # XỬ LÝ ĐĂNG KÝ TÀI KHOẢN
                if register_submit:
                    missing_fields = []
                    if not str(reg_fullname).strip():
                        missing_fields.append("Họ và tên")
                    if not str(reg_user).strip():
                        missing_fields.append("Tên đăng nhập")
                    if not str(reg_pwd).strip():
                        missing_fields.append("Mật khẩu")
                    if not str(reg_pwd2).strip():
                        missing_fields.append("Xác nhận mật khẩu")

                    if missing_fields:
                        st.error("Bạn còn thiếu: " + ", ".join(missing_fields))
                    elif reg_pwd != reg_pwd2:
                        st.error("Mật khẩu xác nhận không khớp.")
                    elif not validate_password_strength(reg_pwd)[0]:
                        st.error(
                            "Mật khẩu chưa đủ mạnh (cần Hoa/thường/số/ký tự đặc biệt, tối thiểu 8 ký tự)."
                        )
                    elif (
                        not st.session_state.reg_otp_verified
                        or st.session_state.reg_otp_sent_to
                        != str(reg_contact or "").strip().lower()
                    ):
                        st.error(
                            "Vui lòng nhấn 'Gửi mã' và hoàn tất xác minh mã qua Email trước khi tạo tài khoản."
                        )
                    elif not reg_needs:
                        st.error(
                            "Vui lòng chọn ít nhất 1 nhu cầu sử dụng để hệ thống gợi ý sản phẩm phù hợp."
                        )
                    else:
                        profile = {
                            "full_name": reg_fullname,
                            "dob": reg_dob.strftime("%Y-%m-%d"),
                            "gender": reg_gender,
                            "email": str(reg_contact or "").strip().lower(),
                            "phone": "",
                            "occupation": reg_occupation,
                            "needs": ",".join(reg_needs),
                            "usage_freq": reg_usage_freq,
                            "is_verified": 1,
                        }

                        ok, msg = register_user(reg_user, reg_pwd, profile)
                        if ok:
                            st.success(msg)
                            st.session_state.reg_otp_sent_to = None
                            st.session_state.reg_otp_verified = False
                            st.session_state.reg_otp_sent_time = 0.0
                        else:
                            st.error(msg)

            # ========================================================
            # TAB QUÊN MẬT KHẨU
            # ========================================================
            with tab_forgot:
                st.caption(
                    "Khôi phục quyền truy cập tài khoản qua mã xác minh gửi tới hộp thư Email của bạn."
                )
                with st.container(border=True):
                    _render_section_badge("01", "Xác thực danh tính")

                    forgot_id = st.text_input(
                        "Tên đăng nhập hoặc Email",
                        key="forgot_id_input",
                        placeholder="Nhập tên đăng nhập hoặc email đã đăng ký",
                    )

                    btn_forgot_send_otp = st.button(
                        "Gửi mã xác minh",
                        key="btn_forgot_send_otp",
                        width="stretch",
                    )

                    if btn_forgot_send_otp:
                        clean_id = str(forgot_id or "").strip()
                        if not clean_id:
                            st.error("Vui lòng nhập tên đăng nhập hoặc email.")
                        else:
                            target_email = clean_id.lower() if is_valid_email(clean_id) else get_user_email_by_username(clean_id)
                            if not target_email or not is_valid_email(target_email):
                                st.error("Không tìm thấy tài khoản hoặc email hợp lệ cho thông tin đã nhập.")
                            else:
                                ok_send, msg_send, _ = create_email_otp(target_email)
                                if ok_send:
                                    st.session_state.forgot_email_target = target_email
                                    st.session_state.forgot_otp_verified = False
                                    st.success(f"Mã OTP đã được gửi tới **{target_email}**. Vui lòng kiểm tra hộp thư.")
                                    st.rerun()
                                else:
                                    st.error(msg_send)

                    if st.session_state.get("forgot_email_target"):
                        target_email = st.session_state.forgot_email_target
                        st.divider()
                        _render_section_badge("02", "Đặt lại mật khẩu")
                        st.info(f"Đang khôi phục mật khẩu cho: **{target_email}**")

                        f_otp = st.text_input(
                            "Mã xác minh (6 chữ số)",
                            key="forgot_otp_input",
                            max_chars=6,
                            placeholder="Nhập 6 số từ email",
                        )

                        f_new_pwd = st.text_input(
                            "Mật khẩu mới",
                            type="password",
                            key="forgot_new_pwd",
                            placeholder="Tối thiểu 8 ký tự, hoa, thường, số, ký tự đặc biệt",
                        )

                        f_new_pwd2 = st.text_input(
                            "Xác nhận mật khẩu mới",
                            type="password",
                            key="forgot_new_pwd2",
                            placeholder="Nhập lại mật khẩu mới",
                        )

                        _render_password_strength_block(f_new_pwd)

                        btn_forgot_submit = st.button(
                            "HOÀN TẤT ĐẶT LẠI MẬT KHẨU",
                            key="btn_forgot_submit_final",
                            type="primary",
                            width="stretch",
                        )

                        if btn_forgot_submit:
                            clean_otp = str(f_otp or "").strip()
                            if not clean_otp or len(clean_otp) != 6 or not clean_otp.isdigit():
                                st.error("Vui lòng nhập đúng 6 chữ số mã xác minh.")
                            elif not f_new_pwd or not f_new_pwd2:
                                st.error("Vui lòng nhập mật khẩu mới và xác nhận.")
                            elif f_new_pwd != f_new_pwd2:
                                st.error("Mật khẩu xác nhận không khớp.")
                            elif not validate_password_strength(f_new_pwd)[0]:
                                st.error(
                                    "Mật khẩu chưa đủ mạnh (cần hoa, thường, số, ký tự đặc biệt, tối thiểu 8 ký tự)."
                                )
                            else:
                                ok_ver, msg_ver = verify_email_otp(target_email, clean_otp)
                                if not ok_ver:
                                    st.error(f"Xác minh thất bại: {msg_ver}")
                                else:
                                    ok_reset, msg_reset = reset_user_password(target_email, f_new_pwd)
                                    if ok_reset:
                                        st.success(msg_reset)
                                        st.session_state.pop("forgot_email_target", None)
                                        st.session_state.pop("forgot_otp_verified", None)
                                    else:
                                        st.error(msg_reset)
