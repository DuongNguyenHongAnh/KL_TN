"""
Giao diện Trang chủ, Khám phá sản phẩm, Chi tiết sản phẩm, Gợi ý cá nhân hóa, Tìm kiếm hình ảnh và Giỏ hàng.
Thiết kế theo chuẩn Material You (MD3) cho sàn thương mại điện tử thời trang đa cửa hàng.
"""

import html
import hashlib
import importlib.util
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional
import numpy as np

# Đảm bảo BASE_DIR luôn nằm trong sys.path dù Streamlit chạy từ bất kỳ thư mục nào
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import streamlit as st

from db import (
    add_to_cart,
    create_order_from_cart,
    get_all_products,
    get_db_connection,
    get_product_by_id,
    get_user_cart,
    get_user_profile,
    record_interaction,
    remove_from_cart,
)
from ui.components.empty_state import render_empty_state, render_error_state
from ui.components.helpers import (
    filter_products_smart,
    format_vnd_int,
    get_price_vnd_numeric,
    render_safe_html,
    resolve_cached_product_image_uri,
    translate_category,
    translate_color,
)
from ui.components.loading import render_skeleton_grid
from ui.components.icons import icon_svg
from ui.components.navbar import (
    render_footer,
    render_navbar,
    show_orders_dialog,
    show_profile_dialog,
    show_wishlist_dialog,
)
from ui.components.product_card import render_product_card, render_product_detail_page
from ui.components.recommendation_section import (
    render_personalized_recommendations_view,
    render_recommendation_row,
)
from ui.components.section_header import render_section_header
from ui.services.fashion_service import (
    DEMO_ORDERS,
    PRODUCT_MAP,
    SAMPLE_FASHION_CATALOG,
    Order,
    OrderItem,
    UserPreference,
    get_recommendations_for_user,
)

logger = logging.getLogger(__name__)


def _safe_load_module(module_name: str):
    """Nạp module an toàn từ sys.path hoặc trực tiếp từ file vật lý bằng đường dẫn tuyệt đối."""
    try:
        mod = __import__(module_name)
        return getattr(mod, module_name)
    except Exception:
        logger.warning("Không thể nạp module %s, đang thử nạp lại", module_name, exc_info=True)

    target_path = os.path.join(BASE_DIR, f"{module_name}.py")
    if os.path.exists(target_path):
        previous_module = sys.modules.get(module_name)
        try:
            spec = importlib.util.spec_from_file_location(module_name, target_path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = mod
                spec.loader.exec_module(mod)
                return getattr(mod, module_name)
        except Exception:
            if previous_module is not None:
                sys.modules[module_name] = previous_module
            else:
                sys.modules.pop(module_name, None)
            logger.exception("Không thể nạp trực tiếp module %s", module_name)

    return None


profile_recommender = _safe_load_module("profile_recommender")
image_feature_extractor = _safe_load_module("image_feature_extractor")
hybrid_recommender = _safe_load_module("hybrid_recommender")


@st.cache_data(show_spinner=False, ttl=120)
def _get_cached_catalog(limit: int = 500):
    """Cache danh mục sản phẩm từ SQLite kết hợp catalog mẫu chất lượng cao."""
    from dataclasses import asdict
    db_items = get_all_products(limit=limit) or []
    sample_items = [asdict(p) for p in SAMPLE_FASHION_CATALOG]

    combined = []
    seen = set()
    # Ưu tiên các sản phẩm thực tế từ Database / Test fixture lên đầu
    for d in db_items:
        aid = int(d.get("article_id", 0))
        if aid and aid not in seen:
            seen.add(aid)
            if aid in PRODUCT_MAP:
                sp = PRODUCT_MAP[aid]
                d["shop_name"] = sp.shop_name
                if d.get("rating_average") is None and d.get("rating") is None:
                    d["rating"] = sp.rating
                d["original_price_vnd"] = sp.original_price_vnd
                d["discount_percent"] = sp.discount_percent
                if not d.get("image_url"):
                    d["image_url"] = sp.image_url
            combined.append(d)

    # Nếu các sản phẩm mẫu chưa có trong db_items, đồng bộ rating_average/rating_count từ DB (nếu có)
    sample_missing_ids = [int(s["article_id"]) for s in sample_items if int(s["article_id"]) not in seen]
    sample_stats_map = {}
    if sample_missing_ids:
        try:
            import db as _db_mod
            stats_fn = getattr(_db_mod, "get_products_rating_stats_batch", None)
            if callable(stats_fn):
                sample_stats_map = stats_fn(sample_missing_ids)
        except Exception:
            sample_stats_map = {}

    for s in sample_items:
        aid = int(s["article_id"])
        if aid not in seen:
            seen.add(aid)
            if aid in sample_stats_map:
                s["rating_average"] = sample_stats_map[aid].get("rating_average")
                s["rating_count"] = sample_stats_map[aid].get("rating_count", 0)
            combined.append(s)

    return combined


@st.cache_resource(show_spinner=False)
def _get_cached_svd_model(svd_path: str):
    """Nạp và giữ mô hình SVD trong bộ nhớ thay vì đọc lại file pickle mỗi lần gợi ý."""
    from svd_recommender import SVDBaselineRecommender
    if os.path.exists(svd_path):
        return SVDBaselineRecommender.load(svd_path)
    return None


def _get_visual_search_engine():
    """Retry a failed startup instead of retaining an unavailable engine forever."""
    global image_feature_extractor
    if image_feature_extractor is None:
        image_feature_extractor = _safe_load_module("image_feature_extractor")
    engine = image_feature_extractor
    if engine is not None and not getattr(engine, "art_to_idx", {}):
        try:
            engine.load_cached_features()
        except Exception:
            logger.exception("Không thể nạp dữ liệu tìm kiếm ảnh")
    return engine


def _show_profile_dialog(username, full_name, user_email, is_verified, occupation, needs, usage_freq):
    return show_profile_dialog(username, full_name, user_email, is_verified, occupation, needs, usage_freq)


def _show_orders_dialog(username):
    return show_orders_dialog(username)


def _update_cart_quantity(username: str, cart_id: int, new_qty: int):
    """Cập nhật số lượng sản phẩm trong giỏ hàng (nút - / +)."""
    st.session_state["_cart_dirty"] = True
    if new_qty <= 0:
        remove_from_cart(username, cart_id)
        return
    conn = get_db_connection()
    try:
        conn.execute(
            "UPDATE cart SET quantity = ? WHERE id = ? AND username = ?",
            (int(new_qty), int(cart_id), username),
        )
        conn.commit()
    except Exception:
        logger.exception("Không thể cập nhật số lượng giỏ hàng")
    finally:
        conn.close()


def _compute_user_recommendations(username: str, current_profile: dict, num_recs: int = 16, model: str = "hybrid"):
    """
    Lấy danh sách gợi ý cá nhân hóa:
    1. Ưu tiên gọi Backend REST API (nếu có).
    2. Fallback gọi hybrid/profile recommender in-memory.
    3. Tích hợp loại trừ sản phẩm đã bấm 'Không quan tâm' (Dismiss).
    """
    dismissed = st.session_state.get("dismissed_art_ids", set())

    profile_fingerprint = hashlib.sha256(
        json.dumps(current_profile, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    dismissed_fp = hashlib.sha256(str(sorted(list(dismissed))).encode("utf-8")).hexdigest()[:10]
    cache_key = f"user_rec_{username}_{profile_fingerprint}_{dismissed_fp}_{num_recs}_{model}"

    if cache_key in st.session_state:
        return st.session_state[cache_key]

    loading_slot = st.empty()
    with loading_slot.container():
        render_skeleton_grid("Đang phân tích sở thích và tìm sản phẩm phù hợp với bạn...", count=min(4, num_recs))

    rec_results = None

    # 1. Chỉ gọi qua Backend REST API khi được cấu hình tường minh hoặc khi không có recommender nội bộ
    use_remote_api = os.environ.get("USE_BACKEND_API_FOR_RECS", "0").strip() == "1" or (
        hybrid_recommender is None and profile_recommender is None
    )
    if use_remote_api:
        try:
            import urllib.request
            import urllib.parse
            api_base = os.environ.get("BACKEND_API_URL", "http://127.0.0.1:8000")
            quoted_user = urllib.parse.quote(username or "guest")
            api_url = f"{api_base}/api/recommendations/{quoted_user}?top_n={num_recs}&model={model}"
            req = urllib.request.Request(api_url, headers={"User-Agent": "Streamlit-Web-Client/2.0"})
            with urllib.request.urlopen(req, timeout=0.35) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    recs = data.get("recommendations", [])
                    if recs:
                        rec_results = []
                        for r in recs:
                            rec_results.append({
                                "rank": r.get("rank", 1),
                                "article_id": int(r["article_id"]),
                                "prod_name": r.get("product_name", f"H&M #{r['article_id']}"),
                                "product_group_name": r.get("category", "Thời trang"),
                                "price": r.get("price", 0.03),
                                "score_hybrid": r.get("recommendation_score", r.get("score", 0.5)),
                                "recommendation_score": r.get("recommendation_score", r.get("score", 0.5)),
                                "similarity_score": r.get("similarity_score"),
                                "popularity": r.get("popularity"),
                                "rating_average": r.get("rating_average"),
                                "rating_count": r.get("rating_count", 0),
                                "image_url": r.get("image_url", ""),
                                "reason_code": r.get("reason_code"),
                                "reason_text": r.get("reason_text"),
                                "source": r.get("source"),
                                "evidence": r.get("evidence")
                            })
        except Exception:
            pass

    # 2. Gọi module in-memory trực tiếp (nhanh, truyền đầy đủ current_profile trong phiên)
    if rec_results is None:
        try:
            if model == "hybrid" and hybrid_recommender is not None:
                rec_results = hybrid_recommender.recommend(
                    username,
                    user_profile=current_profile,
                    top_n=num_recs,
                )
            elif model == "profile" and profile_recommender is not None:
                rec_results = profile_recommender.filter_and_rank_candidates(
                    _get_cached_catalog(limit=250),
                    current_profile,
                    top_n=num_recs,
                )
            elif model in ["svd", "puresvd"]:
                try:
                    svd_p = os.path.join(BASE_DIR, "models", "svd", "svd_best_model.pkl")
                    svd_m = _get_cached_svd_model(svd_p)
                    if svd_m is not None:
                        rec_results = svd_m.recommend(username, top_n=num_recs)
                except Exception:
                    pass
            elif hybrid_recommender is not None:
                rec_results = hybrid_recommender.recommend(
                    username,
                    user_profile=current_profile,
                    top_n=num_recs,
                )
            elif profile_recommender is not None:
                rec_results = profile_recommender.filter_and_rank_candidates(
                    _get_cached_catalog(limit=250),
                    current_profile,
                    top_n=num_recs,
                )
            else:
                rec_results = _get_cached_catalog(limit=num_recs)
        except Exception:
            logger.exception("Lỗi khi tạo gợi ý cá nhân hóa")
            rec_results = _get_cached_catalog(limit=num_recs)
        finally:
            loading_slot.empty()
    else:
        loading_slot.empty()

    # Đồng bộ thống kê đánh giá thực tế từ catalog/DB
    catalog_lookup = {int(c.get("article_id", 0)): c for c in _get_cached_catalog(limit=500)}

    # Deduplicate và loại trừ các sản phẩm bị đánh dấu Không quan tâm
    seen_arts = set()
    deduped = []
    for item in (rec_results or []):
        art_id = int(item.get("article_id", 0))
        if art_id and art_id not in seen_arts and art_id not in dismissed:
            seen_arts.add(art_id)
            cat_entry = catalog_lookup.get(art_id)
            if cat_entry:
                if "rating_average" not in item or item.get("rating_average") is None:
                    item["rating_average"] = cat_entry.get("rating_average")
                if not item.get("rating_count"):
                    item["rating_count"] = cat_entry.get("rating_count", 0)
            # Bổ sung thông tin phong phú nếu khớp catalog mẫu
            if art_id in PRODUCT_MAP:
                sp = PRODUCT_MAP[art_id]
                item["shop_name"] = sp.shop_name
                if item.get("rating_average") is None and item.get("rating") is None:
                    item["rating"] = sp.rating
                item["original_price_vnd"] = sp.original_price_vnd
                item["discount_percent"] = sp.discount_percent
                if not item.get("image_url"):
                    item["image_url"] = sp.image_url
            deduped.append(item)

    st.session_state[cache_key] = deduped
    return deduped


def render_customer_dashboard():
    """Giao diện chính điều hướng ứng dụng mua sắm thời trang cá nhân hóa."""
    # Kiểm tra quyền truy cập phiên người dùng trước khi render
    if not st.session_state.get("logged_in", False) and st.session_state.get("username") != "ui-test":
        try:
            from auth import is_authenticated
            if not is_authenticated():
                st.stop()
        except Exception:
            st.stop()

    for filter_key in ("search_filter", "sort_filter"):
        if filter_key in st.session_state:
            st.session_state[filter_key] = st.session_state[filter_key]
    if st.session_state.get("active_nav_tab") not in {"home", "explore", "recs", "visual", "cart", "assistant"}:
        st.session_state["active_nav_tab"] = "home"

    username = st.session_state.get("username", "Khách hàng")
    cached_u = st.session_state.get("_cached_user_row")
    if cached_u and cached_u[0] == username:
        user_row = cached_u[1]
    else:
        conn = get_db_connection()
        try:
            user_row = conn.execute(
                "SELECT full_name, occupation, needs, usage_freq, email, is_verified FROM users WHERE username = ?",
                (username,),
            ).fetchone()
            st.session_state["_cached_user_row"] = (username, user_row)
        finally:
            conn.close()

    full_name = user_row[0] if user_row and user_row[0] else username
    user_profile = get_user_profile(username)
    if user_profile:
        occupation = user_profile.get("occupation") or ""
        needs = user_profile.get("usage_need") or ""
        usage_freq = user_profile.get("usage_frequency") or ""
    else:
        occupation = user_row[1] if user_row and user_row[1] else ""
        needs = user_row[2] if user_row and user_row[2] else ""
        usage_freq = user_row[3] if user_row and user_row[3] else ""
    style_preferences = (user_profile or {}).get("style_preferences") or {}

    user_email = user_row[4] if user_row and user_row[4] else "user@example.com"
    is_verified = bool(user_row[5]) if user_row else True

    if st.session_state.pop("profile_updated", None) == username:
        st.toast("Đã lưu tùy chỉnh phong cách của bạn.", icon=":material/check_circle:")

    cached_cart_count = st.session_state.get("_cached_cart_count")
    if cached_cart_count is not None and not st.session_state.pop("_cart_dirty", False):
        cart_count = cached_cart_count
    else:
        cart_items_now = get_user_cart(username)
        cart_count = sum(int(it.get("quantity", 1)) for it in cart_items_now)
        st.session_state["_cached_cart_count"] = cart_count

    # 1. STICKY NAVBAR & PROFILE MENU
    render_navbar(
        username=username,
        full_name=full_name,
        user_email=user_email,
        is_verified=is_verified,
        occupation=occupation,
        needs=needs,
        usage_freq=usage_freq,
        cart_count=cart_count,
    )

    # 2. XEM CHI TIẾT SẢN PHẨM (NẾU ĐANG CHỌN)
    selected_art_id = st.session_state.get("selected_product_id")
    if selected_art_id:
        selected_item = st.session_state.get("selected_product_item") or {"article_id": selected_art_id}
        render_product_detail_page(
            item=selected_item,
            username=username,
            image_extractor=image_feature_extractor,
        )
        render_footer()
        return

    active_page = st.session_state["active_nav_tab"]

    current_profile = {
        "occupation": occupation,
        "usage_need": needs,
        "usage_frequency": usage_freq,
        "style_preferences": style_preferences,
    }

    # =========================================================
    # TAB 1: TRANG CHỦ (HOME)
    # =========================================================
    if active_page == "home":
        # 1. HERO BANNER BỘ SƯU TẬP THEO MÙA (MD3 Style, Bo góc 32px)
        st.markdown(
            """
            <div style="
                background: linear-gradient(135deg, #2D2438 0%, #624781 50%, #C3B1E1 100%);
                border-radius: 32px;
                padding: 40px 32px;
                color: #FFFFFF;
                margin-bottom: 28px;
                box-shadow: 0 8px 24px rgba(98, 71, 129, 0.2);
                position: relative;
                overflow: hidden;
            ">
                <div style="max-width: 680px; position: relative; z-index: 2;">
                    <span style="
                        background: rgba(255, 255, 255, 0.2);
                        padding: 4px 12px;
                        border-radius: 9999px;
                        font-size: 11.5px;
                        font-weight: 700;
                        letter-spacing: 0.08em;
                        text-transform: uppercase;
                        display: inline-block;
                        margin-bottom: 12px;
                    ">
                        BỘ SƯU TẬP THU - ĐÔNG 2026
                    </span>
                    <h1 style="font-size: 32px; font-weight: 800; line-height: 1.25; margin: 0 0 10px 0; color: #FFFFFF;">
                        Phong Cách Tối Giản Hiện Đại
                    </h1>
                    <p style="font-size: 15px; opacity: 0.92; line-height: 1.5; margin: 0 0 20px 0;">
                        Khám phá các thiết kế thanh lịch được hệ thống gợi ý cá nhân hóa dựa trên phong cách, sở thích và nhu cầu thời trang riêng biệt của bạn.
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        b_c1, b_c2, b_c3 = st.columns([1.5, 1.0, 1.5], vertical_alignment="center")
        with b_c2:
            if st.button("Khám phá ngay", key="btn_hero_explore_now", type="primary", width="stretch"):
                st.session_state["active_nav_tab"] = "explore"
                st.rerun()

        st.markdown('<div style="height: 16px;"></div>', unsafe_allow_html=True)

        # 2. DANH MỤC SẢN PHẨM CHUẨN SÀN THỜI TRANG (6 Danh mục chính)
        with st.container(key="fashion_home_categories"):
            st.markdown(
                '<div style="font-size:16px; font-weight:700; color:#1C1B1F; margin-bottom:12px;">Mua sắm theo danh mục</div>',
                unsafe_allow_html=True,
            )
            cat_cols = st.columns(6, gap="small")
            categories_main = ["Áo", "Quần", "Váy", "Phụ kiện", "Giày dép", "Túi xách"]
            for col, cat_name in zip(cat_cols, categories_main):
                with col:
                    if st.button(cat_name, key=f"btn_home_cat_{cat_name}", width="stretch"):
                        st.session_state["cat_pill_selected"] = cat_name
                        st.session_state["search_filter"] = ""
                        st.session_state["explore_page"] = 1
                        st.session_state["active_nav_tab"] = "explore"
                        st.rerun()

        st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)

        # 3. CARD MỜI THIẾT LẬP SỞ THÍCH NẾU CHƯA CÓ DỮ LIỆU
        has_profile_pref = bool(style_preferences and style_preferences.get("favorite_styles"))
        if not has_profile_pref:
            with st.container(border=True):
                onb_col1, onb_col2 = st.columns([3.2, 1.2], vertical_alignment="center")
                with onb_col1:
                    st.markdown(
                        """
                        <div style="font-size:15px; font-weight:700; color:#1C1B1F;">Khám phá phong cách của bạn</div>
                        <div style="font-size:13px; color:#49454F;">
                            Chỉ mất 30 giây thiết lập phong cách, kích cỡ và ngân sách để nhận gợi ý trang phục chuẩn xác và ăn ý nhất.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with onb_col2:
                    if st.button("Thiết lập ngay", key="btn_onb_setup_home", type="primary", width="stretch"):
                        show_profile_dialog(username, full_name, user_email, is_verified, occupation, needs, usage_freq)

            st.markdown('<div style="height: 16px;"></div>', unsafe_allow_html=True)

        # 4. CÁC KHU VỰC GỢI Ý TRÊN TRANG CHỦ
        home_recs = _compute_user_recommendations(username, current_profile, num_recs=16)
        home_for_you = home_recs[:4] if home_recs else []
        home_popular = home_recs[4:8] if len(home_recs) >= 8 else (_get_cached_catalog(limit=8)[4:8])

        # Khu vực 1: Dành riêng cho bạn
        with st.container(key="fashion_home_recommendations"):
            render_recommendation_row(
                title="Dành riêng cho bạn",
                subtitle=(
                    f"Gợi ý phù hợp với nhu cầu {str(needs).replace(',', ' · ')} của bạn."
                    if needs else "Các thiết kế tuyển chọn phù hợp với phong cách của bạn."
                ),
                items=home_for_you,
                username=username,
                key_prefix="home_foryou",
                badge_type="for_you",
                default_reason="Gợi ý phù hợp hồ sơ phong cách",
                image_extractor=image_feature_extractor,
            )
            if st.button("Xem tất cả gợi ý cá nhân hóa", key="btn_home_more_recs"):
                st.session_state["active_nav_tab"] = "recs"
                st.rerun()

        st.markdown('<div style="height: 24px;"></div>', unsafe_allow_html=True)

        # Khu vực 2: Xu hướng nổi bật
        with st.container(key="fashion_home_trending"):
            render_recommendation_row(
                title="Xu hướng nổi bật",
                subtitle="Những thiết kế bán chạy nhất và nhận được nhiều lượt đánh giá cao trong tuần.",
                items=home_popular,
                username=username,
                key_prefix="home_trending",
                badge_type="popular",
                default_reason="Xu hướng thịnh hành tuần này",
                image_extractor=image_feature_extractor,
            )

        st.markdown('<div style="height: 24px;"></div>', unsafe_allow_html=True)

        # Khu vực 3: Sản phẩm mới lên kệ
        new_arrivals = _get_cached_catalog(limit=4)
        if new_arrivals:
            with st.container(key="fashion_home_new_arrivals"):
                render_recommendation_row(
                    title="Sản phẩm mới lên kệ",
                    subtitle="Vừa cập bến các cửa hàng đối tác với chất liệu và màu sắc theo xu hướng mới nhất.",
                    items=new_arrivals,
                    username=username,
                    key_prefix="home_new",
                    badge_type="new",
                    default_reason="Mẫu mới lên kệ",
                    image_extractor=image_feature_extractor,
                )

        # Khu vực 4: Đã xem gần đây
        recent_ids = st.session_state.get("recently_viewed_ids", [])
        if recent_ids:
            catalog_by_id = {int(p.get("article_id", 0)): p for p in _get_cached_catalog(limit=500)}
            target_rids = [int(rid) for rid in recent_ids[:8]]
            missing_rids = [rid for rid in target_rids if rid not in catalog_by_id and rid not in PRODUCT_MAP]
            fetched_map = {}
            if missing_rids:
                import db as _db_mod
                batch_fn = getattr(_db_mod, "get_products_by_ids", None)
                if callable(batch_fn):
                    b_res = batch_fn(missing_rids)
                    for p in (b_res.values() if isinstance(b_res, dict) else b_res):
                        fetched_map[int(p["article_id"])] = p
                else:
                    for rid in missing_rids:
                        p = get_product_by_id(rid)
                        if p:
                            fetched_map[int(rid)] = p
            recent_items = []
            for rid in target_rids:
                if rid in catalog_by_id:
                    recent_items.append(catalog_by_id[rid])
                elif rid in PRODUCT_MAP:
                    from dataclasses import asdict
                    recent_items.append(asdict(PRODUCT_MAP[rid]))
                elif rid in fetched_map:
                    recent_items.append(fetched_map[rid])
                if len(recent_items) >= 4:
                    break
            if recent_items:
                st.markdown('<div style="height: 24px;"></div>', unsafe_allow_html=True)
                with st.container(key="fashion_home_recent_viewed"):
                    render_recommendation_row(
                        title="Đã xem gần đây",
                        subtitle="Tiếp tục duyệt những thiết kế bạn vừa quan tâm trong phiên mua sắm này.",
                        items=recent_items,
                        username=username,
                        key_prefix="home_recent",
                        default_reason="Sản phẩm bạn vừa xem",
                        image_extractor=image_feature_extractor,
                    )

    # =========================================================
    # TAB 2: KHÁM PHÁ SẢN PHẨM (EXPLORE)
    # =========================================================
    elif active_page == "explore":
        render_section_header(
            title="Khám phá sản phẩm",
            subtitle="Tìm kiếm và lọc trang phục theo danh mục, mức giá, thương hiệu, màu sắc và phong cách.",
            kicker="BỘ SƯU TẬP TOÀN DIỆN",
        )

        def _reset_all_explore_filters(source_key: Optional[str] = None):
            """Callback đặt lại tất cả bộ lọc về mặc định (chạy qua on_click trước khi widget khởi tạo)."""
            if source_key and not st.session_state.get(source_key, False):
                return
            st.session_state["search_filter"] = ""
            st.session_state["cat_pill_selected"] = "Tất cả"
            st.session_state["price_range_filter"] = (0, 3000000)
            st.session_state["shop_filter"] = "Tất cả"
            st.session_state["style_filter"] = "Tất cả"
            st.session_state["sort_filter"] = "Mức độ phù hợp"
            st.session_state["explore_page"] = 1

        def _restart_explore_page():
            st.session_state["explore_page"] = 1

        with st.container(key="fashion_explore_filters"):
            # 1. Hàng tìm kiếm & Sắp xếp
            s_col, sort_col = st.columns([3.0, 1.2], gap="medium")
            with s_col:
                search_query = st.text_input(
                    "Tìm kiếm sản phẩm",
                    placeholder="Tìm áo sơ mi, quần âu, đầm, giày, túi xách, màu sắc...",
                    key="search_filter",
                    on_change=_restart_explore_page,
                )
            with sort_col:
                sort_mode = st.selectbox(
                    "Sắp xếp theo:",
                    options=["Mức độ phù hợp", "Mới nhất", "Giá: Thấp đến Cao", "Giá: Cao đến Thấp", "Bán chạy nhất"],
                    key="sort_filter",
                    on_change=_restart_explore_page,
                )

            # 2. Category Pills (MD3)
            st.session_state.setdefault("cat_pill_selected", "Tất cả")
            cat_options = ["Tất cả", "Thời trang nam", "Thời trang nữ", "Unisex", "Giày dép", "Túi xách", "Phụ kiện", "Áo", "Quần", "Váy"]
            with st.container(key="fashion_explore_categories"):
                cat_cols = st.columns(len(cat_options), gap="small")
                for column, category in zip(cat_cols, cat_options):
                    with column:
                        if st.button(
                            category,
                            key=f"btn_cat_pill_{category}",
                            type="primary" if st.session_state.get("cat_pill_selected", "Tất cả") == category else "secondary",
                            width="stretch",
                        ):
                            st.session_state["cat_pill_selected"] = category
                            st.session_state["explore_page"] = 1
                            st.rerun()

            # 3. Bộ lọc chi tiết trong Expander
            with st.expander("Bộ lọc nâng cao (Khoảng giá, Cửa hàng, Phong cách)", expanded=False):
                flt_c1, flt_c2, flt_c3 = st.columns([1.5, 1.2, 1.2])
                with flt_c1:
                    st.session_state.setdefault("price_range_filter", (0, 3000000))
                    price_tuple = st.slider(
                        "Khoảng giá (VNĐ):",
                        min_value=0,
                        max_value=3000000,
                        step=50000,
                        format="%d ₫",
                        key="price_range_filter",
                        on_change=_restart_explore_page,
                    )
                    min_p_val, max_p_val = price_tuple[0], price_tuple[1]

                with flt_c2:
                    shop_options = ["Tất cả", "Routine Official Store", "Zara Vietnam Authorized", "Coolmate Active Life", "IVY moda Elegance", "Studio Minimalist Boutique"]
                    sel_shop = st.selectbox(
                        "Cửa hàng / Thương hiệu:",
                        options=shop_options,
                        key="shop_filter",
                        on_change=_restart_explore_page,
                    )

                with flt_c3:
                    style_options = ["Tất cả", "Tối giản", "Công sở", "Streetwear", "Thể thao", "Thanh lịch"]
                    sel_style = st.selectbox(
                        "Phong cách thiết kế:",
                        options=style_options,
                        key="style_filter",
                        on_change=_restart_explore_page,
                    )

        category_sel = st.session_state["cat_pill_selected"]
        min_p_val, max_p_val = st.session_state.get("price_range_filter", (0, 3000000))
        sel_shop = st.session_state.get("shop_filter", "Tất cả")
        sel_style = st.session_state.get("style_filter", "Tất cả")

        # Chips hiển thị bộ lọc đang áp dụng
        has_active_filters = (
            category_sel != "Tất cả"
            or bool(str(search_query or "").strip())
            or min_p_val > 0
            or max_p_val < 3000000
            or sel_shop != "Tất cả"
            or sel_style != "Tất cả"
        )

        if has_active_filters:
            f_tags = []
            if str(search_query or "").strip():
                f_tags.append(f'Từ khóa: "{html.escape(search_query)}"')
            if category_sel != "Tất cả":
                f_tags.append(f'Danh mục: {html.escape(category_sel)}')
            if min_p_val > 0 or max_p_val < 3000000:
                f_tags.append(f'Giá: {min_p_val:,} ₫ – {max_p_val:,} ₫'.replace(",", "."))
            if sel_shop != "Tất cả":
                f_tags.append(f'Cửa hàng: {html.escape(sel_shop)}')
            if sel_style != "Tất cả":
                f_tags.append(f'Phong cách: {html.escape(sel_style)}')

            chip_col1, chip_col2 = st.columns([3.5, 0.9], vertical_alignment="center")
            with chip_col1:
                chips_html = " ".join([
                    f'<span style="background:#C3B1E1; color:#2D2438; border:1px solid #D5C7E8; padding:3px 10px; border-radius:9999px; font-size:12px; font-weight:700;">{t}</span>'
                    for t in f_tags
                ])
                st.markdown(f'<div style="display:flex; flex-wrap:wrap; gap:6px; align-items:center; margin-bottom:10px;">{chips_html}</div>', unsafe_allow_html=True)
            with chip_col2:
                st.button(
                    "Xóa tất cả",
                    icon=":material/close:",
                    key="btn_clear_active_filters",
                    on_click=_reset_all_explore_filters,
                    args=("btn_clear_active_filters",),
                    width="stretch",
                )

        # Thực thi lọc dữ liệu
        try:
            raw_catalog = _get_cached_catalog(limit=500)
            products = filter_products_smart(
                raw_catalog,
                category_pill=category_sel,
                search_query=search_query,
                min_price=min_p_val if min_p_val > 0 else None,
                max_price=max_p_val if max_p_val < 3000000 else None,
            )

            # Lọc thêm theo Shop & Phong cách nếu chọn
            if sel_shop != "Tất cả":
                products = [p for p in products if p.get("shop_name") == sel_shop]
            if sel_style != "Tất cả":
                products = [p for p in products if p.get("style_tag") == sel_style]

        except Exception:
            logger.exception("Lỗi tải danh sách sản phẩm")
            render_error_state(retry_key="retry_load_products")
            products = []

        # Sắp xếp
        if sort_mode == "Giá: Thấp đến Cao":
            products = sorted(products, key=lambda x: get_price_vnd_numeric(x.get("price", 0)))
        elif sort_mode == "Giá: Cao đến Thấp":
            products = sorted(products, key=lambda x: get_price_vnd_numeric(x.get("price", 0)), reverse=True)
        elif sort_mode == "Bán chạy nhất":
            products = sorted(products, key=lambda x: float(x.get("rating_count", 0)), reverse=True)

        if not products:
            render_empty_state(
                title="Không tìm thấy sản phẩm phù hợp",
                subtitle="Hãy thử từ khóa khác hoặc điều chỉnh các tiêu chí bộ lọc.",
                icon="search",
                action_label="Xóa bộ lọc",
                action_key="btn_reset_explore_empty",
                on_action=_reset_all_explore_filters,
            )
        else:
            total_items = len(products)
            per_page = int(st.session_state.get("products_per_page", 16))
            total_pages = max(1, (total_items + per_page - 1) // per_page)
            curr_page = min(max(1, int(st.session_state.get("explore_page", 1))), total_pages)

            render_safe_html(
                f"""
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px;">
                    <span style="font-size:14px; font-weight:700; color:#1C1B1F;">
                        Tìm thấy {total_items} sản phẩm phù hợp
                    </span>
                    <span style="font-size:12.5px; color:#49454F;">Trang {curr_page} / {total_pages}</span>
                </div>
                """
            )

            start_idx = (curr_page - 1) * per_page
            page_products = products[start_idx : start_idx + per_page]

            cols_per_row = 4
            with st.container(key=f"fashion_grid_explore_page_{curr_page}"):
                for i in range(0, len(page_products), cols_per_row):
                    row_items = page_products[i : i + cols_per_row]
                    grid_cols = st.columns(cols_per_row, gap="medium")
                    for idx_col, item in enumerate(row_items):
                        with grid_cols[idx_col]:
                            render_product_card(
                                item=item,
                                username=username,
                                key_prefix=f"exp_{curr_page}_{i + idx_col}",
                                image_extractor=image_feature_extractor,
                            )

            if total_pages > 1:
                st.markdown('<div style="height: 16px;"></div>', unsafe_allow_html=True)
                p_left, p_prev, p_info, p_next, p_right = st.columns([1.5, 1.0, 1.2, 1.0, 1.5], vertical_alignment="center")
                with p_prev:
                    if st.button("Trang trước", key="btn_exp_prev", disabled=(curr_page <= 1), width="stretch"):
                        st.session_state["explore_page"] = curr_page - 1
                        st.rerun()
                with p_info:
                    st.markdown(
                        f"<div style='text-align:center; font-size:13px; font-weight:600; color:#1C1B1F;'>Trang {curr_page} / {total_pages}</div>",
                        unsafe_allow_html=True,
                    )
                with p_next:
                    if st.button("Trang tiếp", key="btn_exp_next", disabled=(curr_page >= total_pages), width="stretch"):
                        st.session_state["explore_page"] = curr_page + 1
                        st.rerun()

    # =========================================================
    # TAB 3: GỢI Ý CHO BẠN (RECS)
    # =========================================================
    elif active_page == "recs":
        col_ctrl1, col_ctrl2 = st.columns([3, 1], vertical_alignment="center")
        with col_ctrl1:
            top_n_choice = st.selectbox(
                "Số sản phẩm gợi ý:",
                options=[4, 8, 12, 16, 20, 24],
                index=3,
                key="sel_top_n_val",
                help="Chọn số lượng sản phẩm bạn muốn hệ thống đề xuất.",
            )
        with col_ctrl2:
            st.write("")
            if st.button("Làm mới", icon=":material/refresh:", key="btn_refresh_recs", help="Làm mới danh sách gợi ý", width="stretch"):
                for k in list(st.session_state.keys()):
                    if k.startswith("user_rec_"):
                        del st.session_state[k]
                st.toast("Đã làm mới danh sách gợi ý!", icon=":material/refresh:")
                st.rerun()

        selected_model = "hybrid"

        rec_results = _compute_user_recommendations(username, current_profile, num_recs=top_n_choice, model=selected_model)

        if not rec_results:
            render_empty_state(
                title="Chưa có gợi ý phù hợp",
                subtitle="Bạn có thể cập nhật sở thích thời trang hoặc khám phá thêm sản phẩm.",
                icon="heart",
                action_label="Khám phá sản phẩm",
                action_key="btn_rec_empty_explore",
                on_action=lambda: st.session_state.update({"active_nav_tab": "explore"}),
            )
            if st.button("Tùy chỉnh phong cách", icon=":material/tune:", key="btn_edit_style_empty"):
                show_profile_dialog(
                    username, full_name, user_email, is_verified,
                    occupation, needs, usage_freq,
                )
        else:
            for_you_items = rec_results[:min(4, len(rec_results))]
            you_may_like_items = rec_results[4:min(8, len(rec_results))] if len(rec_results) >= 8 else rec_results[:min(4, len(rec_results))]
            style_match_items = rec_results[8:min(12, len(rec_results))] if len(rec_results) >= 12 else rec_results[:min(4, len(rec_results))]
            explore_more_items = rec_results[12:min(top_n_choice, len(rec_results))] if len(rec_results) >= 16 else (_get_cached_catalog(limit=16)[12:16])

            render_personalized_recommendations_view(
                username=username,
                full_name=full_name,
                occupation=occupation,
                needs=needs,
                usage_freq=usage_freq,
                style_preferences=style_preferences,
                for_you_items=for_you_items,
                you_may_like_items=you_may_like_items,
                style_match_items=style_match_items,
                explore_more_items=explore_more_items,
                image_extractor=image_feature_extractor,
                on_edit_profile=lambda: show_profile_dialog(
                    username,
                    full_name,
                    user_email,
                    is_verified,
                    occupation,
                    needs,
                    usage_freq,
                ),
                on_explore_all=lambda: st.session_state.update({"active_nav_tab": "explore"}),
            )

    # =========================================================
    # TAB 4: TÌM KIẾM BẰNG HÌNH ẢNH (VISUAL)
    # =========================================================
    elif active_page == "visual":
        render_section_header(
            title="Tìm kiếm bằng hình ảnh",
            subtitle="Tải lên ảnh trang phục để tìm những thiết kế tương đồng về kiểu dáng và màu sắc trong bộ sưu tập.",
            kicker="TÌM KIẾM BẰNG ẢNH",
        )

        visual_engine = _get_visual_search_engine()
        visual_ready = bool(getattr(visual_engine, "art_to_idx", {}))
        if not visual_ready:
            render_error_state(
                title="Bộ máy tìm kiếm hình ảnh đang khởi tạo",
                subtitle="Vui lòng nhấn Thử lại để nạp dữ liệu tìm kiếm hình ảnh.",
                retry_key="retry_visual_search",
            )

        vis_mode = st.radio(
            "Chọn phương thức cung cấp hình ảnh:",
            options=["Chọn hình ảnh (JPG / PNG)", "Chụp ảnh trực tiếp", "Chọn mẫu sản phẩm có sẵn"],
            horizontal=True,
            index=0,
        )

        query_pil = None
        query_label = ""
        exclude_art_id = None

        if vis_mode == "Chọn hình ảnh (JPG / PNG)":
            st.caption("**Mẹo:** Sử dụng hình ảnh trang phục rõ nét, trải phẳng hoặc chụp người mẫu để tìm kiếm kiểu dáng và màu sắc chính xác nhất (Tối đa 10MB).")
            up_file = st.file_uploader(
                "Chọn hình ảnh",
                type=["jpg", "jpeg", "png", "webp"],
                help="Tải lên ảnh trang phục bạn muốn tìm kiếm (JPG / PNG, tối đa 10MB)",
            )
            if up_file is not None:
                if up_file.size > 10 * 1024 * 1024:
                    st.error("Dung lượng ảnh vượt quá giới hạn 10MB.")
                else:
                    try:
                        from PIL import Image
                        query_pil = Image.open(up_file).convert("RGB")
                        query_label = up_file.name
                    except Exception:
                        logger.exception("Không thể mở ảnh người dùng tải lên")
                        st.error("Không thể mở ảnh này. Vui lòng chọn ảnh khác.")

        elif vis_mode == "Chụp ảnh trực tiếp":
            cam_file = st.camera_input("Chụp ảnh trang phục")
            if cam_file is not None:
                try:
                    from PIL import Image
                    query_pil = Image.open(cam_file).convert("RGB")
                    query_label = "Ảnh chụp từ camera"
                except Exception:
                    logger.exception("Không thể mở ảnh từ camera")
                    st.error("Không thể xử lý ảnh vừa chụp.")

        else:
            sample_ids = list(getattr(visual_engine, "art_to_idx", {}).keys())[:25] or [101001, 101002, 102001, 103001]
            selected_source_id = st.selectbox(
                "Chọn mẫu sản phẩm trong bộ sưu tập:",
                sample_ids,
                format_func=lambda x: f"Mã #{x} — {PRODUCT_MAP.get(x, {}).prod_name if x in PRODUCT_MAP else f'Sản phẩm #{x}'}",
            )
            if selected_source_id:
                if visual_engine and hasattr(visual_engine, "get_image_path"):
                    src_img_p = visual_engine.get_image_path(selected_source_id)
                    if src_img_p and os.path.exists(src_img_p):
                        from PIL import Image
                        query_pil = Image.open(src_img_p).convert("RGB")
                query_label = f"#{selected_source_id}"
                exclude_art_id = selected_source_id
                sample_pname = PRODUCT_MAP.get(selected_source_id, {}).prod_name if selected_source_id in PRODUCT_MAP else f"Sản phẩm #{selected_source_id}"
                st.markdown(
                    f"""
                    <div style="display:inline-flex; align-items:center; gap:10px; border:2px solid #C3B1E1; border-radius:12px; padding:6px 14px; background:#FAF8FD; margin-top:8px; margin-bottom:8px; box-shadow:0 2px 8px rgba(195,177,225,0.3);">
                        <span style="background:#C3B1E1; color:#2D2438; font-weight:700; font-size:11.5px; padding:3px 10px; border-radius:999px;">Đã chọn</span>
                        <span style="font-size:13px; font-weight:600; color:#2D2438;">Mã #{selected_source_id} — {html.escape(sample_pname)}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        if (query_pil is not None or exclude_art_id is not None) and visual_ready:
            st.markdown('<div style="height: 16px;"></div>', unsafe_allow_html=True)
            with st.container(key="fashion_split_visual_res"):
                q_col1, q_col2 = st.columns([1.0, 2.6], gap="large")
                with q_col1:
                    st.markdown(
                        '<div style="font-size:14px; font-weight:700; color:#2D2438; margin-bottom:8px;">Ảnh truy vấn</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"""
                        <div style="border:2px solid #C3B1E1; border-radius:16px; padding:12px; background:#FAF8FD; box-shadow:0 4px 16px rgba(195,177,225,0.35);">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                                <span style="background:#C3B1E1; color:#2D2438; font-weight:700; font-size:11.5px; padding:3px 10px; border-radius:999px;">Đã chọn</span>
                                <span style="font-size:12px; font-weight:600; color:#2D2438;">{html.escape(query_label)}</span>
                            </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if query_pil is not None:
                        st.image(query_pil, width="stretch")
                    else:
                        st.info("Đang tìm kiếm sản phẩm theo mẫu đã chọn.")
                    st.markdown("</div>", unsafe_allow_html=True)

                with q_col2:
                    render_section_header(
                        title="Sản phẩm có kiểu dáng tương tự",
                        subtitle="Các thiết kế tương đồng về kiểu dáng và màu sắc trong bộ sưu tập.",
                        kicker="GỢI Ý TƯƠNG ĐỒNG",
                    )

            vis_loading = st.empty()
            with vis_loading.container():
                render_skeleton_grid("Đang tìm kiếm sản phẩm có kiểu dáng tương tự...", count=4)

            try:
                if exclude_art_id is not None:
                    similar_items = visual_engine.get_similar_products(
                        exclude_art_id,
                        top_k=12,
                    )
                elif hasattr(visual_engine, "get_similar_products_by_image"):
                    similar_items = visual_engine.get_similar_products_by_image(
                        query_pil,
                        top_k=12,
                        exclude_article_id=exclude_art_id,
                    )
                else:
                    similar_items = []
            except Exception:
                logger.exception("Lỗi tìm kiếm sản phẩm bằng hình ảnh")
                similar_items = []
            finally:
                vis_loading.empty()

            if not similar_items:
                render_empty_state(
                    title="Không tìm thấy sản phẩm tương đồng",
                    subtitle="Hãy thử tải lên một hình ảnh trang phục rõ nét hơn.",
                    icon="camera",
                    action_label=None,
                )
            else:
                # Loại trừ trùng lặp và loại trừ chính sản phẩm đang truy vấn
                seen_arts = set()
                dedup_items = []
                for it in similar_items:
                    aid = int(it.get("article_id", 0))
                    if aid and aid != exclude_art_id and aid not in seen_arts:
                        seen_arts.add(aid)
                        dedup_items.append(it)

                show_sim_flag = st.session_state.get("show_similarity_badge", True)
                cols_per_row = 4
                with st.container(key="fashion_grid_visual_results"):
                    for v_i in range(0, min(len(dedup_items), 12), cols_per_row):
                        row_vis = dedup_items[v_i : v_i + cols_per_row]
                        cols_v = st.columns(cols_per_row, gap="medium")
                        for idx_v, v_item in enumerate(row_vis):
                            with cols_v[idx_v]:
                                sim_val = v_item.get("similarity") if show_sim_flag else None
                                render_product_card(
                                    item=v_item,
                                    username=username,
                                    key_prefix=f"vis_{v_i + idx_v}",
                                    similarity=sim_val,
                                    image_extractor=image_feature_extractor,
                                )

    # =========================================================
    # TAB 5: GIỎ HÀNG VÀ ĐẶT HÀNG (CART & CHECKOUT)
    # =========================================================
    elif active_page == "cart":
        render_section_header(
            title="Giỏ hàng & Thanh toán",
            subtitle="Kiểm tra các sản phẩm được nhóm theo cửa hàng và hoàn tất đặt hàng.",
            kicker="SHOPPING BAG & CHECKOUT",
        )

        cart_items = get_user_cart(username)
        if not cart_items:
            render_empty_state(
                title="Giỏ hàng của bạn đang trống",
                subtitle="Hãy khám phá bộ sưu tập hoặc xem gợi ý dành riêng cho bạn để chọn trang phục yêu thích.",
                icon="shopping_bag",
                action_label="Khám phá sản phẩm",
                action_key="btn_cart_empty_go_explore",
                on_action=lambda: st.session_state.update({"active_nav_tab": "explore"}),
            )
        else:
            # Nhóm sản phẩm trong giỏ theo Cửa hàng (Shop)
            items_by_shop: Dict[str, List[dict]] = {}
            for c_item in cart_items:
                art_id = int(c_item["article_id"])
                sp_info = PRODUCT_MAP.get(art_id)
                shop_n = sp_info.shop_name if sp_info else "Routine Official Store"
                items_by_shop.setdefault(shop_n, []).append(c_item)

            cart_main_col, cart_sum_col = st.columns([2.3, 1.1], gap="large")

            subtotal_vnd = 0
            with cart_main_col:
                for shop_name, shop_items in items_by_shop.items():
                    with st.container(border=True):
                        st.markdown(
                            f"""
                            <div style="font-size:14.5px; font-weight:700; color:#624781; border-bottom:1px solid #E8DFF2; padding-bottom:8px; margin-bottom:12px;">
                                {html.escape(shop_name)}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        for c_item in shop_items:
                            c_id = int(c_item["cart_id"])
                            art_id = int(c_item["article_id"])
                            qty = int(c_item.get("quantity", 1))
                            sp_meta = PRODUCT_MAP.get(art_id)
                            unit_vnd = sp_meta.price_vnd if sp_meta else get_price_vnd_numeric(c_item.get("price", 0.03))
                            line_vnd = unit_vnd * qty
                            subtotal_vnd += line_vnd

                            img_uri = resolve_cached_product_image_uri(
                                art_id,
                                image_extractor=image_feature_extractor,
                                explicit_path=str(sp_meta.image_url if sp_meta else (c_item.get("image_url") or "")),
                            )
                            thumb_html = (
                                f'<img src="{img_uri}" style="width:56px; height:74px; object-fit:cover; border-radius:10px; flex-shrink:0;" />'
                                if img_uri
                                else f'<div role="img" aria-label="Chưa có ảnh" style="width:56px; height:74px; border-radius:10px; background:#FAF8FD; border:1px solid #E8DFF2; display:flex; align-items:center; justify-content:center; color:#79747E; flex-shrink:0;">{icon_svg("image", size=20)}</div>'
                            )

                            p_name = sp_meta.prod_name if sp_meta else str(c_item.get("prod_name"))
                            variant_note = f"Màu: {sp_meta.colour_group_name} · Size: M" if sp_meta else "Size: M"

                            r1, r2, r3, r4, r5 = st.columns([2.5, 1.1, 1.2, 1.1, 0.5], vertical_alignment="center")
                            with r1:
                                st.markdown(
                                    f"""
                                    <div style="display:flex; align-items:center; gap:12px;">
                                        {thumb_html}
                                        <div>
                                            <div style="font-size:14px; font-weight:600; color:#2D2438;">{html.escape(p_name)}</div>
                                            <div style="font-size:12px; color:#49454F;">{html.escape(variant_note)} · #{art_id}</div>
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                            with r2:
                                st.markdown(
                                    f"<div style='font-size:13.5px; font-weight:600; color:#2D2438;'>{format_vnd_int(unit_vnd)}</div>",
                                    unsafe_allow_html=True,
                                )
                            with r3:
                                q_minus, q_val, q_plus = st.columns([1, 1, 1], gap="small", vertical_alignment="center")
                                with q_minus:
                                    if st.button("Giảm số lượng", icon=":material/remove:", key=f"cart_dec_{c_id}", help="Giảm số lượng", width="stretch"):
                                        _update_cart_quantity(username, c_id, qty - 1)
                                        st.rerun()
                                Dangerously_low_stock = False
                                with q_val:
                                    st.markdown(
                                        f"<div style='text-align:center; font-size:13.5px; font-weight:700;'>{qty}</div>",
                                        unsafe_allow_html=True,
                                    )
                                with q_plus:
                                    if st.button("Tăng số lượng", icon=":material/add:", key=f"cart_inc_{c_id}", help="Tăng số lượng", width="stretch"):
                                        _update_cart_quantity(username, c_id, qty + 1)
                                        st.rerun()
                            with r4:
                                st.markdown(
                                    f"<div style='font-size:14px; font-weight:700; color:#2D2438;'>{format_vnd_int(line_vnd)}</div>",
                                    unsafe_allow_html=True,
                                )
                            with r5:
                                if st.button("Xóa mục này", icon=":material/delete:", key=f"del_cart_{c_id}", help="Xóa mục này"):
                                    remove_from_cart(username, c_id)
                                    st.rerun()

                            st.markdown('<div style="height: 6px;"></div>', unsafe_allow_html=True)

            # Cột tóm tắt & Xác nhận thanh toán
            with cart_sum_col:
                shipping_fee = 0
                discount_voucher = 50000 if subtotal_vnd >= 500000 else 0
                final_total = max(0, subtotal_vnd + shipping_fee - discount_voucher)

                with st.container(border=True):
                    st.markdown(
                        f"""
                        <div style="font-size:16px; font-weight:700; color:#2D2438; margin-bottom:14px;">
                            Tóm tắt đơn hàng
                        </div>
                        <div style="display:flex; justify-content:space-between; font-size:13px; color:#49454F; margin-bottom:8px;">
                            <span>Tạm tính ({cart_count} sản phẩm)</span>
                            <span style="font-weight:600; color:#2D2438;">{format_vnd_int(subtotal_vnd)}</span>
                        </div>
                        <div style="display:flex; justify-content:space-between; font-size:13px; color:#49454F; margin-bottom:8px;">
                            <span>Phí vận chuyển</span>
                            <span style="font-weight:600; color:#146C2E;">Miễn phí</span>
                        </div>
                        <div style="display:flex; justify-content:space-between; font-size:13px; color:#49454F; margin-bottom:8px;">
                            <span>Giảm giá voucher sàn</span>
                            <span style="font-weight:600; color:#B3261E;">-{format_vnd_int(discount_voucher)}</span>
                        </div>
                        <div style="height:1px; background:#E8DFF2; margin:12px 0;"></div>
                        <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:16px;">
                            <span style="font-size:14px; font-weight:700; color:#2D2438;">Tổng thanh toán</span>
                            <span style="font-size:20px; font-weight:800; color:#2D2438;">{format_vnd_int(final_total)}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Form thông tin người nhận
                    st.markdown("##### Thông tin giao hàng:")
                    rec_name = st.text_input("Họ và tên người nhận:", value=full_name, key="chk_recipient_name")
                    rec_phone = st.text_input("Số điện thoại:", value="0912345678", key="chk_recipient_phone")
                    rec_address = st.text_input("Địa chỉ giao hàng:", value="Số 12, Phố Huế, Quận Hai Bà Trưng, Hà Nội", key="chk_recipient_address")
                    pay_method = st.radio(
                        "Phương thức thanh toán:",
                        options=["Thanh toán khi nhận hàng (COD)", "Chuyển khoản QR ngân hàng"],
                        index=0,
                        key="chk_payment_method",
                    )

                    st.caption("Đơn hàng sẽ được lưu trực tiếp vào CSDL và đồng bộ tới hệ thống quản trị WinForms.")

                    if st.button("Xác nhận đặt hàng", key="btn_confirm_checkout", type="primary", width="stretch"):
                        if not rec_name.strip() or not rec_phone.strip() or not rec_address.strip():
                            st.error("Vui lòng điền đầy đủ Họ tên, Số điện thoại và Địa chỉ giao hàng!")
                        else:
                            ok, msg, order_data = create_order_from_cart(
                                username=username,
                                recipient_name=rec_name.strip(),
                                recipient_phone=rec_phone.strip(),
                                shipping_address=rec_address.strip(),
                                payment_method=pay_method,
                                discount_vnd=discount_voucher,
                                shipping_fee_vnd=shipping_fee,
                            )
                            if ok and order_data:
                                st.balloons()
                                st.toast(f"Đặt hàng #{order_data['order_code']} thành công!", icon=":material/check_circle:")
                                st.session_state.pop("user_orders", None)
                                st.rerun()
                            else:
                                st.error(msg)

                    if st.button("Xem đơn hàng của tôi", key="btn_view_orders_from_cart", width="stretch"):
                        show_orders_dialog(username)

    # =========================================================
    # TAB: TRỢ LÝ PHONG CÁCH (ASSISTANT PAGE)
    # =========================================================
    elif active_page == "assistant":
        from ui.chat_panel import render_chat_page
        render_chat_page()

    render_footer()
