"""
Module dịch vụ quản lý và phân phối hình ảnh sản phẩm tập trung (CHICSTYLE Image Service).
Đảm bảo tính nhất quán trên mọi môi trường (Đơn server, Đa server, Web Streamlit, WinForms Admin và API).

Các chức năng cốt lõi:
1. Chuẩn hóa mã sản phẩm H&M (article_id) 10 chữ số bảo toàn số 0 đầu (ví dụ: '0108775015').
2. Ánh xạ đường dẫn vật lý độc lập với thư mục chạy lệnh (sử dụng BASE_DIR tuyệt đối).
3. Động cơ tạo và nén Thumbnail (320px) chất lượng cao, giảm 85-90% dung lượng truyền tải.
4. Phân giải URL cho Client Web và WinForms với Version Fingerprint (Cache-Control immutable).
5. Cơ chế Fallback an toàn cho sản phẩm thiếu ảnh trong dataset gốc H&M.
6. Chống Path Traversal và kiểm tra tính toàn vẹn file ảnh.
"""

from __future__ import annotations

import base64
import functools
import hashlib
import io
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlsplit

from PIL import Image

logger = logging.getLogger(__name__)

# Thư mục gốc tuyệt đối của dự án
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGES_DIR = BASE_DIR / "output" / "images"
DEFAULT_THUMBNAILS_DIR = BASE_DIR / "output" / "thumbnails"

# 6 ID sản phẩm thiếu ảnh đã xác nhận từ tập dữ liệu H&M gốc
KNOWN_DATASET_MISSING_IDS = frozenset({
    179208001,
    216961011,
    218354021,
    241602023,
    396135047,
    396135048,
})

WINDOWS_ABS_PATH_RE = re.compile(r"^[a-zA-Z]:[\\/]")


def normalize_article_id(article_id: Union[int, str]) -> str:
    """
    Chuẩn hóa mã sản phẩm thành chuỗi 10 chữ số theo đúng chuẩn bộ dữ liệu H&M.
    Bảo toàn các số 0 ở đầu (ví dụ: 108775015 -> '0108775015').
    """
    try:
        val_int = int(str(article_id).strip())
        if val_int < 0:
            return ""
        return f"{val_int:010d}"
    except (ValueError, TypeError):
        return ""


def get_image_subpath(article_id: Union[int, str]) -> str:
    """
    Trả về đường dẫn tương đối chuẩn của file ảnh theo cấu trúc H&M:
    {3 chữ số đầu}/{10 chữ số}.jpg (ví dụ: '010/0108775015.jpg').
    """
    clean_id = normalize_article_id(article_id)
    if not clean_id:
        return ""
    return f"{clean_id[:3]}/{clean_id}.jpg"


def get_image_root_dir() -> Path:
    """
    Xác định thư mục gốc chứa ảnh thực tế.
    Ưu tiên cấu hình biến môi trường PRODUCT_IMAGE_ROOT hoặc DATASET_ROOT.
    Fallback về thư mục 'output/images' hoặc 'images' trong repo.
    """
    env_img = os.getenv("PRODUCT_IMAGE_ROOT", "").strip()
    if env_img:
        p = Path(env_img).expanduser()
        if not p.is_absolute():
            p = (BASE_DIR / p).resolve()
        if p.is_dir():
            return p

    env_ds = os.getenv("DATASET_ROOT", "").strip()
    if env_ds:
        p_ds = Path(env_ds).expanduser()
        if not p_ds.is_absolute():
            p_ds = (BASE_DIR / p_ds).resolve()
        cand = p_ds / "output" / "images"
        if cand.is_dir():
            return cand

    if DEFAULT_IMAGES_DIR.is_dir():
        return DEFAULT_IMAGES_DIR

    cand_images = BASE_DIR / "images"
    if cand_images.is_dir():
        return cand_images

    return DEFAULT_IMAGES_DIR


def get_thumbnails_root_dir() -> Path:
    """Thư mục lưu trữ cache thumbnail 320px."""
    DEFAULT_THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_THUMBNAILS_DIR


def is_safe_relative_path(path_str: str) -> bool:
    """Kiểm tra đường dẫn không chứa ký tự path traversal ('..') hoặc ký tự bất hợp lệ."""
    norm = str(path_str or "").strip().replace("\\", "/")
    if not norm or norm.startswith(("/", "\\")):
        return False
    parts = Path(norm).parts
    return ".." not in parts and not any(":" in part for part in parts)


@functools.lru_cache(maxsize=2048)
def get_product_image_file_path(article_id: Union[int, str], explicit_path: str = "") -> Optional[str]:
    """
    Tìm đường dẫn file ảnh vật lý trên đĩa cứng cho sản phẩm:
    - Tìm theo đường dẫn tương đối chuẩn {3 chữ số đầu}/{10 chữ số}.jpg
    - Tìm theo explicit_path nếu an toàn và tồn tại
    - Đảm bảo kiểm tra file tồn tại và có dung lượng > 0
    """
    clean_id = normalize_article_id(article_id)
    if not clean_id:
        return None

    img_root = get_image_root_dir()

    # 1. Thử explicit_path nếu được truyền vào
    if explicit_path:
        clean_exp = explicit_path.strip().replace("\\", "/")
        if not clean_exp.startswith(("http://", "https://", "data:", "/api/")):
            if WINDOWS_ABS_PATH_RE.match(clean_exp) or clean_exp.startswith("/"):
                p_exp = Path(clean_exp).expanduser().resolve()
                if p_exp.is_file() and p_exp.stat().st_size > 0:
                    return str(p_exp)
            elif is_safe_relative_path(clean_exp):
                cand = (BASE_DIR / clean_exp).resolve()
                if cand.is_file() and cand.stat().st_size > 0:
                    return str(cand)
                cand_root = (img_root / clean_exp).resolve()
                if cand_root.is_file() and cand_root.stat().st_size > 0:
                    return str(cand_root)

    # 2. Thử đường dẫn chuẩn H&M: {root}/{subfolder}/{clean_id}.{ext}
    subfolder = clean_id[:3]
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        cand = img_root / subfolder / f"{clean_id}{ext}"
        if cand.is_file() and cand.stat().st_size > 0:
            return str(cand)
        cand_flat = img_root / f"{clean_id}{ext}"
        if cand_flat.is_file() and cand_flat.stat().st_size > 0:
            return str(cand_flat)

    return None


@functools.lru_cache(maxsize=1024)
def get_image_version_hash(file_path_str: str) -> str:
    """Tạo chuỗi version hash dựa trên thời gian sửa đổi và kích thước file để bật cache HTTP dài hạn."""
    try:
        p = Path(file_path_str)
        st = p.stat()
        raw_key = f"{int(st.st_mtime)}_{st.st_size}"
        return hashlib.md5(raw_key.encode("utf-8")).hexdigest()[:8]
    except Exception:
        return "1"


def get_or_create_thumbnail_path(
    article_id: Union[int, str],
    max_dim: int = 320,
    quality: int = 82,
) -> Optional[str]:
    """
    Lấy hoặc tạo mới file ảnh thu nhỏ (Thumbnail 320px) chất lượng cao:
    - Giảm kích thước file từ ~150KB xuống ~10-15KB.
    - Giữ tỷ lệ 3:4 chuẩn thời trang và màu sắc chính xác.
    - Được cache vĩnh viễn trên đĩa tại output/thumbnails/...
    """
    orig_path_str = get_product_image_file_path(article_id)
    if not orig_path_str:
        return None

    clean_id = normalize_article_id(article_id)
    if not clean_id:
        return None

    thumb_root = get_thumbnails_root_dir()
    subfolder = clean_id[:3]
    target_dir = thumb_root / subfolder
    target_dir.mkdir(parents=True, exist_ok=True)
    target_thumb = target_dir / f"{clean_id}_{max_dim}.jpg"

    orig_p = Path(orig_path_str)
    try:
        orig_mtime = orig_p.stat().st_mtime
    except OSError:
        return None

    # Nếu thumbnail đã tồn tại và mới hơn ảnh gốc -> dùng ngay
    if target_thumb.is_file() and target_thumb.stat().st_size > 0:
        if target_thumb.stat().st_mtime >= orig_mtime:
            return str(target_thumb)

    # Sinh thumbnail mới
    try:
        with Image.open(orig_p) as im:
            im_rgb = im.convert("RGB")
            im_rgb.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
            target_thumb_tmp = target_thumb.with_suffix(".tmp")
            im_rgb.save(target_thumb_tmp, format="JPEG", quality=quality, optimize=True)
            target_thumb_tmp.replace(target_thumb)
        return str(target_thumb)
    except Exception as exc:
        logger.warning("Không thể tạo thumbnail cho sản phẩm #%s: %s", clean_id, exc)
        return orig_path_str


@functools.lru_cache(maxsize=1024)
def get_cached_thumbnail_data_uri(article_id: Union[int, str]) -> Optional[str]:
    """
    Chuyển thumbnail sang chuỗi Data URI (data:image/jpeg;base64,...) được cache trong RAM.
    Kích thước chỉ ~10-14KB (nhỏ hơn 10 lần so với ảnh gốc), an toàn khi Streamlit không có public URL.
    """
    thumb_path = get_or_create_thumbnail_path(article_id, max_dim=300, quality=80)
    if not thumb_path or not os.path.isfile(thumb_path):
        return None
    try:
        with open(thumb_path, "rb") as f:
            data = f.read()
        if not data:
            return None
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception as exc:
        logger.warning("Lỗi khi đọc thumbnail data uri cho #%s: %s", article_id, exc)
        return None


def resolve_product_image_url(
    article_id: Union[int, str],
    explicit_path: str = "",
    use_thumbnail: bool = True,
    prefer_external: bool = True,
    public_base_url: Optional[str] = None,
) -> Optional[str]:
    """
    Phân giải URL tối ưu cho client hiển thị ảnh:
    - Nếu có PUBLIC_MEDIA_BASE_URL: trả về URL HTTP công khai trỏ về endpoint API phục vụ ảnh/thumbnail.
    - Nếu là URL bên ngoài (Unsplash): trả về trực tiếp.
    - Mặc định trả về route tương đối /api/products/{id}/thumbnail?v={hash}.
    - Nếu không tìm thấy ảnh: trả về None (kích hoạt fallback).
    """
    clean_id = normalize_article_id(article_id)
    raw_exp = str(explicit_path or "").strip()

    # 1. Nếu là link Unsplash hoặc link web bên ngoài
    if prefer_external and raw_exp.startswith(("http://", "https://")):
        parsed = urlsplit(raw_exp)
        if not any(lh in parsed.netloc for lh in ("127.0.0.1", "localhost")):
            return raw_exp

    # 2. Kiểm tra xem file ảnh có tồn tại trên đĩa không
    img_path = get_product_image_file_path(clean_id, explicit_path=raw_exp)
    if not img_path:
        return None

    v_hash = get_image_version_hash(img_path)
    endpoint_type = "thumbnail" if use_thumbnail else "image"
    relative_route = f"/api/products/{clean_id}/{endpoint_type}?v={v_hash}"

    # 3. Nếu có cấu hình URL công khai
    base = public_base_url if public_base_url is not None else os.getenv("PUBLIC_MEDIA_BASE_URL", "").strip().rstrip("/")
    if base:
        return f"{base}{relative_route}"

    return relative_route


def save_uploaded_product_image(
    article_id: Union[int, str],
    file_bytes: bytes,
    filename: str = "",
) -> Tuple[bool, str, str]:
    """
    Xử lý an toàn khi Admin upload ảnh sản phẩm mới:
    1. Kiểm tra dung lượng hợp lệ (tối đa 10MB).
    2. Giải mã bằng PIL kiểm tra file ảnh hợp lệ (chống upload mã độc).
    3. Lưu vào kho ảnh chung: output/images/{subfolder}/{clean_id}.jpg.
    4. Xóa cache thumbnail cũ và tự động sinh thumbnail mới.
    5. Trả về: (thành công: bool, thông báo: str, đường dẫn lưu: str).
    """
    clean_id = normalize_article_id(article_id)
    if not clean_id:
        return False, "Mã sản phẩm không hợp lệ.", ""

    if not file_bytes or len(file_bytes) > 10 * 1024 * 1024:
        return False, "Kích thước file ảnh không hợp lệ (tối đa 10MB).", ""

    try:
        with Image.open(io.BytesIO(file_bytes)) as pil_img:
            im_format = pil_img.format
            if im_format not in ("JPEG", "JPG", "PNG", "WEBP"):
                return False, f"Định dạng ảnh không được hỗ trợ: {im_format}. Chỉ chấp nhận JPEG, PNG, WEBP.", ""
            im_rgb = pil_img.convert("RGB")
    except Exception as exc:
        return False, f"Không thể giải mã file ảnh: {exc}", ""

    img_root = get_image_root_dir()
    subfolder = clean_id[:3]
    target_dir = img_root / subfolder
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / f"{clean_id}.jpg"

    try:
        # Ghi đè file ảnh an toàn qua file tạm
        tmp_file = target_file.with_suffix(".upload.tmp")
        im_rgb.save(tmp_file, format="JPEG", quality=90, optimize=True)
        tmp_file.replace(target_file)

        # Xóa cache
        get_product_image_file_path.cache_clear()
        get_image_version_hash.cache_clear()
        get_cached_thumbnail_data_uri.cache_clear()

        # Xóa các thumbnail cũ của sản phẩm này
        thumb_root = get_thumbnails_root_dir()
        thumb_sub = thumb_root / subfolder
        if thumb_sub.is_dir():
            for old_thumb in thumb_sub.glob(f"{clean_id}_*.jpg"):
                try:
                    old_thumb.unlink(missing_ok=True)
                except OSError:
                    pass

        # Sinh sẵn thumbnail mới ngay lập tức
        get_or_create_thumbnail_path(clean_id, max_dim=320, quality=82)

        rel_path = f"output/images/{subfolder}/{clean_id}.jpg"
        return True, "Cập nhật ảnh sản phẩm thành công.", rel_path
    except Exception as exc:
        logger.exception("Lỗi khi lưu ảnh upload cho #%s", clean_id)
        return False, f"Lỗi hệ thống khi lưu ảnh: {exc}", ""


def clear_all_image_caches():
    """Xóa toàn bộ các bộ nhớ đệm phân giải ảnh."""
    get_product_image_file_path.cache_clear()
    get_image_version_hash.cache_clear()
    get_cached_thumbnail_data_uri.cache_clear()
