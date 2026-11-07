"""Shared choices and validation for optional fashion profile details."""

import math
from collections.abc import Mapping


FASHION_GROUP_OPTIONS = ("Nam", "Nữ", "Unisex")
STYLE_OPTIONS = (
    "Basic", "Tối giản", "Thanh lịch", "Công sở", "Thể thao", "Streetwear",
    "Vintage", "Nữ tính", "Cá tính",
)
COLOR_OPTIONS = (
    "Đen", "Trắng", "Xám", "Be", "Nâu", "Xanh dương", "Xanh lá",
    "Đỏ", "Hồng", "Cam", "Vàng", "Tím",
)
MATERIAL_REQUIREMENT_OPTIONS = ("Thoáng mát", "Co giãn", "Mềm", "Ít nhăn")
MATERIAL_OPTIONS = (
    "Cotton", "Linen", "Len", "Lụa", "Denim", "Polyester", "Nylon",
    "Da", "Da tổng hợp", "Ren", "Viscose",
)
PRODUCT_TYPE_OPTIONS = ("Áo", "Quần", "Váy", "Đầm", "Giày", "Phụ kiện")
FIT_OPTIONS = ("", "Ôm", "Vừa người", "Rộng", "Oversize")
SIZE_SYSTEM_OPTIONS = ("", "Quốc tế (XS–XXL)", "EU", "UK", "US", "Việt Nam", "Khác")
SIZE_CATEGORY_OPTIONS = ("tops", "bottoms", "shoes")
HEIGHT_RANGE_CM = (0.1, 300.0)
MEASUREMENT_FIELDS = {
    "chest_cm": ("Vòng ngực", 0.1, 350.0),
    "shoulder_cm": ("Rộng vai", 0.1, 150.0),
    "waist_cm": ("Vòng eo", 0.1, 350.0),
    "hips_cm": ("Vòng mông", 0.1, 350.0),
    "inseam_cm": ("Chiều dài chân trong", 0.1, 150.0),
    "foot_length_cm": ("Chiều dài bàn chân", 0.1, 60.0),
    "foot_width_cm": ("Độ rộng bàn chân", 0.1, 40.0),
}


def _choices(value, options, label):
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label}: hãy chọn các lựa chọn trong danh sách.")
    result = []
    for item in value:
        if not isinstance(item, str) or item not in options:
            raise ValueError(f"{label}: có lựa chọn không hợp lệ.")
        if item not in result:
            result.append(item)
    return result


def _optional_number(value, label, minimum=0.0, maximum=None, integer=False):
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        raise ValueError(f"{label}: vui lòng nhập một số hợp lệ.") from None
    if isinstance(value, bool) or not math.isfinite(number):
        raise ValueError(f"{label}: vui lòng nhập một số hữu hạn hợp lệ.")
    if number < minimum or (maximum is not None and number > maximum):
        if maximum is not None:
            raise ValueError(f"{label}: nhập giá trị từ {minimum:g} đến {maximum:g} cm.")
        raise ValueError(f"{label}: không được nhỏ hơn {minimum:g}.")
    if integer and not number.is_integer():
        raise ValueError(f"{label}: vui lòng nhập số tiền nguyên đồng.")
    return int(number) if integer else number


def _text(value, label, max_length=120):
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{label}: vui lòng nhập văn bản.")
    value = value.strip()
    if len(value) > max_length:
        raise ValueError(f"{label}: tối đa {max_length} ký tự.")
    return value


def normalize_style_preferences(value=None):
    """Return independent, complete defaults or raise ValueError for invalid input.

    All fields are optional. Preserve both sizing methods so switching methods
    does not erase a previously entered size or measurement.
    """
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise ValueError("Thông tin phong cách phải là một bộ thông tin hợp lệ.")

    result = {}
    for key, options, label in (
        ("fashion_groups", FASHION_GROUP_OPTIONS, "Nhóm thời trang"),
        ("favorite_styles", STYLE_OPTIONS, "Phong cách yêu thích"),
        ("favorite_colors", COLOR_OPTIONS, "Màu sắc yêu thích"),
        ("avoided_colors", COLOR_OPTIONS, "Màu sắc muốn tránh"),
        ("material_preferences", MATERIAL_REQUIREMENT_OPTIONS, "Yêu cầu chất liệu"),
        ("avoided_materials", MATERIAL_OPTIONS, "Chất liệu muốn tránh"),
        ("product_types", PRODUCT_TYPE_OPTIONS, "Loại sản phẩm quan tâm"),
    ):
        result[key] = _choices(value.get(key), options, label)

    if set(result["favorite_colors"]) & set(result["avoided_colors"]):
        raise ValueError("Một màu không thể vừa là màu yêu thích vừa là màu muốn tránh.")

    result["height_cm"] = _optional_number(
        value.get("height_cm"), "Chiều cao", *HEIGHT_RANGE_CM,
    )
    mode = value.get("size_input_mode", "usual_sizes")
    if mode not in ("usual_sizes", "measurements"):
        raise ValueError("Cách nhập kích thước không hợp lệ.")
    result["size_input_mode"] = mode

    sizes = value.get("usual_sizes") or {}
    if not isinstance(sizes, Mapping):
        raise ValueError("Size thường mặc không hợp lệ.")
    result["usual_sizes"] = {}
    for category in SIZE_CATEGORY_OPTIONS:
        item = sizes.get(category) or {}
        if not isinstance(item, Mapping):
            raise ValueError("Size thường mặc theo danh mục không hợp lệ.")
        system = _text(item.get("system"), "Hệ size")
        if system not in SIZE_SYSTEM_OPTIONS:
            raise ValueError("Hệ size không hợp lệ.")
        result["usual_sizes"][category] = {
            "size": _text(item.get("size"), "Size", max_length=40),
            "brand": _text(item.get("brand"), "Hãng"),
            "system": system,
        }

    measurements = value.get("measurements") or {}
    if not isinstance(measurements, Mapping):
        raise ValueError("Số đo cơ thể không hợp lệ.")
    result["measurements"] = {
        key: _optional_number(measurements.get(key), label, minimum, maximum)
        for key, (label, minimum, maximum) in MEASUREMENT_FIELDS.items()
    }

    fit = value.get("fit_preference") or ""
    if fit not in FIT_OPTIONS:
        raise ValueError("Độ vừa mong muốn không hợp lệ.")
    result["fit_preference"] = fit
    for key, label in (
        ("budget_min_vnd", "Ngân sách tối thiểu"),
        ("budget_max_vnd", "Ngân sách tối đa"),
    ):
        result[key] = _optional_number(value.get(key), label, integer=True)
    lower, upper = result["budget_min_vnd"], result["budget_max_vnd"]
    if lower is not None and upper is not None and lower > upper:
        raise ValueError("Ngân sách tối thiểu không được lớn hơn ngân sách tối đa.")
    return result


def validate_style_preferences(value):
    """Return a user-facing validation error, or an empty list for valid input."""
    try:
        normalize_style_preferences(value)
    except ValueError as error:
        return [str(error)]
    return []
