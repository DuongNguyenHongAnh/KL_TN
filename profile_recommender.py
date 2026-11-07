"""
Module Gợi ý dựa trên Hồ sơ Người dùng (Profile-based Recommender)
Ánh xạ thông tin cá nhân hóa (Nghề nghiệp, Nhu cầu sử dụng, Tần suất)
vào các danh mục và thuộc tính sản phẩm thời trang H&M.

Thuật toán:
User Profile (Occupation, Usage Need, Frequency)
        ↓
Configurable Category & Keyword Alignment
        ↓
Profile Match Score S_profile(u, i) in [0.0, 1.0]
        ↓
Candidate Filtering & Ranking
"""

import re
from typing import Dict, List, Optional

# Bảng ánh xạ cấu hình từ Nhu cầu sử dụng -> Nhóm hàng & từ khóa ưu tiên
NEED_MAPPING = {
    "Đi làm / Công sở": {
        "groups": ["Garment Upper body", "Garment Lower body", "Shoes", "Accessories"],
        "keywords": ["shirt", "blazer", "trousers", "suit", "blouse", "office", "formal", "dress", "classic", "tailored"],
        "weight_boost": 1.25
    },
    "Đi học": {
        "groups": ["Garment Upper body", "Garment Lower body", "Accessories"],
        "keywords": ["t-shirt", "tee", "hoodie", "sweater", "jeans", "denim", "backpack", "casual", "sneakers", "cotton"],
        "weight_boost": 1.20
    },
    "Dạo phố / Đời thường": {
        "groups": ["Garment Upper body", "Garment Lower body", "Accessories"],
        "keywords": ["casual", "dress", "skirt", "top", "shorts", "t-shirt", "sandals", "everyday", "relaxed"],
        "weight_boost": 1.15
    },
    "Dự tiệc / Sự kiện": {
        "groups": ["Garment Upper body", "Garment Lower body", "Accessories"],
        "keywords": ["dress", "evening", "party", "silk", "glitter", "heels", "necklace", "earrings", "suit", "elegant"],
        "weight_boost": 1.30
    },
    "Thể thao / Vận động": {
        "groups": ["Garment Upper body", "Garment Lower body", "Shoes"],
        "keywords": ["sports", "active", "joggers", "leggings", "sneakers", "running", "training", "shorts", "stretch"],
        "weight_boost": 1.35
    },
    "Du lịch": {
        "groups": ["Garment Upper body", "Garment Lower body", "Swimwear", "Accessories"],
        "keywords": ["sunglasses", "hat", "swimwear", "bikini", "shorts", "sundress", "beach", "linen", "vacation"],
        "weight_boost": 1.25
    }
}

# Bảng ánh xạ theo Nghề nghiệp -> Phong cách thời trang đặc trưng
OCCUPATION_MAPPING = {
    "Học sinh / Sinh viên": {
        "favored_keywords": ["t-shirt", "jeans", "hoodie", "sneakers", "denim", "casual", "young"],
        "price_preference": "budget"
    },
    "Nhân viên văn phòng": {
        "favored_keywords": ["shirt", "trousers", "blazer", "blouse", "skirt", "suit", "formal", "office"],
        "price_preference": "mid_high"
    },
    "Kinh doanh / Bán hàng": {
        "favored_keywords": ["blazer", "dress", "shirt", "watch", "smart", "suit", "shoes"],
        "price_preference": "high"
    },
    "Giáo viên": {
        "favored_keywords": ["shirt", "skirt", "trousers", "blouse", "dress", "cardigan", "modest"],
        "price_preference": "mid"
    },
    "Kỹ thuật / IT": {
        "favored_keywords": ["t-shirt", "polo", "hoodie", "jeans", "shorts", "comfy", "sneakers"],
        "price_preference": "mid"
    },
    "Khác": {
        "favored_keywords": ["casual", "fashion", "top", "pants"],
        "price_preference": "all"
    }
}

# Bảng ánh xạ theo Tần suất sử dụng sản phẩm -> Phân loại trang phục phù hợp
FREQUENCY_MAPPING = {
    "Hàng ngày": {
        "keywords": ["basic", "cotton", "casual", "t-shirt", "jeans", "everyday", "comfy", "denim", "essential"],
        "bonus": 0.10,
        "description": "Ưu tiên trang phục cơ bản, chất liệu thoáng mát, bền đẹp dùng thường nhật"
    },
    "Vài lần/tuần": {
        "keywords": ["polo", "shirt", "blouse", "chinos", "skirt", "casual", "smart", "trousers"],
        "bonus": 0.10,
        "description": "Ưu tiên trang phục đa năng, dễ phối đồ linh hoạt trong tuần"
    },
    "Vài lần/tháng": {
        "keywords": ["smart", "dress", "blazer", "trousers", "skirt", "elegant", "fashion", "jacket"],
        "bonus": 0.10,
        "description": "Ưu tiên trang phục chỉn chu cho các cuộc gặp gỡ, dạo phố cuối tuần"
    },
    "Theo mùa / Dịp đặc biệt": {
        "keywords": ["party", "evening", "suit", "blazer", "dress", "silk", "glitter", "heels", "jacket", "coat", "swimwear", "bikini"],
        "bonus": 0.12,
        "description": "Ưu tiên trang phục sự kiện, dự tiệc, đi du lịch hoặc thời trang theo mùa"
    }
}

class ProfileRecommender:
    def __init__(self):
        self.need_rules = NEED_MAPPING
        self.occ_rules = OCCUPATION_MAPPING
        self.freq_rules = FREQUENCY_MAPPING

    def get_rule_explanations(self) -> dict:
        """Trả về cấu trúc quy tắc ánh xạ và trọng số để phục vụ giải thích và báo cáo."""
        return {
            "need_rules": self.need_rules,
            "occupation_rules": self.occ_rules,
            "frequency_rules": self.freq_rules,
            "weights": {
                "base_score": 0.40,
                "need_group_bonus": 0.25,
                "need_keyword_bonus": 0.15,
                "occupation_keyword_bonus": 0.15,
                "frequency_keyword_bonus": 0.10
            }
        }

    def calculate_match_score(self, product: dict, profile: dict) -> float:
        """
        Tính điểm phù hợp giữa sản phẩm thời trang và hồ sơ người dùng.
        Sử dụng thực tế cả ba trường: Nghề nghiệp, Nhu cầu sử dụng, Tần suất sử dụng.
        Trả về điểm trong khoảng [0.0, 1.0].
        """
        if not profile:
            return 0.5

        score = 0.4 # Điểm cơ sở

        need = profile.get("usage_need") or profile.get("needs") or ""
        selected_needs = {value.strip() for value in str(need).split(",") if value.strip()}
        occupation = profile.get("occupation") or ""
        freq = profile.get("usage_frequency") or profile.get("usage_freq") or ""

        group_name = product.get("_group") or str(product.get("product_group_name") or "")
        full_text = product.get("_search_text")
        if full_text is None:
            prod_name = str(product.get("prod_name") or "").lower()
            desc = str(product.get("detail_desc") or "").lower()
            full_text = f"{prod_name} {desc}"

        # 1. Khớp theo Nhu cầu sử dụng (Usage Need)
        need_bonus = 0.0
        for selected_need in selected_needs:
            rule = self.need_rules.get(selected_need)
            if rule is None:
                continue
            rule_bonus = 0.0
            # Thưởng nếu thuộc đúng nhóm hàng
            if group_name in rule["groups"]:
                rule_bonus += 0.25
            # Thưởng nếu xuất hiện từ khóa phù hợp
            for kw in rule["keywords"]:
                if kw in full_text:
                    rule_bonus += 0.15
                    break
            # Nhiều nhu cầu: lấy mức phù hợp cao nhất, giữ nguyên thang điểm.
            need_bonus = max(need_bonus, rule_bonus)
        score += need_bonus

        # 2. Khớp theo Nghề nghiệp (Occupation)
        if occupation in self.occ_rules:
            occ_rule = self.occ_rules[occupation]
            for kw in occ_rule["favored_keywords"]:
                if kw in full_text:
                    score += 0.15
                    break

        # 3. Điều chỉnh theo Tần suất sử dụng sản phẩm (Usage Frequency)
        if freq in self.freq_rules:
            freq_rule = self.freq_rules[freq]
            if any(k in full_text for k in freq_rule["keywords"]):
                score += freq_rule["bonus"]

        return min(1.0, round(score, 4))

    def filter_and_rank_candidates(self, candidate_products: List[dict], profile: dict, top_n: int = 10) -> List[dict]:
        """
        Lọc và xếp hạng danh sách ứng viên sản phẩm dựa trên hồ sơ người dùng.
        """
        scored_products = []
        for prod in candidate_products:
            m_score = self.calculate_match_score(prod, profile)
            item_copy = dict(prod)
            item_copy["profile_match_score"] = m_score
            scored_products.append(item_copy)

        # Sắp xếp giảm dần theo điểm phù hợp
        scored_products.sort(key=lambda x: x["profile_match_score"], reverse=True)
        for rank, p in enumerate(scored_products[:top_n], 1):
            p["rank"] = rank
        return scored_products[:top_n]

# Singleton instance
profile_recommender = ProfileRecommender()

if __name__ == "__main__":
    rec = ProfileRecommender()
    sample_prod = {
        "article_id": 108775015,
        "prod_name": "Strap top office shirt",
        "product_group_name": "Garment Upper body",
        "detail_desc": "Jersey strap top classic tailored formal wear"
    }
    sample_prof = {
        "occupation": "Nhân viên văn phòng",
        "usage_need": "Đi làm / Công sở",
        "usage_frequency": "Hàng ngày"
    }
    s = rec.calculate_match_score(sample_prod, sample_prof)
    print(f"Test Profile Match Score: {s:.4f}")
