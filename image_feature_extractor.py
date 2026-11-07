"""
Module Trích xuất Đặc trưng Ảnh Sản phẩm bằng Transfer Learning (Computer Vision)
Sử dụng mạng nơ-ron tích chập Pretrained ResNet50 để sinh vector đặc trưng 2048 chiều.
Tính toán độ đo tương đồng Cosine Similarity phục vụ Gợi ý sản phẩm tương tự về hình ảnh.

Quy trình xử lý (Feature Extraction Pipeline):
Product Image
      ↓
Preprocessing (Resize 224x224, RGB, Normalize ImageNet)
      ↓
Pretrained ResNet50 Backbone (Global Average Pooling)
      ↓
2048-dimensional Feature Vector (L2 Normalized)
      ↓
Cosine Similarity Search -> Top-N Similar Products
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import glob
import shutil
import logging
import sqlite3
import functools
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = Path(BASE_DIR).resolve()
MODELS_CV_DIR = os.path.join(BASE_DIR, "models", "computer_vision")
os.makedirs(MODELS_CV_DIR, exist_ok=True)

CACHE_FEATURE_PATH = os.path.join(MODELS_CV_DIR, "image_features_resnet50.npz")
DB_PATH = os.path.join(BASE_DIR, "users.db")


@functools.lru_cache(maxsize=8)
def _discover_fallback_dataset_roots(base_dir_str: str) -> Tuple[Path, ...]:
    """Tự động dò tìm thư mục dữ liệu lân cận (nếu có) mà không hardcode đường dẫn máy cá nhân."""
    base = Path(base_dir_str).resolve()
    discovered: List[Path] = [base]
    for parent in list(base.parents)[:5]:
        for folder_name in ("XLDATA (2)", "XLDATA", "XLDATA (3)"):
            cand = parent / folder_name
            if (cand / "XLDATA" / "output" / "images").is_dir():
                discovered.append(cand / "XLDATA")
            if (cand / "output" / "images").is_dir():
                discovered.append(cand)
    return tuple(dict.fromkeys(discovered))


def _resolve_config_path(raw_path: str, base_dir: Path = BASE_PATH) -> Path:
    """Chuẩn hóa đường dẫn cấu hình (hỗ trợ cả đường dẫn tương đối theo BASE_DIR và đường dẫn tuyệt đối)."""
    cleaned = str(raw_path or "").strip().replace("\\", "/")
    p = Path(cleaned).expanduser()
    if not p.is_absolute():
        return (base_dir / p).resolve()
    return p


def default_image_root() -> str:
    configured_images = os.getenv("PRODUCT_IMAGE_ROOT")
    if configured_images:
        cand = _resolve_config_path(configured_images)
        if cand.is_dir():
            return str(cand)
    dataset_env = os.getenv("DATASET_ROOT")
    if dataset_env:
        ds_cand = _resolve_config_path(dataset_env) / "output" / "images"
        if ds_cand.is_dir():
            return str(ds_cand)
    local_repo_images = BASE_PATH / "output" / "images"
    if local_repo_images.is_dir():
        return str(local_repo_images)
    for ds_root in _discover_fallback_dataset_roots(str(BASE_PATH)):
        cand = ds_root / "output" / "images"
        if cand.is_dir():
            return str(cand)
    return str(local_repo_images)


def _is_safe_relative_subpath(raw_source: str) -> bool:
    """Chặn path traversal (..) hoặc ký tự bất hợp lệ trong đường dẫn tương đối."""
    norm = str(raw_source or "").strip().replace("\\", "/")
    if not norm:
        return False
    parts = Path(norm).parts
    return ".." not in parts


@functools.lru_cache(maxsize=4096)
def _resolve_product_image_cached(
    article_id: int,
    source: str,
    image_root_str: str,
    env_img_root: str,
    env_ds_root: str,
    local_only: bool,
) -> Optional[str]:
    parsed = urlsplit(source)
    if parsed.scheme.lower() in ("http", "https") and parsed.netloc:
        return None if local_only else source

    dataset_roots: List[Path] = []
    if env_ds_root:
        ds_p = _resolve_config_path(env_ds_root)
        if ds_p.is_dir():
            dataset_roots.append(ds_p)
    dataset_roots.append(BASE_PATH)
    dataset_roots.extend(_discover_fallback_dataset_roots(str(BASE_PATH)))
    dataset_roots = list(dict.fromkeys(dataset_roots))

    image_roots: List[Path] = []
    if image_root_str:
        ir_p = _resolve_config_path(image_root_str)
        if ir_p.is_dir():
            image_roots.append(ir_p)
    if env_img_root:
        eir_p = _resolve_config_path(env_img_root)
        if eir_p.is_dir():
            image_roots.append(eir_p)
    image_roots.append(BASE_PATH / "output" / "images")
    image_roots.append(BASE_PATH / "images")
    for root in dataset_roots:
        image_roots.append(root / "output" / "images")
        image_roots.append(root / "images")
    image_roots = [r for r in dict.fromkeys(image_roots) if r.is_dir()]

    candidates: List[Path] = []
    if source and not source.startswith(("/api/", "data:")):
        clean_src = source.replace("\\", "/")
        if _is_safe_relative_subpath(clean_src):
            stored = Path(clean_src).expanduser()
            matches_article = not stored.stem.isdecimal() or int(stored.stem) == article_id
            if matches_article:
                if stored.is_absolute():
                    candidates.append(stored)
                else:
                    candidates.extend(root / stored for root in dataset_roots)
                    candidates.extend(root / stored for root in image_roots)
                    if "images" in stored.parts:
                        relative = Path(*stored.parts[stored.parts.index("images") + 1:])
                        candidates.extend(root / relative for root in image_roots)

    article_name = f"{article_id:010d}"
    subfolder = article_name[:3]
    for root in image_roots:
        for name in (article_name, str(article_id)) if article_name != str(article_id) else (article_name,):
            for extension in (".jpg", ".jpeg", ".png", ".webp"):
                candidates.append(root / subfolder / f"{name}{extension}")
                candidates.append(root / f"{name}{extension}")

    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        try:
            if candidate.is_file() and candidate.stat().st_size > 0:
                return str(candidate.resolve())
        except OSError:
            continue
    return None


def clear_image_resolution_cache():
    """Xóa cache đường dẫn ảnh khi thay đổi cấu hình thư mục ảnh."""
    _resolve_product_image_cached.cache_clear()
    _discover_fallback_dataset_roots.cache_clear()


def resolve_product_image(product, *, image_root=None, local_only=False):
    """Tìm đúng ảnh theo mã sản phẩm và đường dẫn đã lưu trong danh mục."""
    if isinstance(product, dict):
        try:
            article_id = int(product["article_id"])
        except (KeyError, TypeError, ValueError):
            return None
        source = str(product.get("image_url") or "").strip()
    else:
        try:
            article_id = int(product)
        except (TypeError, ValueError):
            return None
        source = ""
    if article_id <= 0:
        return None
    env_img_root = str(os.getenv("PRODUCT_IMAGE_ROOT") or "").strip()
    env_ds_root = str(os.getenv("DATASET_ROOT") or "").strip()
    img_root_str = str(image_root or "").strip()

    res = _resolve_product_image_cached(
        article_id,
        source,
        img_root_str,
        env_img_root,
        env_ds_root,
        bool(local_only),
    )
    if res and not res.startswith(("http://", "https://")) and not os.path.isfile(res):
        _resolve_product_image_cached.cache_clear()
        res = _resolve_product_image_cached(
            article_id,
            source,
            img_root_str,
            env_img_root,
            env_ds_root,
            bool(local_only),
        )
    return res


DEFAULT_IMAGE_ROOT = default_image_root()

class ImageFeatureExtractor:
    def __init__(self, image_root: Optional[str] = None):
        self.image_root = image_root
        self.device = "cpu"
        self.model = None
        self.transform = None
        self.article_ids: np.ndarray = np.array([])
        self.features: np.ndarray = np.zeros((0, 2048), dtype=np.float32)
        self.art_to_idx: Dict[int, int] = {}
        self._meta_cache: Dict[int, dict] = {}
        self.load_cached_features()
        self._load_meta_cache()

    def _load_meta_cache(self):
        """Tải trước toàn bộ metadata sản phẩm vào RAM để tránh truy vấn SQLite lặp lại trong vòng lặp."""
        if self._meta_cache:
            return
        try:
            conn = sqlite3.connect(DB_PATH, timeout=10.0)
            rows = conn.execute("SELECT article_id, prod_name, product_group_name, price, image_url FROM products").fetchall()
            conn.close()
            for r in rows:
                self._meta_cache[int(r[0])] = {
                    "prod_name": r[1] or f"H&M Article {r[0]}",
                    "product_group_name": r[2] or "Thời trang",
                    "price": float(r[3]) if r[3] else 0.03,
                    "image_url": r[4] or ""
                }
        except Exception:
            pass

    def _init_transforms(self):
        """Khởi tạo pipeline tiền xử lý ảnh chuẩn ImageNet."""
        try:
            import torchvision.transforms as transforms
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        except Exception:
            self.transform = None

    def _load_model(self):
        """Tải mô hình ResNet50 Pretrained từ torchvision."""
        if self.model is not None:
            return self.model

        try:
            import torch
            import torch.nn as nn
            from torchvision.models import resnet50, ResNet50_Weights

            print("[CV] Đang nạp mô hình Pretrained ResNet50 (Transfer Learning)...")
            weights = ResNet50_Weights.DEFAULT
            model = resnet50(weights=weights)
            # Thay thế lớp Fully Connected cuối cùng bằng Identity để lấy vector đặc trưng 2048 chiều
            model.fc = nn.Identity()
            model.eval()
            self.model = model.to(self.device)
            print("[CV] Đã nạp thành công mô hình ResNet50 backbone (Feature Dim = 2048)!")
            return self.model
        except Exception as e:
            print(f"[CV Warning] Chưa thể khởi tạo ResNet50 từ torchvision: {e}")
            return None

    def get_image_path(self, article_id: int) -> Optional[str]:
        """Xác định đường dẫn file ảnh sản phẩm trên đĩa cứng."""
        return resolve_product_image(
            {"article_id": article_id, **self._meta_cache.get(int(article_id), {})},
            image_root=self.image_root,
            local_only=True,
        )

    def _extract_via_base_python(self, image_path: str) -> Optional[np.ndarray]:
        """Dự phòng: gọi trình thông dịch python khả dụng nếu môi trường hiện tại chưa có torch."""
        base_py = os.getenv("PYTHON_EXECUTABLE") or shutil.which("python") or sys.executable
        if not base_py or not os.path.exists(base_py) or os.path.abspath(base_py) == os.path.abspath(sys.executable):
            return None
        import subprocess
        import json
        cmd = [
            base_py, "-c",
            f"import sys; sys.path.insert(0, r'{BASE_DIR}'); from image_feature_extractor import image_feature_extractor; vec = image_feature_extractor.extract_single_image(r'{image_path}'); import json; print('__VEC__' + json.dumps(vec.tolist()) if vec is not None else '__ERR__')"
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
            for line in res.stdout.splitlines():
                if line.startswith("__VEC__"):
                    raw = json.loads(line[7:])
                    return np.array(raw, dtype=np.float32)
        except Exception:
            pass
        return None

    def extract_single_image(self, image_path_or_pil) -> Optional[np.ndarray]:
        """Trích xuất vector 2048 chiều cho một ảnh đơn lẻ và chuẩn hóa L2 norm."""
        if self.transform is None:
            self._init_transforms()
        model = self._load_model()
        if model is None or self.transform is None:
            # Cơ chế dự phòng thông minh qua base python
            if isinstance(image_path_or_pil, str) and os.path.exists(image_path_or_pil):
                fallback_vec = self._extract_via_base_python(image_path_or_pil)
                if fallback_vec is not None:
                    return fallback_vec
            elif hasattr(image_path_or_pil, "save"):
                temp_p = os.path.join(MODELS_CV_DIR, "temp_query.jpg")
                try:
                    image_path_or_pil.save(temp_p)
                    fallback_vec = self._extract_via_base_python(temp_p)
                    if fallback_vec is not None:
                        return fallback_vec
                except Exception:
                    pass
            return None

        try:
            import torch
            if isinstance(image_path_or_pil, str):
                if not os.path.exists(image_path_or_pil):
                    return None
                img = Image.open(image_path_or_pil).convert("RGB")
            else:
                img = image_path_or_pil.convert("RGB")

            tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = model(tensor)
                # Chuẩn hóa L2 norm an toàn, xử lý vector norm = 0
                norm = torch.norm(feat, p=2, dim=1, keepdim=True)
                norm = torch.clamp(norm, min=1e-12)
                feat = feat / norm
                vec = feat.squeeze(0).cpu().numpy().astype(np.float32)
                if not np.all(np.isfinite(vec)):
                    return None
                return vec
        except Exception as e:
            print(f"[CV] Lỗi trích xuất ảnh {image_path_or_pil}: {e}")
            return None

    def build_and_save_catalog_features(self, limit: int = 500) -> bool:
        """
        Trích xuất đặc trưng offline cho toàn bộ danh mục sản phẩm H&M
        và lưu trữ vào file cache .npz để phục vụ truy vấn thời gian thực.
        """
        model = self._load_model()
        if model is None:
            print("[CV Error] Không thể nạp mô hình ResNet50 để trích xuất.")
            return False

        try:
            conn = sqlite3.connect(DB_PATH)
            rows = conn.execute("SELECT article_id FROM products LIMIT ?", (limit,)).fetchall()
            conn.close()
            catalog_art_ids = [r[0] for r in rows]
        except Exception:
            catalog_art_ids = []

        print(f"[CV] Bắt đầu trích xuất đặc trưng ảnh cho {len(catalog_art_ids)} sản phẩm...")
        extracted_ids = []
        extracted_features = []

        for idx, art_id in enumerate(catalog_art_ids, 1):
            img_path = self.get_image_path(art_id)
            if img_path:
                vec = self.extract_single_image(img_path)
                if vec is not None and vec.shape == (2048,):
                    extracted_ids.append(art_id)
                    extracted_features.append(vec)

            if idx % 100 == 0 or idx == len(catalog_art_ids):
                print(f"  -> Tiến độ trích xuất: {idx}/{len(catalog_art_ids)} sản phẩm...")

        if not extracted_ids:
            print("[CV Error] Không tìm thấy ảnh hợp lệ để trích xuất.")
            return False

        self.article_ids = np.array(extracted_ids, dtype=np.int64)
        self.features = np.array(extracted_features, dtype=np.float32)
        self.art_to_idx = {art: i for i, art in enumerate(self.article_ids)}

        # Lưu file cache .npz
        np.savez_compressed(
            CACHE_FEATURE_PATH,
            article_ids=self.article_ids,
            features=self.features
        )
        print(f"[CV] Đã lưu thành công {len(self.article_ids)} vector đặc trưng ảnh (2048 chiều) vào: {CACHE_FEATURE_PATH}")
        return True

    def load_cached_features(self) -> bool:
        """Nạp vector đặc trưng ảnh đã lưu sẵn từ file cache .npz vào RAM."""
        if os.path.exists(CACHE_FEATURE_PATH):
            try:
                data = np.load(CACHE_FEATURE_PATH)
                self.article_ids = data["article_ids"]
                self.features = data["features"]
                self.art_to_idx = {art: i for i, art in enumerate(self.article_ids)}
                print(f"[CV] Đã nạp thành công {len(self.article_ids)} vector ảnh ResNet50 từ cache (Shape: {self.features.shape})!")
                return True
            except Exception as e:
                print(f"[CV Warning] Lỗi nạp cache đặc trưng ảnh: {e}")
        return False

    def get_status(self) -> dict:
        """Trả về thông số kỹ thuật và trạng thái thực tế của pipeline Transfer Learning."""
        import time
        artifact_exists = os.path.exists(CACHE_FEATURE_PATH)
        file_size_bytes = os.path.getsize(CACHE_FEATURE_PATH) if artifact_exists else 0
        file_mtime = time.ctime(os.path.getmtime(CACHE_FEATURE_PATH)) if artifact_exists else None
        
        status_str = "READY" if len(self.features) > 0 else ("NOT_LOADED" if artifact_exists else "NOT_FOUND")
        return {
            "status": status_str,
            "backbone": "ResNet50 (Pretrained ImageNet)",
            "classification_head": "Identity (Global Average Pooling 2048-D)",
            "feature_dimension": 2048,
            "normalization": "L2 Normalized (||v||_2 = 1.0)",
            "similarity_metric": "Cosine Similarity: sim(A, B) = A · B",
            "input_resolution": "224x224 RGB",
            "input_normalization": "mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]",
            "total_vectors": len(self.article_ids),
            "feature_matrix_shape": list(self.features.shape),
            "artifact_file": os.path.basename(CACHE_FEATURE_PATH),
            "artifact_path": CACHE_FEATURE_PATH,
            "artifact_exists": artifact_exists,
            "artifact_size_bytes": file_size_bytes,
            "artifact_size_mb": round(file_size_bytes / (1024 * 1024), 2),
            "artifact_last_modified": file_mtime,
            "device": str(self.device),
        }

    def get_similar_products(
        self,
        target_article_id: int,
        top_k: Optional[int] = 10,
        include_metadata: bool = True,
    ) -> List[dict]:
        """
        Tìm kiếm Top-K sản phẩm có hình ảnh tương đồng nhất dựa trên Cosine Similarity.
        Công thức: sim(A, B) = A · B (do cả 2 vector đã được chuẩn hóa L2 norm).
        top_k=None trả về toàn bộ kết quả, không bao gồm sản phẩm gốc.
        """
        target_art = int(target_article_id)
        if target_art not in self.art_to_idx or len(self.features) == 0:
            return []

        q_idx = self.art_to_idx[target_art]
        query_vec = self.features[q_idx]

        # Tính Cosine Similarity đồng loạt bằng phép nhân ma trận
        sims = np.dot(self.features, query_vec)
        ranked_indices = np.argsort(sims)[::-1]
        top_indices = ranked_indices[ranked_indices != q_idx][:top_k]

        if not include_metadata:
            return [
                {
                    "rank": rank,
                    "product_id": int(self.article_ids[idx]),
                    "article_id": int(self.article_ids[idx]),
                    "similarity": round(float(sims[idx]), 4),
                    "similarity_score": round(float(sims[idx]), 4),
                    "cosine_similarity": round(float(sims[idx]), 4),
                }
                for rank, idx in enumerate(top_indices, 1)
            ]

        # Lấy thông tin sản phẩm trực tiếp từ RAM (O(1)) không tốn thời gian truy vấn SQLite
        self._load_meta_cache()
        results = []
        for rank, idx in enumerate(top_indices, 1):
            art = int(self.article_ids[idx])
            sim_score = float(sims[idx])
            meta = self._meta_cache.get(art, {})
            prod_name = meta.get("prod_name", f"H&M Article {art}")
            group_name = meta.get("product_group_name", "Thời trang")
            price = float(meta.get("price", 0.03))
            img_p = self.get_image_path(art)

            results.append({
                "rank": rank,
                "product_id": art,
                "article_id": art,
                "product_name": prod_name,
                "prod_name": prod_name,
                "category": group_name,
                "product_group_name": group_name,
                "similarity": round(sim_score, 4),
                "similarity_score": round(sim_score, 4),
                "cosine_similarity": round(sim_score, 4),
                "price": price,
                "image": f"/api/products/{art}/image",
                "image_url": f"/api/products/{art}/image",
                "image_path": img_p or "",
                "local_image_path": img_p or ""
            })
        return results

    def get_similar_products_by_image(
        self, 
        image_path_or_pil, 
        top_k: Optional[int] = 10,
        exclude_article_id: Optional[int] = None
    ) -> List[dict]:
        """
        Tìm kiếm Top-K sản phẩm trong catalog có hình ảnh tương đồng nhất với một ảnh tải lên hoặc chụp từ camera.
        Trích xuất vector 2048 chiều bằng ResNet50 Transfer Learning và tính Cosine Similarity với 494 vector cache.
        top_k=None trả về toàn bộ kết quả sau khi loại sản phẩm gốc (nếu có).
        """
        if len(self.features) == 0:
            return []

        query_vec = self.extract_single_image(image_path_or_pil)
        if query_vec is None:
            return []

        # Tính Cosine Similarity đồng loạt với toàn bộ catalog
        sims = np.dot(self.features, query_vec)

        ranked_indices = np.argsort(sims)[::-1]
        if exclude_article_id is not None and int(exclude_article_id) in self.art_to_idx:
            q_idx = self.art_to_idx[int(exclude_article_id)]
            ranked_indices = ranked_indices[ranked_indices != q_idx]

        top_indices = ranked_indices[:top_k]

        self._load_meta_cache()
        results = []
        for rank, idx in enumerate(top_indices, 1):
            art = int(self.article_ids[idx])
            sim_score = float(sims[idx])
            meta = self._meta_cache.get(art, {})
            prod_name = meta.get("prod_name", f"H&M Article {art}")
            group_name = meta.get("product_group_name", "Thời trang")
            price = float(meta.get("price", 0.03))
            img_p = self.get_image_path(art)

            results.append({
                "rank": rank,
                "product_id": art,
                "article_id": art,
                "product_name": prod_name,
                "prod_name": prod_name,
                "category": group_name,
                "product_group_name": group_name,
                "similarity": round(sim_score, 4),
                "similarity_score": round(sim_score, 4),
                "cosine_similarity": round(sim_score, 4),
                "price": price,
                "image": f"/api/products/{art}/image",
                "image_url": f"/api/products/{art}/image",
                "image_path": img_p or "",
                "local_image_path": img_p or ""
            })

        return results

# Singleton instance
image_feature_extractor = ImageFeatureExtractor()
