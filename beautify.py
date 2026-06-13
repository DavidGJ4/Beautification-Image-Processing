"""
Homework 2: Facial Image Beautification Using Frequency-Domain Techniques
==========================================================================
Universal flaw_reduction pipeline: auto face analysis, adaptive masks, FFT +
bilateral smoothing. Modes: red_blemish (spots) | line_flaw (wrinkles/lines).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "images")
OUT_DIR = os.path.join(BASE_DIR, "output")
FACE_MODEL = os.path.join(
    BASE_DIR, "face-smoothing-ref", "face-smoothing-main", "models", "opencv_face_detector_uint8.pb"
)
FACE_CONFIG = os.path.join(
    BASE_DIR, "face-smoothing-ref", "face-smoothing-main", "models", "opencv_face_detector.pbtxt"
)
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

MAX_IMAGE_SIZE = 768
FACE_CONF = 0.55


# face-smoothing-main reference HSV range (configs.yaml)
HSV_SKIN_LOW = np.array([0, 80, 80], dtype=np.uint8)
HSV_SKIN_HIGH = np.array([200, 255, 255], dtype=np.uint8)


# ─────────────────────────────────────────────────────────────
# Profiles — bilateral params from face-smoothing-main + FFT for assignment
# ─────────────────────────────────────────────────────────────
def is_red_blemish(profile: BeautyProfile) -> bool:
    return profile.flaw_mode == "red_blemish"


def is_line_flaw(profile: BeautyProfile) -> bool:
    return profile.flaw_mode == "line_flaw"


@dataclass(frozen=True)
class BeautyProfile:
    name: str
    flaw_mode: str               # red_blemish | line_flaw (both display as flaw_reduction)
    bilateral_d: int
    bilateral_sigma_color: float
    bilateral_sigma_space: float
    bilateral_passes: int
    fft_sigma: float             # Gaussian LP sigma (frequency domain)
    fft_blend: float             # how much FFT-smooth mixes on skin
    sigma_mid_lo: float
    sigma_mid_hi: float
    mid_attenuate: float         # BP wrinkle-band suppression
    forehead_blend: float        # max extra forehead FFT fade (scaled by wrinkle map)
    flaw_inpaint_radius: int
    adaptive_base: float = 0.0   # min bilateral blend on smooth skin
    adaptive_gain: float = 1.0   # extra blend at strongest wrinkles
    before_sigma_lp: float = 14.0


PROFILES: Dict[str, BeautyProfile] = {
    "red_blemish": BeautyProfile(
        name="flaw_reduction",
        flaw_mode="red_blemish",
        bilateral_d=17,
        bilateral_sigma_color=100.0,
        bilateral_sigma_space=100.0,
        bilateral_passes=3,
        fft_sigma=26.0,
        fft_blend=0.10,
        sigma_mid_lo=5.0,
        sigma_mid_hi=28.0,
        mid_attenuate=0.0,
        forehead_blend=0.0,
        flaw_inpaint_radius=5,
    ),
    "line_flaw": BeautyProfile(
        name="flaw_reduction",
        flaw_mode="line_flaw",
        bilateral_d=15,
        bilateral_sigma_color=50.0,
        bilateral_sigma_space=50.0,
        bilateral_passes=1,
        fft_sigma=30.0,
        fft_blend=0.14,
        sigma_mid_lo=3.0,
        sigma_mid_hi=12.0,
        mid_attenuate=0.30,
        forehead_blend=0.22,
        flaw_inpaint_radius=5,
        adaptive_base=0.06,
        adaptive_gain=0.34,
    ),
}


def infer_profile_from_name(image_name: str) -> BeautyProfile | None:
    """Optional filename hint when auto-analysis is ambiguous."""
    key = image_name.lower()
    if any(t in key for t in ("woman", "female", "girl", "wom", "acne", "blemish", "spot")):
        return PROFILES["red_blemish"]
    if any(t in key for t in ("man", "male", "wrinkle", "old", "aged")):
        return PROFILES["line_flaw"]
    return None


@dataclass
class FaceGeometry:
    """Face bounding box with normalized landmark helpers (any image size)."""
    bbox: List[int]
    fw: int
    fh: int
    cx: int
    cy: int

    @classmethod
    def from_bbox(cls, bbox: List[int]) -> "FaceGeometry":
        x1, y1, x2, y2 = bbox
        return cls(bbox, x2 - x1, y2 - y1, (x1 + x2) // 2, (y1 + y2) // 2)

    def point(self, nx: float, ny: float) -> Tuple[int, int]:
        x1, y1 = self.bbox[0], self.bbox[1]
        return x1 + int(nx * self.fw), y1 + int(ny * self.fh)

    def ellipse_mask(self, shape: Tuple[int, int], nx: float, ny: float, rx: float, ry: float) -> np.ndarray:
        h, w = shape
        m = np.zeros((h, w), dtype=np.uint8)
        cx, cy = self.point(nx, ny)
        cv2.ellipse(m, (cx, cy), (max(1, int(rx * self.fw)), max(1, int(ry * self.fh))), 0, 0, 360, 255, -1)
        return m


@dataclass
class FaceAnalysis:
    """Full portrait analysis — drives profile choice, masks, and enhancement."""
    bgr: np.ndarray
    geom: FaceGeometry
    skin_mask: np.ndarray
    profile: BeautyProfile
    flaw_mask: np.ndarray
    wrinkle_map: np.ndarray | None = None
    enhancement_boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    annotate_bgr: Tuple[int, int, int] = (0, 255, 255)


# ─────────────────────────────────────────────────────────────
# Image I/O
# ─────────────────────────────────────────────────────────────
def load_portrait(path: str, max_size: int = MAX_IMAGE_SIZE) -> np.ndarray | None:
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = min(max_size / max(h, w), 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def get_all_portraits() -> list[Tuple[str, np.ndarray]]:
    """Load every portrait in images/ — works with any filenames."""
    pairs: list[Tuple[str, np.ndarray]] = []
    if not os.path.isdir(IMG_DIR):
        raise FileNotFoundError(f"Missing folder: {IMG_DIR}")
    for fn in sorted(os.listdir(IMG_DIR)):
        if not fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            continue
        img = load_portrait(os.path.join(IMG_DIR, fn))
        if img is not None:
            pairs.append((os.path.splitext(fn)[0], img))
    if not pairs:
        raise FileNotFoundError(f"No images in {IMG_DIR}. Add portrait photos.")
    return pairs


# ─────────────────────────────────────────────────────────────
# Kernel cache (face-smoothing uses repeated filter ops per ROI)
# ─────────────────────────────────────────────────────────────
@lru_cache(maxsize=64)
def _cached_gaussian(shape: Tuple[int, int], sigma: float) -> np.ndarray:
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(cols, dtype=np.float64) - ccol
    v = np.arange(rows, dtype=np.float64) - crow
    u, v = np.meshgrid(u, v)
    return np.exp(-(u ** 2 + v ** 2) / (2.0 * sigma ** 2))


def make_gaussian_kernel(shape: Tuple[int, int], sigma: float) -> np.ndarray:
    return _cached_gaussian(shape, round(sigma, 2))


def make_highpass_kernel(shape: Tuple[int, int], sigma: float) -> np.ndarray:
    return 1.0 - make_gaussian_kernel(shape, sigma)


def make_bandpass_kernel(shape: Tuple[int, int], sigma_low: float, sigma_high: float) -> np.ndarray:
    return (make_gaussian_kernel(shape, sigma_high)
            - make_gaussian_kernel(shape, sigma_low))


def make_butterworth_lpf(shape: Tuple[int, int], D0: float, n: int = 2) -> np.ndarray:
    """Butterworth low-pass H(u,v) — smooth transition, no ringing."""
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(cols, dtype=np.float64) - ccol
    v = np.arange(rows, dtype=np.float64) - crow
    u, v = np.meshgrid(u, v)
    D = np.sqrt(u ** 2 + v ** 2)
    return 1.0 / (1.0 + (D / (D0 + 1e-6)) ** (2 * n))


def make_butterworth_hpf(shape: Tuple[int, int], D0: float, n: int = 1) -> np.ndarray:
    """Butterworth high-pass — smooth cutoff avoids Gibbs / halo artifacts."""
    return 1.0 - make_butterworth_lpf(shape, D0, n)


def make_butterworth_brf(shape: Tuple[int, int], D0: float, W: float, n: int = 2) -> np.ndarray:
    """Butterworth band-reject — blocks mid-band (skin texture), keeps low + high."""
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(cols, dtype=np.float64) - ccol
    v = np.arange(rows, dtype=np.float64) - crow
    u, v = np.meshgrid(u, v)
    D = np.sqrt(u ** 2 + v ** 2)
    num = D * W
    den = np.maximum(D ** 2 - D0 ** 2, 1e-6)
    return 1.0 / (1.0 + (num / den) ** (2 * n))


def _padded_shape(orig_shape: Tuple[int, int]) -> Tuple[int, int]:
    m, n = orig_shape
    p = int(2 ** np.ceil(np.log2(max(2 * m - 1, 1))))
    q = int(2 ** np.ceil(np.log2(max(2 * n - 1, 1))))
    return p, q


def apply_frequency_filter(channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    m, n = channel.shape
    p, q = kernel.shape
    padded = np.pad(channel.astype(np.float64), ((0, p - m), (0, q - n)), mode="reflect")
    spectrum = np.fft.fftshift(np.fft.fft2(padded))
    return np.fft.ifft2(np.fft.ifftshift(spectrum * kernel)).real[:m, :n]


# ─────────────────────────────────────────────────────────────
# face-smoothing-main core — DNN face ROI + HSV mask + bilateral
# ─────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _face_detector() -> cv2.dnn.Net:
    return cv2.dnn.readNetFromTensorflow(FACE_MODEL, FACE_CONFIG)


def detect_face_bboxes(bgr: np.ndarray) -> list[list[int]]:
    """OpenCV DNN face detector (same model as face-smoothing-main)."""
    h, w = bgr.shape[:2]
    net = _face_detector()
    blob = cv2.dnn.blobFromImage(bgr, 1.0, (200, 200), (104, 117, 123), False, False)
    net.setInput(blob)
    detections = net.forward()
    bboxes: list[list[int]] = []
    for i in range(detections.shape[2]):
        if detections[0, 0, i, 2] > FACE_CONF:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            bboxes.append([x1, y1, x2, y2])
    return bboxes


def _fallback_face_bbox(shape: Tuple[int, int]) -> list[int]:
    h, w = shape
    cx, cy = w // 2, int(h * 0.46)
    bw, bh = int(w * 0.52), int(h * 0.62)
    return [cx - bw // 2, cy - bh // 2, cx + bw // 2, cy + bh // 2]


def _expand_bbox(bbox: list[int], shape: Tuple[int, int], pad_frac: float = 0.10) -> list[int]:
    x1, y1, x2, y2 = bbox
    h, w = shape
    px = int((x2 - x1) * pad_frac)
    py = int((y2 - y1) * pad_frac)
    return [max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py)]


def _roi_edge_feather(roi_shape: Tuple[int, int]) -> np.ndarray:
    """Soft falloff at ROI boundary — removes rectangular face-box seam."""
    rh, rw = roi_shape
    yy, xx = np.mgrid[0:rh, 0:rw].astype(np.float64)
    cx, cy = (rw - 1) / 2.0, (rh - 1) / 2.0
    nx = (xx - cx) / max(cx, 1)
    ny = (yy - cy) / max(cy, 1)
    r = np.sqrt(nx * nx + ny * ny)
    return np.clip(1.15 - r, 0, 1).astype(np.float32)


def build_hsv_skin_mask_roi(roi_bgr: np.ndarray) -> np.ndarray:
    """HSV skin mask inside face ROI only (face-smoothing configs.yaml)."""
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, HSV_SKIN_LOW, HSV_SKIN_HIGH)


def build_skin_mask(bgr: np.ndarray, geom: FaceGeometry) -> np.ndarray:
    x1, y1, x2, y2 = geom.bbox
    skin = np.zeros(bgr.shape[:2], dtype=np.uint8)
    skin[y1:y2, x1:x2] = build_hsv_skin_mask_roi(bgr[y1:y2, x1:x2])
    return skin


def infer_profile_from_image(
    bgr: np.ndarray, skin_mask: np.ndarray, name_hint: str = "",
) -> BeautyProfile:
    """Choose acne vs wrinkle profile from measured skin statistics."""
    hint = infer_profile_from_name(name_hint)
    skin = skin_mask > 0
    if not np.any(skin):
        return hint or PROFILES["red_blemish"]

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    a_ch = lab[:, :, 1].astype(np.float32)
    local_a = cv2.GaussianBlur(a_ch, (41, 41), 0)
    red_frac = float(np.mean((a_ch > local_a + 2.0) & skin))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    red_hsv = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 45, 45]), np.array([14, 255, 255])),
        cv2.inRange(hsv, np.array([165, 45, 45]), np.array([180, 255, 255])),
    )
    red_hsv_frac = float(np.mean((red_hsv > 0) & skin))
    acne_score = 0.6 * red_frac + 0.4 * red_hsv_frac

    y = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float64)
    pad = _padded_shape(y.shape)
    mid = apply_frequency_filter(y, make_bandpass_kernel(pad, 3.0, 12.0))
    wrinkle_e = cv2.GaussianBlur(np.abs(mid), (11, 11), 0)
    wrinkle_score = float(np.percentile(wrinkle_e[skin], 82))
    line_score = wrinkle_score / max(acne_score, 0.001)

    if hint is not None:
        return hint
    if acne_score > 0.065:
        return PROFILES["red_blemish"]
    if wrinkle_score > 4.2 and acne_score < 0.04:
        return PROFILES["line_flaw"]
    if line_score > 130 and acne_score < 0.05:
        return PROFILES["line_flaw"]
    if acne_score > 0.035:
        return PROFILES["red_blemish"]
    if line_score > 90:
        return PROFILES["line_flaw"]
    return PROFILES["red_blemish"]


def build_under_eye_boost(shape: Tuple[int, int], geom: FaceGeometry) -> np.ndarray:
    """Under-eye / crow's feet zone — bbox-relative, scales with any face size."""
    boost = np.zeros(shape, dtype=np.float32)
    for nx in (0.32, 0.68):
        patch = geom.ellipse_mask(shape, nx, 0.56, 0.15, 0.08)
        boost = np.maximum(boost, patch.astype(np.float32) / 255.0)
    cheek = geom.ellipse_mask(shape, 0.5, 0.62, 0.38, 0.10).astype(np.float32) / 255.0
    boost = np.clip(boost + cheek * 0.25, 0.0, 1.0)
    return cv2.GaussianBlur(boost, (19, 19), 0)


def build_wrinkle_strength_map(
    bgr: np.ndarray, skin_mask: np.ndarray, profile: BeautyProfile,
    geom: FaceGeometry | None = None,
) -> np.ndarray:
    """
    Per-pixel wrinkle severity in [0, 1] from mid-band luminance energy.
    Stronger wrinkles -> higher value -> more smoothing allowed there.
    """
    y = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float64)
    pad = _padded_shape(y.shape)
    h_bp = make_bandpass_kernel(pad, profile.sigma_mid_lo, profile.sigma_mid_hi)
    mid = apply_frequency_filter(y, h_bp)
    energy = cv2.GaussianBlur(np.abs(mid), (11, 11), 0)

    skin = skin_mask > 0
    if not np.any(skin):
        return np.zeros(y.shape, dtype=np.float32)

    vals = energy[skin]
    lo = float(np.percentile(vals, 35))
    hi = float(np.percentile(vals, 90))
    strength = (energy - lo) / (hi - lo + 1e-6)
    strength = np.clip(strength, 0.0, 1.0)
    strength *= skin_mask.astype(np.float64) / 255.0
    if geom is not None:
        under_eye = build_under_eye_boost(bgr.shape[:2], geom)
        strength = np.clip(strength + 0.58 * under_eye, 0.0, 1.0)
        forehead = geom.ellipse_mask(bgr.shape[:2], 0.5, 0.20, 0.40, 0.09).astype(np.float32) / 255.0
        strength = np.clip(strength + 0.10 * forehead, 0.0, 1.0)
    return cv2.GaussianBlur(strength.astype(np.float32), (17, 17), 0)


def _adaptive_blend_alpha(
    skin_mask: np.ndarray, wrinkle_map: np.ndarray | None, profile: BeautyProfile,
) -> np.ndarray:
    """Blend weight per pixel: light on smooth skin, stronger only on wrinkles."""
    skin = skin_mask.astype(np.float32) / 255.0
    if is_line_flaw(profile) and wrinkle_map is not None:
        w = profile.adaptive_base + profile.adaptive_gain * wrinkle_map
        return np.clip(skin * w, 0.0, 1.0)
    return skin


def smooth_portrait_reference(
    bgr: np.ndarray,
    profile: BeautyProfile,
    wrinkle_map: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    face-smoothing-main bilateral + HSV mask inside face ROI.
    Wrinkle profile: partial blend scaled by wrinkle_strength map (not 100% skin).
    """
    output = bgr.copy()
    full_skin = np.zeros(bgr.shape[:2], dtype=np.uint8)
    bboxes = [_expand_bbox(b, bgr.shape[:2]) for b in (detect_face_bboxes(bgr) or [_fallback_face_bbox(bgr.shape[:2])])]

    for x1, y1, x2, y2 in bboxes:
        if x2 <= x1 or y2 <= y1:
            continue

        roi = output[y1:y2, x1:x2]
        temp = roi.copy()
        skin_mask = build_hsv_skin_mask_roi(roi)
        full_skin[y1:y2, x1:x2] = cv2.bitwise_or(full_skin[y1:y2, x1:x2], skin_mask)

        wrinkle_roi = None
        if wrinkle_map is not None:
            wrinkle_roi = wrinkle_map[y1:y2, x1:x2]

        blurred = roi.copy()
        for _ in range(profile.bilateral_passes):
            blurred = cv2.bilateralFilter(
                blurred, profile.bilateral_d,
                profile.bilateral_sigma_color, profile.bilateral_sigma_space,
            )

        alpha = _adaptive_blend_alpha(skin_mask, wrinkle_roi, profile)
        alpha3 = np.stack([alpha, alpha, alpha], axis=2)
        smoothed_roi = clip_to_uint8(temp.astype(np.float32) * (1.0 - alpha3) + blurred.astype(np.float32) * alpha3)

        feather = _roi_edge_feather((y2 - y1, x2 - x1))
        feather3 = np.stack([feather, feather, feather], axis=2)
        output[y1:y2, x1:x2] = clip_to_uint8(
            temp.astype(np.float32) * (1.0 - feather3) + smoothed_roi.astype(np.float32) * feather3
        )

    return output, full_skin


def _filter_small_flaw_blobs(
    mask: np.ndarray, shape: Tuple[int, int], compact_only: bool = False,
) -> np.ndarray:
    """Keep only compact blemish-sized blobs — not whole cheeks or mustache hair."""
    h, w = shape
    max_area = max(80, int(h * w * 0.0035))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        compact = min(bw, bh) / max(bw, bh, 1)
        if not (10 <= area <= max_area):
            continue
        if compact_only and compact < 0.35:
            continue
        out[labels == i] = 255
    return out


def detect_skin_flaws(
    bgr: np.ndarray, skin_mask: np.ndarray, profile: BeautyProfile,
    geom: FaceGeometry | None = None,
) -> np.ndarray:
    """Detect individual acne pimples or isolated moles — not wrinkles or whole regions."""
    skin = skin_mask > 0
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    a_ch = lab[:, :, 1].astype(np.float32)
    l_ch = lab[:, :, 0].astype(np.float32)
    local_a = cv2.GaussianBlur(a_ch, (41, 41), 0)
    local_l = cv2.GaussianBlur(l_ch, (41, 41), 0)

    lap = np.abs(cv2.Laplacian(l_ch, cv2.CV_32F))
    texture = cv2.GaussianBlur(lap, (7, 7), 0)

    if is_line_flaw(profile):
        h, w = bgr.shape[:2]
        flaw = np.zeros((h, w), dtype=bool)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        local_g = cv2.GaussianBlur(gray, (41, 41), 0)
        if geom is not None:
            for nx in (0.38, 0.62):
                patch = geom.ellipse_mask((h, w), nx, 0.70, 0.06, 0.045)
                flaw = flaw | ((gray < local_g - 11.0) & (patch > 0))
        else:
            for dx in (-int(w * 0.11), int(w * 0.11)):
                patch = np.zeros((h, w), dtype=np.uint8)
                cv2.ellipse(patch, (w // 2 + dx, int(h * 0.635)), (int(w * 0.05), int(h * 0.04)), 0, 0, 360, 255, -1)
                flaw = flaw | ((gray < local_g - 11.0) & (patch > 0))
    else:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        red_hsv = cv2.bitwise_or(
            cv2.inRange(hsv, np.array([0, 50, 50]), np.array([12, 255, 255])),
            cv2.inRange(hsv, np.array([168, 50, 50]), np.array([180, 255, 255])),
        )
        red_blemish = ((a_ch > local_a + 3.0) & (red_hsv > 0) & (texture > 1.5) & skin)
        dark_spot = ((l_ch < local_l - 12.0) & (texture > 2.5) & skin)
        flaw = red_blemish | dark_spot

    raw = flaw.astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, k, iterations=1)
    raw = cv2.dilate(raw, k, iterations=1)
    raw = cv2.bitwise_and(raw, skin_mask)
    compact = is_line_flaw(profile)
    return _filter_small_flaw_blobs(raw, bgr.shape[:2], compact_only=compact)


def remove_skin_flaws(bgr: np.ndarray, flaw_mask: np.ndarray, profile: BeautyProfile) -> np.ndarray:
    """Inpaint only small detected flaw pixels — no global whitening."""
    if cv2.countNonZero(flaw_mask) == 0:
        return bgr
    if is_line_flaw(profile):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        flaw_mask = cv2.dilate(flaw_mask, k, iterations=1)
    guide = bgr.copy()
    for _ in range(profile.bilateral_passes):
        guide = cv2.bilateralFilter(
            guide, profile.bilateral_d,
            profile.bilateral_sigma_color, profile.bilateral_sigma_space,
        )
    inpainted = cv2.inpaint(guide, flaw_mask, profile.flaw_inpaint_radius, cv2.INPAINT_NS)
    fm = (flaw_mask > 0).astype(np.float32)
    fm3 = np.stack([fm, fm, fm], axis=2)
    return clip_to_uint8(bgr.astype(np.float32) * (1.0 - fm3) + inpainted.astype(np.float32) * fm3)


def build_forehead_mask(shape: Tuple[int, int], geom: FaceGeometry | None = None) -> np.ndarray:
    """Forehead zone from face bbox when available."""
    if geom is not None:
        m = geom.ellipse_mask(shape, 0.5, 0.20, 0.44, 0.11)
    else:
        h, w = shape
        m = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(m, (w // 2, int(h * 0.255)), (int(w * 0.26), int(h * 0.075)), 0, 0, 360, 255, -1)
    return cv2.GaussianBlur(m.astype(np.float64), (25, 25), 0) / 255.0


def decompose_frequency_layers(channel: np.ndarray, profile: BeautyProfile) -> Dict[str, np.ndarray]:
    pad = _padded_shape(channel.shape)
    h_lp = make_gaussian_kernel(pad, profile.fft_sigma)
    h_bp = make_bandpass_kernel(pad, profile.sigma_mid_lo, profile.sigma_mid_hi)
    low = apply_frequency_filter(channel, h_lp)
    detail = channel - low
    mid = apply_frequency_filter(channel, h_bp)
    mid_residual = apply_frequency_filter(detail, h_bp)
    return {
        "low": low, "mid": mid, "detail": detail,
        "mid_residual": mid_residual, "orig": channel,
    }


def apply_fft_skin_smooth(
    bgr: np.ndarray,
    skin_mask: np.ndarray,
    profile: BeautyProfile,
    wrinkle_map: np.ndarray | None = None,
    geom: FaceGeometry | None = None,
) -> np.ndarray:
    """
    Frequency-domain Gaussian LP on Y + BP wrinkle attenuation.
    Wrinkle profile: FFT strength follows wrinkle_strength map (local, not global).
    """
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb).astype(np.float64)
    y, cr, cb = ycrcb[:, :, 0], ycrcb[:, :, 1], ycrcb[:, :, 2]
    layers = decompose_frequency_layers(y, profile)

    y_smooth = layers["low"]
    if profile.mid_attenuate > 0 and wrinkle_map is not None:
        y_smooth = y_smooth - profile.mid_attenuate * layers["mid_residual"] * wrinkle_map
    elif profile.mid_attenuate > 0:
        y_smooth = y_smooth - profile.mid_attenuate * layers["mid_residual"]

    skin_f = cv2.GaussianBlur(skin_mask.astype(np.float64), (31, 31), 0) / 255.0
    if wrinkle_map is not None:
        alpha = skin_f * wrinkle_map * profile.fft_blend
    else:
        alpha = skin_f * profile.fft_blend
    y_out = y * (1.0 - alpha) + y_smooth * alpha

    if profile.forehead_blend > 0 and wrinkle_map is not None:
        fh = build_forehead_mask(y.shape, geom)
        fb = profile.forehead_blend * fh * wrinkle_map
        y_out = y_out * (1.0 - fb) + layers["low"] * fb
    elif profile.forehead_blend > 0:
        fh = build_forehead_mask(y.shape, geom)
        fb = profile.forehead_blend * fh
        y_out = y_out * (1.0 - fb) + layers["low"] * fb

    return _ycrcb_to_bgr(np.clip(y_out, 0, 255), cr, cb)


def apply_under_eye_line_pass(bgr: np.ndarray, analysis: FaceAnalysis, profile: BeautyProfile) -> np.ndarray:
    """Extra targeted under-eye / crow's feet pass — line_flaw mode only."""
    if not is_line_flaw(profile) or analysis.wrinkle_map is None:
        return bgr
    under = build_under_eye_boost(bgr.shape[:2], analysis.geom)
    alpha = np.clip(under * analysis.wrinkle_map * 0.42, 0.0, 0.42)

    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb).astype(np.float64)
    y, cr, cb = ycrcb[:, :, 0], ycrcb[:, :, 1], ycrcb[:, :, 2]
    layers = decompose_frequency_layers(y, profile)
    y_target = layers["low"] - 0.22 * layers["mid_residual"] * analysis.wrinkle_map
    y_out = y * (1.0 - alpha) + y_target * alpha
    return _ycrcb_to_bgr(np.clip(y_out, 0, 255), cr, cb)


def clip_to_uint8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0, 255).astype(np.uint8)


def _ycrcb_to_bgr(y: np.ndarray, cr: np.ndarray, cb: np.ndarray) -> np.ndarray:
    ycc = np.stack([clip_to_uint8(y), clip_to_uint8(cr), clip_to_uint8(cb)], axis=2)
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


def _mask_to_boxes(mask: np.ndarray, min_area: int = 20, pad: int = 4) -> List[Tuple[int, int, int, int]]:
    """Connected components -> bounding boxes for flaw preview."""
    h, w = mask.shape
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes: List[Tuple[int, int, int, int]] = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        x = max(0, stats[i, cv2.CC_STAT_LEFT] - pad)
        y = max(0, stats[i, cv2.CC_STAT_TOP] - pad)
        bw = min(w - x, stats[i, cv2.CC_STAT_WIDTH] + 2 * pad)
        bh = min(h - y, stats[i, cv2.CC_STAT_HEIGHT] + 2 * pad)
        boxes.append((x, y, bw, bh))
    return boxes


def collect_enhancement_boxes(analysis: FaceAnalysis) -> Tuple[List[Tuple[int, int, int, int]], Tuple[int, int, int]]:
    """Regions that will be enhanced — yellow (acne) or green (wrinkles/moles)."""
    if is_red_blemish(analysis.profile):
        boxes = _mask_to_boxes(analysis.flaw_mask, min_area=10, pad=5)
        return boxes, (0, 255, 255)

    boxes: List[Tuple[int, int, int, int]] = []
    if analysis.wrinkle_map is not None:
        wm = ((analysis.wrinkle_map > 0.28).astype(np.uint8) * 255)
        wm = cv2.bitwise_and(wm, analysis.skin_mask)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        wm = cv2.morphologyEx(wm, cv2.MORPH_CLOSE, k, iterations=1)
        boxes = _mask_to_boxes(wm, min_area=35, pad=6)
    under_eye = build_under_eye_boost(analysis.bgr.shape[:2], analysis.geom)
    ue_mask = ((under_eye > 0.35).astype(np.uint8) * 255)
    ue_mask = cv2.bitwise_and(ue_mask, analysis.skin_mask)
    boxes.extend(_mask_to_boxes(ue_mask, min_area=80, pad=4))
    boxes.extend(_mask_to_boxes(analysis.flaw_mask, min_area=8, pad=4))
    return boxes, (0, 255, 0)


def render_before_preview(bgr: np.ndarray, analysis: FaceAnalysis) -> np.ndarray:
    """Before image: original with neon boxes on detected enhancement targets."""
    img = bgr.copy()
    color = analysis.annotate_bgr
    for x, y, bw, bh in analysis.enhancement_boxes:
        x2, y2 = x + bw, y + bh
        for t in (3, 1):
            cv2.rectangle(img, (x - t, y - t), (x2 + t, y2 + t), color, t)
    x1, y1, x2, y2 = analysis.geom.bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), (180, 180, 180), 1)
    return img


def apply_gradient_acne_tone(bgr: np.ndarray, analysis: FaceAnalysis) -> np.ndarray:
    """
    Multi-scale LAB chrominance fix — pulls red acne toward local skin gradient
    without flattening the whole face to one color.
    """
    skin = analysis.skin_mask.astype(np.float32) / 255.0
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    a_fine = cv2.GaussianBlur(a, (0, 0), 14)
    a_coarse = cv2.GaussianBlur(a, (0, 0), 42)
    b_fine = cv2.GaussianBlur(b, (0, 0), 14)
    b_coarse = cv2.GaussianBlur(b, (0, 0), 42)
    local_a = 0.72 * a_fine + 0.28 * a_coarse
    local_b = 0.72 * b_fine + 0.28 * b_coarse

    red_excess = np.clip((a - a_fine - 1.2) / 10.0, 0.0, 1.0)
    flaw_w = analysis.flaw_mask.astype(np.float32) / 255.0
    red_weight = np.clip(np.maximum(red_excess, flaw_w * 0.7), 0.0, 1.0) * skin

    strength = red_weight * 0.88
    a_out = a * (1.0 - strength) + local_a * strength
    b_out = b * (1.0 - strength * 0.42) + local_b * (strength * 0.42)

    pad = _padded_shape(a.shape)
    a_lp = apply_frequency_filter(a, make_gaussian_kernel(pad, 16.0))
    b_lp = apply_frequency_filter(b, make_gaussian_kernel(pad, 20.0))
    lp_blend = np.clip(red_weight * 0.40, 0.0, 0.40)
    a_out = a_out * (1.0 - lp_blend) + a_lp * lp_blend
    b_out = b_out * (1.0 - lp_blend * 0.5) + b_lp * (lp_blend * 0.5)

    y_fine = cv2.GaussianBlur(L, (0, 0), 8)
    y_coarse = cv2.GaussianBlur(L, (0, 0), 28)
    local_y = 0.65 * y_fine + 0.35 * y_coarse
    y_blend = np.clip(red_weight * 0.25, 0.0, 0.25)
    L_out = L * (1.0 - y_blend) + local_y * y_blend

    lab[:, :, 0] = np.clip(L_out, 0, 255)
    lab[:, :, 1] = np.clip(a_out, 0, 255)
    lab[:, :, 2] = np.clip(b_out, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def analyze_portrait(bgr: np.ndarray, name_hint: str = "") -> FaceAnalysis:
    """Analyze any portrait: face geometry, skin mask, profile, flaw/wrinkle maps."""
    bboxes = detect_face_bboxes(bgr) or [_fallback_face_bbox(bgr.shape[:2])]
    geom = FaceGeometry.from_bbox(_expand_bbox(bboxes[0], bgr.shape[:2]))
    skin_mask = build_skin_mask(bgr, geom)
    profile = infer_profile_from_image(bgr, skin_mask, name_hint)
    flaw_mask = detect_skin_flaws(bgr, skin_mask, profile, geom)
    wrinkle_map = None
    if is_line_flaw(profile):
        wrinkle_map = build_wrinkle_strength_map(bgr, skin_mask, profile, geom)

    analysis = FaceAnalysis(
        bgr=bgr, geom=geom, skin_mask=skin_mask, profile=profile,
        flaw_mask=flaw_mask, wrinkle_map=wrinkle_map,
    )
    boxes, color = collect_enhancement_boxes(analysis)
    analysis.enhancement_boxes = boxes
    analysis.annotate_bgr = color
    return analysis


# ─────────────────────────────────────────────────────────────
# Pipelines
# ─────────────────────────────────────────────────────────────
def enhance_before(analysis: FaceAnalysis) -> dict:
    """Before = flaw/wrinkle preview with neon target boxes."""
    bgr = analysis.bgr
    profile = analysis.profile
    shape = bgr.shape[:2]
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0].astype(np.float64)
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    pad = _padded_shape(y.shape)
    y_lp = apply_frequency_filter(y, make_gaussian_kernel(pad, profile.before_sigma_lp))
    return {
        "image": render_before_preview(bgr, analysis),
        "lp": _ycrcb_to_bgr(y_lp, cr, cb),
        "H_LP": make_gaussian_kernel(shape, profile.before_sigma_lp),
        "profile": profile,
        "analysis": analysis,
    }


def enhance_after(analysis: FaceAnalysis) -> dict:
    """Beautify using pre-computed analysis (any portrait)."""
    bgr = analysis.bgr
    profile = analysis.profile
    shape = bgr.shape[:2]
    skin_mask = analysis.skin_mask
    flaw_mask = analysis.flaw_mask
    wrinkle_map = analysis.wrinkle_map
    geom = analysis.geom

    work = bgr
    if is_line_flaw(profile) and cv2.countNonZero(flaw_mask) > 0:
        work = remove_skin_flaws(bgr, flaw_mask, profile)

    result, skin_mask = smooth_portrait_reference(work, profile, wrinkle_map)
    result = apply_fft_skin_smooth(result, skin_mask, profile, wrinkle_map, geom)
    if is_line_flaw(profile):
        result = apply_under_eye_line_pass(result, analysis, profile)

    if is_red_blemish(profile):
        result = apply_gradient_acne_tone(result, analysis)

    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb).astype(np.float64)
    y, cr, cb = ycrcb[:, :, 0], ycrcb[:, :, 1], ycrcb[:, :, 2]
    layers = decompose_frequency_layers(y, profile)
    y_smooth = layers["low"]
    if profile.mid_attenuate > 0:
        y_smooth = y_smooth - profile.mid_attenuate * layers["mid_residual"]
    hp_vis = layers["detail"] - layers["mid_residual"]

    return {
        "image": result,
        "lp": _ycrcb_to_bgr(layers["low"], cr, cb),
        "bp": _ycrcb_to_bgr(layers["mid"] + 128, cr, cb),
        "hp": _ycrcb_to_bgr(np.clip(hp_vis + 128, 0, 255), cr, cb),
        "smooth_layer": _ycrcb_to_bgr(y_smooth, cr, cb),
        "flaw_mask": flaw_mask,
        "skin_mask": skin_mask,
        "wrinkle_map": clip_to_uint8(wrinkle_map * 255) if wrinkle_map is not None else skin_mask,
        "H_LP": make_gaussian_kernel(shape, profile.fft_sigma),
        "H_BP": make_bandpass_kernel(shape, profile.sigma_mid_lo, profile.sigma_mid_hi),
        "H_HP": make_highpass_kernel(shape, 10.0),
        "profile": profile,
        "analysis": analysis,
    }


def compute_magnitude_spectrum(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    mag = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray))))
    return (mag / mag.max() * 255).astype(np.uint8)


def radial_profile_vectorized(mag: np.ndarray) -> np.ndarray:
    """Vectorised radial mean — O(HW) via bincount."""
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.indices((h, w))
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.int32)
    max_r = min(cx, cy)
    rr = np.clip(rr, 0, max_r - 1)
    sums = np.bincount(rr.ravel(), weights=mag.ravel(), minlength=max_r)
    counts = np.bincount(rr.ravel(), minlength=max_r)
    return sums / np.maximum(counts, 1)


# ─────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────
def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_comparison_figure(name, original, before_d, after_d, out_dir):
    prof = after_d["profile"]
    fig = plt.figure(figsize=(20, 10))
    fig.patch.set_facecolor("#0f0f14")
    gs = gridspec.GridSpec(2, 5, figure=fig, hspace=0.38, wspace=0.22,
                           left=0.03, right=0.98, top=0.90, bottom=0.05)
    tw = dict(color="white", fontsize=9, fontweight="bold", pad=5)
    panels = [
        (0, 0, bgr_to_rgb(original), "Original"),
        (0, 1, bgr_to_rgb(before_d["image"]), "Before (restoration)"),
        (0, 2, bgr_to_rgb(after_d["image"]), "After (cosmetic)"),
        (0, 3, compute_magnitude_spectrum(original), "FFT |F(u,v)|"),
        (0, 4, compute_magnitude_spectrum(after_d["image"]), "FFT after"),
        (1, 0, bgr_to_rgb(before_d["lp"]), f"LP σ={prof.before_sigma_lp:.0f}"),
        (1, 1, bgr_to_rgb(after_d["lp"]), f"LP sigma={prof.fft_sigma:.0f}"),
        (1, 2, bgr_to_rgb(after_d["bp"]), f"BP σ {prof.sigma_mid_lo:.0f}–{prof.sigma_mid_hi:.0f}"),
        (1, 3, bgr_to_rgb(after_d["hp"]), "HP detail layer"),
        (1, 4, after_d["flaw_mask"], "Detected flaws"),
    ]
    for row, col, data, title in panels:
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(data, cmap="gray" if data.ndim == 2 else None)
        ax.set_title(title, **tw)
        ax.axis("off")
    fig.suptitle(f"Frequency-Domain Beautification — {name}", color="white", fontsize=14, y=0.97)
    path = os.path.join(out_dir, f"{name}_comparison.png")
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def save_triptych_figure(
    name: str,
    original: np.ndarray,
    targets_img: np.ndarray,
    after_img: np.ndarray,
    out_dir: str,
) -> str:
    """Original → Target spots → After in one horizontal panel."""
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.patch.set_facecolor("#1a1a1f")
    tw = dict(color="white", fontsize=11, fontweight="bold", pad=8)
    for ax, img, title in zip(
        axes,
        (bgr_to_rgb(original), bgr_to_rgb(targets_img), bgr_to_rgb(after_img)),
        ("Original", "Target spots", "After (cosmetic)"),
    ):
        ax.imshow(img)
        ax.set_title(title, **tw)
        ax.axis("off")
    fig.suptitle(f"{name}  —  Original  →  Target spots  →  After", color="white", fontsize=14, y=1.02)
    path = os.path.join(out_dir, f"{name}_triptych.png")
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def save_professor_figure(name, original, after_d, out_dir):
    """
    Side-by-side deliverable with FFT equation overlay on spectrum panel.
    """
    prof = after_d["profile"]
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor("#0f0f14")
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1.2, 1],
                           hspace=0.28, wspace=0.18, left=0.04, right=0.96, top=0.88, bottom=0.06)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(bgr_to_rgb(original))
    ax0.set_title("Original f(x,y)", color="white", fontsize=11, fontweight="bold")
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(bgr_to_rgb(after_d["image"]))
    ax1.set_title(f"Beautified g(x,y) — {prof.name}", color="#33dd88", fontsize=11, fontweight="bold")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 2])
    diff = np.clip(np.abs(original.astype(np.int16) - after_d["image"].astype(np.int16)) * 3, 0, 255)
    ax2.imshow(bgr_to_rgb(diff.astype(np.uint8)))
    ax2.set_title("Change map (×3)", color="white", fontsize=11, fontweight="bold")
    ax2.axis("off")

    gray_o = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gray_a = cv2.cvtColor(after_d["image"], cv2.COLOR_BGR2GRAY).astype(np.float64)
    spec_o = compute_magnitude_spectrum(original)
    spec_a = compute_magnitude_spectrum(after_d["image"])

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(spec_o, cmap="inferno")
    ax3.set_title("Original spectrum", color="white", fontsize=10)
    ax3.axis("off")

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(spec_a, cmap="inferno")
    ax4.set_title("Filtered spectrum", color="white", fontsize=10)
    ax4.axis("off")
    ax4.text(0.5, -0.12,
             r"$F(u,v)=\mathrm{FFT}\{f(x,y)\}$" + "\n"
             r"$G(u,v)=H_{\mathrm{LP,BP,HP}}\!\cdot\!F$" + "\n"
             r"$g(x,y)=\mathrm{real}\{\mathrm{IFFT}\{G\}\}$",
             transform=ax4.transAxes, ha="center", color="#aaaacc", fontsize=10)

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor("#1a1a28")
    p_o = radial_profile_vectorized(np.abs(np.fft.fftshift(np.fft.fft2(gray_o))))
    p_a = radial_profile_vectorized(np.abs(np.fft.fftshift(np.fft.fft2(gray_a))))
    x = np.arange(len(p_o))
    ax5.plot(x, p_o, "#5588ff", label="Original", lw=1.5)
    ax5.plot(x, p_a, "#33dd88", label="Beautified", lw=1.5)
    ax5.set_title("Radial energy (high-freq → right)", color="white", fontsize=10)
    ax5.legend(facecolor="#1a1a28", labelcolor="white", fontsize=8)
    ax5.tick_params(colors="gray")

    fig.suptitle(f"HW2 Frequency-Domain Facial Beautification — {name}",
                 color="white", fontsize=13, fontweight="bold", y=0.96)
    path = os.path.join(out_dir, f"{name}_professor.png")
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def save_filter_figure(name, before_d, after_d, out_dir):
    prof = after_d["profile"]
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    fig.patch.set_facecolor("#0f0f14")
    for ax, (h, t) in zip(axes[:4], [
        (before_d["H_LP"], f"Before LP σ={prof.before_sigma_lp:.0f}"),
        (after_d["H_LP"], f"After LP sigma={prof.fft_sigma:.0f}"),
        (after_d["H_BP"], f"BP σ {prof.sigma_mid_lo:.0f}–{prof.sigma_mid_hi:.0f}"),
        (after_d["H_HP"], "HP detail"),
    ]):
        ax.imshow(h, cmap="viridis", vmin=0, vmax=1)
        ax.set_title(t, color="white", fontsize=8)
        ax.axis("off")
    cy, cx = after_d["H_LP"].shape[0] // 2, after_d["H_LP"].shape[1] // 2
    axes[4].plot(after_d["H_LP"][cy, cx:], color="#33dd88", lw=2)
    axes[4].set_facecolor("#1a1a28")
    axes[4].set_title("LP radial profile", color="white", fontsize=8)
    axes[4].tick_params(colors="gray", labelsize=7)
    path = os.path.join(out_dir, f"{name}_filters.png")
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def save_pipeline_figure(name, original, after_d, out_dir):
    prof = after_d["profile"]
    fig = plt.figure(figsize=(20, 8))
    fig.patch.set_facecolor("#0f0f14")
    gs = gridspec.GridSpec(2, 5, figure=fig, hspace=0.35, wspace=0.2,
                           left=0.03, right=0.98, top=0.88, bottom=0.06)
    tw = dict(color="white", fontsize=8, fontweight="bold", pad=4)
    steps = [
        (0, 0, bgr_to_rgb(original), "Input f(x,y)"),
        (0, 1, compute_magnitude_spectrum(original), "FFT |F(u,v)|"),
        (0, 2, bgr_to_rgb(after_d["lp"]), "LP smooth base"),
        (0, 3, bgr_to_rgb(after_d["bp"]), "BP mid-band"),
        (0, 4, bgr_to_rgb(after_d["hp"]), "HP detail"),
        (1, 0, after_d["skin_mask"], "HSV skin mask"),
        (1, 1, after_d["flaw_mask"], "Auto flaw mask"),
        (1, 2, after_d.get("wrinkle_map", after_d["smooth_layer"]), "Wrinkle strength map"),
        (1, 3, bgr_to_rgb(after_d["image"]), "Final output"),
        (1, 4, compute_magnitude_spectrum(after_d["image"]), "Output spectrum"),
    ]
    for row, col, data, title in steps:
        ax = fig.add_subplot(gs[row, col])
        cmap = "inferno" if data.ndim == 2 and col in (1, 4) else ("gray" if data.ndim == 2 else None)
        ax.imshow(data, cmap=cmap)
        ax.set_title(title, **tw)
        ax.axis("off")
    fig.suptitle(f"Pipeline — {prof.name} — {name}", color="white", fontsize=13, y=0.97)
    path = os.path.join(out_dir, f"{name}_pipeline.png")
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def save_hist_figure(name, original, before_img, after_img, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor("#0f0f14")
    for ax, img, title, col in zip(axes, [original, before_img, after_img],
                                    ["Original", "Before", "After"],
                                    ["#5588ff", "#ffaa33", "#33dd88"]):
        ax.hist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).ravel(), bins=256,
                range=(0, 255), color=col, alpha=0.8)
        ax.set_title(title, color="white")
        ax.set_facecolor("#1a1a28")
        ax.tick_params(colors="gray")
    path = os.path.join(out_dir, f"{name}_histograms.png")
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _face_skin_region(skin_mask: np.ndarray, chin_frac: float = 0.96) -> np.ndarray:
    """Skin pixels above the neck (tiny trim only — used for offset / chin anchor)."""
    skin = skin_mask > 128
    ys, xs = np.where(skin)
    if len(xs) < 50:
        return skin
    y_top = int(ys.min())
    y_chin = y_top + int((int(ys.max()) - y_top) * chin_frac)
    return skin & (np.arange(skin_mask.shape[0], dtype=np.int32)[:, None] <= y_chin)


def _skin_norm_offset(geom: FaceGeometry, skin_mask: np.ndarray) -> Tuple[float, float]:
    """Skin centroid minus DNN bbox center, in normalized face units."""
    ys, xs = np.where(skin_mask > 128)
    if len(xs) < 50:
        return 0.0, 0.0
    return (
        (float(xs.mean()) - geom.cx) / max(geom.fw, 1),
        (float(ys.mean()) - geom.cy) / max(geom.fh, 1),
    )


def _soft_jaw_taper_f(
    shape: Tuple[int, int],
    geom: FaceGeometry,
    skin_mask: np.ndarray,
    cx: int,
    chin_y: int,
) -> np.ndarray:
    """Soft triangular jaw: wide at cheek level, tapering to a rounded chin point."""
    h, w = shape
    dnx, dny = _skin_norm_offset(geom, skin_mask)
    left = geom.point(0.22 + dnx, 0.80 + dny)
    right = geom.point(0.78 + dnx, 0.80 + dny)
    chin = geom.point(0.50 + dnx, 0.90 + dny)
    chin_y = min(chin_y, int(chin[1]))
    taper_y0 = int(geom.point(0.50 + dnx, 0.68 + dny)[1])

    tri = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(tri, np.array([left, right, chin], dtype=np.int32), 255)
    tri_f = cv2.GaussianBlur(tri.astype(np.float32), (27, 27), 0) / 255.0

    yy = np.arange(h, dtype=np.float32)[:, None]
    jaw_hw = max(int(geom.fw * 0.44), int(abs(right[0] - left[0]) // 2), 1)
    t = np.clip((yy - taper_y0) / max(chin_y - taper_y0, 1), 0.0, 1.0)
    half_w = jaw_hw * (1.0 - 0.90 * t)
    xx = np.arange(w, dtype=np.float32)[None, :]
    trap = (np.abs(xx - cx) <= half_w).astype(np.float32)
    trap = np.maximum(trap, (yy <= taper_y0).astype(np.float32))
    trap *= (yy <= chin_y + 3).astype(np.float32)
    trap = cv2.GaussianBlur(trap, (21, 21), 0)
    return np.clip(np.maximum(tri_f, trap), 0.0, 1.0)


def _skin_face_oval_u8(
    shape: Tuple[int, int], geom: FaceGeometry, skin_mask: np.ndarray,
) -> np.ndarray:
    """Full face oval (original size) with a soft triangular jaw instead of a round neck spill."""
    h, w = shape
    skin = skin_mask > 128
    ys, xs = np.where(skin)
    dnx, dny = _skin_norm_offset(geom, skin_mask)
    if len(xs) >= 50:
        cx = int(xs.mean())
        cy = int(ys.mean())
        x_span = int(xs.max() - xs.min())
        y_span = int(ys.max() - ys.min())
        rx = max(int(x_span * 0.56), int(geom.fw * 0.50), 1)
        ry = max(int(y_span * 0.56), int(geom.fh * 0.54), 1)
        chin_y = int(geom.point(0.50 + dnx, 0.90 + dny)[1])
    else:
        cx, cy = geom.cx, geom.cy
        rx = max(1, int(geom.fw * 0.50))
        ry = max(1, int(geom.fh * 0.54))
        chin_y = int(geom.point(0.50 + dnx, 0.90 + dny)[1])

    oval = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(oval, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
    jaw = _soft_jaw_taper_f(shape, geom, skin_mask, cx, chin_y)
    combined = (oval.astype(np.float32) / 255.0) * jaw
    return np.clip(combined * 255.0, 0, 255).astype(np.uint8)


def _heatmap_face_mask(
    shape: Tuple[int, int], geom: FaceGeometry, skin_mask: np.ndarray,
) -> np.ndarray:
    """Feathered skin-aligned oval — same region as WOMAN beautification blend."""
    oval = _skin_face_oval_u8(shape, geom, skin_mask)
    return cv2.GaussianBlur(oval.astype(np.float32), (31, 31), 0) / 255.0


def _mask_diff_to_face(diff_bgr: np.ndarray, face_m: np.ndarray) -> np.ndarray:
    """Restrict diff to feathered face oval; dilate so cheeks/forehead are covered."""
    gray = (
        cv2.cvtColor(diff_bgr, cv2.COLOR_BGR2GRAY)
        if diff_bgr.ndim == 3 else diff_bgr.copy()
    )
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    spread = cv2.dilate(gray, k, iterations=1)
    return np.clip(spread.astype(np.float32) * face_m, 0, 255).astype(np.uint8)


def save_difference_map(name, original, before_img, after_img, out_dir, analysis=None):
    def diff(a, b):
        return np.clip(np.abs(a.astype(np.int16) - b.astype(np.int16)) * 4, 0, 255).astype(np.uint8)

    d_b, d_a = diff(original, before_img), diff(original, after_img)

    if analysis is not None:
        face_m = _heatmap_face_mask(original.shape[:2], analysis.geom, analysis.skin_mask)
        d_b = _mask_diff_to_face(d_b, face_m)
        d_a = _mask_diff_to_face(d_a, face_m)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.patch.set_facecolor("#0f0f14")
    for row, col, data, title, cmap in [
        (0, 0, bgr_to_rgb(original), "Original", None),
        (0, 1, bgr_to_rgb(before_img), "Before", None),
        (0, 2, bgr_to_rgb(cv2.cvtColor(d_b, cv2.COLOR_GRAY2BGR)), "Diff before", None),
        (0, 3, d_b, "Heatmap", "hot"),
        (1, 0, bgr_to_rgb(original), "Original", None),
        (1, 1, bgr_to_rgb(after_img), "After", None),
        (1, 2, bgr_to_rgb(cv2.cvtColor(d_a, cv2.COLOR_GRAY2BGR)), "Diff after", None),
        (1, 3, d_a, "Heatmap", "hot"),
    ]:
        if cmap:
            axes[row, col].imshow(data, cmap=cmap, vmin=0, vmax=255)
        else:
            axes[row, col].imshow(data, cmap=cmap)
        axes[row, col].set_title(title, color="white", fontsize=9, fontweight="bold")
        axes[row, col].axis("off")
    path = os.path.join(out_dir, f"{name}_difference_map.png")
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def save_spectrum_figure(name, original, before_img, after_img, out_dir):
    specs, profiles = [], []
    for img in [original, before_img, after_img]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
        mag = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
        specs.append((np.log1p(mag) / np.log1p(mag).max() * 255).astype(np.uint8))
        profiles.append(radial_profile_vectorized(mag))

    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor("#0f0f14")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3,
                           left=0.06, right=0.97, top=0.90, bottom=0.08)
    for col, (spec, title) in enumerate(zip(specs, ["Original", "Before", "After"])):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(spec, cmap="inferno")
        ax.set_title(f"Spectrum: {title}", color="white", fontsize=10, fontweight="bold")
        ax.axis("off")

    ax = fig.add_subplot(gs[1, :])
    ax.set_facecolor("#1a1a28")
    x = np.arange(len(profiles[0]))
    for p, c, lbl in zip(profiles, ["#5588ff", "#ffaa33", "#33dd88"],
                         ["Original", "Before", "After"]):
        ax.plot(x, p, color=c, lw=1.5, label=lbl)
    ax.set_title("Radial frequency energy (vectorised)", color="white", fontsize=10)
    ax.legend(facecolor="#1a1a28", labelcolor="white")
    ax.tick_params(colors="gray")
    path = os.path.join(out_dir, f"{name}_spectrum.png")
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def save_rgb_channel_figure(name, original, before_img, after_img, out_dir):
    ch_names, ch_colors = ["Blue", "Green", "Red"], ["#4488ff", "#44cc44", "#ff4444"]
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#0f0f14")
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.25,
                           left=0.05, right=0.97, top=0.93, bottom=0.05)
    for col, (img, ct) in enumerate(zip([original, before_img, after_img],
                                         ["Original", "Before", "After"])):
        for row, ci in enumerate(range(3)):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(img[:, :, ci], cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"{ct} — {ch_names[ci]}", color="white", fontsize=8)
            ax.axis("off")
    for ch_idx, cc in enumerate(ch_colors):
        ax = fig.add_subplot(gs[3, ch_idx])
        ax.set_facecolor("#1a1a28")
        for img, lbl, ls in [(original, "Orig", "-"), (before_img, "Before", "--"), (after_img, "After", ":")]:
            h, b = np.histogram(img[:, :, ch_idx].ravel(), bins=256, range=(0, 255))
            ax.plot(b[:-1], h, color=cc, linestyle=ls, lw=1.2, label=lbl)
        ax.legend(facecolor="#1a1a28", labelcolor="white", fontsize=8)
        ax.tick_params(colors="gray")
    path = os.path.join(out_dir, f"{name}_rgb_channels.png")
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────
# Metrics + report
# ─────────────────────────────────────────────────────────────
def psnr(img_a, img_b):
    mse = np.mean((img_a.astype(np.float64) - img_b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def ssim_channel(a, b):
    from skimage.metrics import structural_similarity
    score, _ = structural_similarity(clip_to_uint8(a), clip_to_uint8(b), full=True, data_range=255)
    return score


def compute_metrics(original, enhanced):
    return {
        "PSNR": psnr(original, enhanced),
        "SSIM": float(np.mean([ssim_channel(original[:, :, c], enhanced[:, :, c]) for c in range(3)])),
        "StdDev_orig": float(original.astype(np.float64).std()),
        "StdDev_enh": float(enhanced.astype(np.float64).std()),
    }


def build_report(face_data: list, out_dir: str) -> str:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        HRFlowable, Image as RLImage, PageBreak, Paragraph,
        SimpleDocTemplate, Spacer, Table, TableStyle,
    )

    pdf_path = os.path.join(out_dir, "report.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                            leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm)
    styles = getSampleStyleSheet()
    avail_w = A4[0] - 4 * cm
    title_style = ParagraphStyle("T", parent=styles["Title"], fontSize=18,
                                 textColor=colors.HexColor("#1a237e"))
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=14,
                        textColor=colors.HexColor("#283593"), spaceBefore=10)
    body = ParagraphStyle("B", parent=styles["BodyText"], fontSize=10,
                          leading=15, alignment=TA_JUSTIFY)
    code = ParagraphStyle("C", fontName="Courier", fontSize=8.5, leftIndent=1 * cm)
    caption_style = ParagraphStyle("Cap", parent=styles["Italic"], fontSize=9, alignment=TA_CENTER)

    def img_block(path, caption=None):
        out = [RLImage(path, width=avail_w, height=avail_w * 0.56)]
        if caption:
            out.append(Paragraph(caption, caption_style))
        return out

    story = [
        Spacer(1, 1 * cm),
        Paragraph("Homework 2: Frequency-Domain Facial Beautification", title_style),
        HRFlowable(width=avail_w, thickness=2, color=colors.HexColor("#3949ab")),
        Paragraph(
            "Pipeline: universal flaw_reduction — auto face analysis, neon-box before preview, "
            "adaptive line-flaw smoothing (bbox-relative), gradient LAB fix for red blemishes.",
            body),
        Paragraph("F(u,v)=FFT{f}; G=H·F; g=real{IFFT{G}}", code),
    ]

    for i, fd in enumerate(face_data):
        story.append(Paragraph(f"Results — {fd['name']} ({fd['profile_name']})", h1))
        for key, cap in [
            ("fig_professor", "Professor summary: original vs beautified, FFT equation, radial energy."),
            ("fig_comparison", "Full comparison with adaptive weight map."),
            ("fig_pipeline", "Pipeline steps and masks."),
            ("fig_filters", "Filter kernels H(u,v)."),
            ("fig_spec", "FFT spectra and vectorised radial profile."),
            ("fig_hist", "Histograms."),
            ("fig_diff", "Difference maps."),
            ("fig_rgb", "RGB channel analysis."),
        ]:
            story += img_block(fd[key], cap)
            story.append(Spacer(1, 0.15 * cm))
        mb, ma = fd["metrics_before"], fd["metrics_after"]
        mt = Table([
            ["Metric", "Before", "After"],
            ["PSNR", f"{mb['PSNR']:.2f}", f"{ma['PSNR']:.2f}"],
            ["SSIM", f"{mb['SSIM']:.4f}", f"{ma['SSIM']:.4f}"],
            ["Std-dev", f"{mb['StdDev_enh']:.2f}", f"{ma['StdDev_enh']:.2f}"],
        ], colWidths=[5 * cm, 5 * cm, 5 * cm])
        mt.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eaf6"))]))
        story.append(mt)
        if i < len(face_data) - 1:
            story.append(PageBreak())
    doc.build(story)
    return pdf_path



# ── FFT helpers & masks (disassembly lines 1234–1551) ─────────────────────────

def _soft_mask(mask_u8: np.ndarray, ksize: int) -> np.ndarray:
    return cv2.GaussianBlur(
        mask_u8.astype(np.float64), (ksize, ksize), 0,
    ) / 255.0


def _fft_lp(channel: np.ndarray, sigma: float) -> np.ndarray:
    pad = _padded_shape(channel.shape)
    return apply_frequency_filter(
        channel.astype(np.float64),
        make_gaussian_kernel(pad, sigma),
    )


def _fft_local_skin_field(channel: np.ndarray, sigma_fine: float, sigma_coarse: float) -> np.ndarray:
    fine = _fft_lp(channel, sigma_fine)
    coarse = _fft_lp(channel, sigma_coarse)
    return fine * 0.68 + coarse * 0.32


def _exclude_features(mask: np.ndarray, geom: FaceGeometry, shape: Tuple[int, int]) -> np.ndarray:
    out = mask.copy()
    for nx, ny, rx, ry in (
        (0.34, 0.40, 0.16, 0.10), (0.66, 0.40, 0.16, 0.10),
        (0.50, 0.54, 0.12, 0.12), (0.50, 0.72, 0.22, 0.09),
    ):
        zone = geom.ellipse_mask(shape, nx, ny, rx, ry)
        out = cv2.bitwise_and(out, cv2.bitwise_not(zone))
    return out


def _exclude_features_woman1(mask: np.ndarray, geom: FaceGeometry, shape: Tuple[int, int]) -> np.ndarray:
    out = _exclude_features(mask, geom, shape)
    for nx, ny, rx, ry in (
        (0.50, 0.66, 0.30, 0.11), (0.50, 0.76, 0.22, 0.10),
        (0.40, 0.63, 0.14, 0.09), (0.60, 0.63, 0.14, 0.09),
    ):
        zone = geom.ellipse_mask(shape, nx, ny, rx, ry)
        out = cv2.bitwise_and(out, cv2.bitwise_not(zone))
    return out


def _filter_small_flaw_blobs_compact(
    mask: np.ndarray, shape: Tuple[int, int], compact_only: bool = False,
) -> np.ndarray:
    return _filter_small_flaw_blobs(mask, shape, compact_only=compact_only)


def _freckle_mask_bandpass(L: np.ndarray, skin_mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    pad = _padded_shape(L.shape)
    h_bp = make_bandpass_kernel(pad, 4.0, 15.0)
    mid = apply_frequency_filter(L.astype(np.float64), h_bp)
    energy = np.abs(mid)
    skin = skin_mask > 0
    if not np.any(skin):
        return np.zeros(shape, dtype=np.uint8)
    thresh = float(np.percentile(energy[skin], 72))
    raw = (energy > thresh).astype(np.uint8) * 255
    raw = raw & skin.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, k, iterations=1)
    return _filter_small_flaw_blobs_compact(raw, shape, compact_only=False)


def _target_freckle_mask(
    flaw_mask: np.ndarray, bp_mask: np.ndarray, geom: FaceGeometry, shape: Tuple[int, int],
) -> np.ndarray:
    compact = _filter_small_flaw_blobs(flaw_mask, shape, compact_only=False)
    merged = cv2.bitwise_or(compact, bp_mask)
    return _exclude_features(merged, geom, shape)


def _load_image(path: str, max_size: int = MAX_IMAGE_SIZE) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    scale = min(max_size / max(h, w), 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


# ── WOMAN1 acne restoration ─────────────────────────────────────────────────

def _woman1_acne_mask(
    bgr: np.ndarray, skin_mask: np.ndarray, geom: FaceGeometry, shape: Tuple[int, int],
) -> np.ndarray:
    skin = skin_mask > 0
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    a_ch = lab[:, :, 1].astype(np.float32)
    l_ch = lab[:, :, 0].astype(np.float32)
    local_a = cv2.GaussianBlur(a_ch, (41, 41), 0)
    lap = np.abs(cv2.Laplacian(l_ch, cv2.CV_32F))
    texture = cv2.GaussianBlur(lap, (7, 7), 0)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    red_hsv = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 58, 58]), np.array([11, 255, 255])),
        cv2.inRange(hsv, np.array([169, 58, 58]), np.array([180, 255, 255])),
    )
    red_blemish = (
        (a_ch > local_a + 3.5) & (red_hsv > 0) & (texture > 2.0) & skin
    )
    raw = red_blemish.astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, k, iterations=1)
    raw = cv2.dilate(raw, k, iterations=1)
    raw = cv2.bitwise_and(raw, skin_mask)
    h, w = shape
    max_area = max(60, int(h * w * 0.0025))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    out = np.zeros_like(raw)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        compact = min(bw, bh) / max(bw, bh, 1)
        if not (8 <= area <= max_area):
            continue
        if compact < 0.25:
            continue
        out[labels == i] = 255
    return _exclude_features_woman1(out, geom, shape)


def _woman1_safe_skin_mask(analysis: FaceAnalysis, shape: Tuple[int, int]) -> np.ndarray:
    x1, y1, x2, y2 = analysis.geom.bbox
    face_skin = np.zeros(shape, dtype=np.uint8)
    face_skin[y1:y2, x1:x2] = analysis.skin_mask[y1:y2, x1:x2]
    skin_f = _soft_mask(face_skin, 29)
    mouth_zone = analysis.geom.ellipse_mask(shape, 0.50, 0.70, 0.28, 0.12)
    return skin_f * (1.0 - _soft_mask(mouth_zone, 21))


def restore_woman1_before(bgr: np.ndarray, analysis: FaceAnalysis) -> Tuple[np.ndarray, float]:
    skin_mask = analysis.skin_mask
    skin_f = _soft_mask(skin_mask, 27)
    skin_px = skin_mask > 0
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    sigma_lp = 12.0
    L_lp = _fft_lp(L, sigma_lp)
    L_dn = L * 0.52 + L_lp * 0.48
    if np.any(skin_px):
        lo, hi = np.percentile(L_dn[skin_px], (2.0, 98.0))
        L_norm = np.clip((L_dn - lo) / max(hi - lo, 1.0) * 255.0, 0, 255)
        blend = np.clip(skin_f * 0.32, 0.0, 0.32)
        L_out = L_dn * (1.0 - blend) + L_norm * blend
    else:
        L_out = L_dn
    a_lp = _fft_lp(a, 10.0)
    b_lp = _fft_lp(b, 10.0)
    chroma_dn = np.clip(skin_f * 0.22, 0.0, 0.22)
    a_out = a * (1.0 - chroma_dn) + a_lp * chroma_dn
    b_out = b * (1.0 - chroma_dn) + b_lp * chroma_dn
    lab_out = np.stack([
        np.clip(L_out, 0, 255), np.clip(a_out, 0, 255), np.clip(b_out, 0, 255),
    ], axis=2)
    restored = cv2.cvtColor(clip_to_uint8(lab_out), cv2.COLOR_LAB2BGR).astype(np.float64)
    skin3 = np.stack([skin_f, skin_f, skin_f], axis=2)
    result = clip_to_uint8(
        bgr.astype(np.float64) * (1.0 - skin3) + restored * skin3,
    )
    return result, sigma_lp


def beautify_woman1_fft(bgr: np.ndarray, analysis: FaceAnalysis) -> Tuple[np.ndarray, float]:
    shape = bgr.shape[:2]
    safe_skin = _woman1_safe_skin_mask(analysis, shape)
    skin_pixels = analysis.skin_mask > 0
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    a_ref, b_ref, L_ref = a.copy(), b.copy(), L.copy()
    if np.any(skin_pixels):
        med_a = float(np.median(a[skin_pixels]))
        med_b = float(np.median(b[skin_pixels]))
        med_l = float(np.median(L[skin_pixels]))
        a_ref[~skin_pixels] = med_a
        b_ref[~skin_pixels] = med_b
        L_ref[~skin_pixels] = med_l
    a_tone = _fft_local_skin_field(a_ref, 9.0, 24.0)
    b_tone = _fft_local_skin_field(b_ref, 9.0, 26.0)
    L_tone = _fft_local_skin_field(L_ref, 9.0, 28.0)
    a_local = _fft_local_skin_field(a, 8.0, 18.0)
    red_map = np.clip((a - a_local - 0.4) / 6.5, 0.0, 1.0) * safe_skin
    chroma_w = np.clip(red_map * 0.88, 0.0, 0.88)
    a_out = a * (1.0 - chroma_w) + a_tone * chroma_w
    b_out = b * (1.0 - chroma_w * 0.38) + b_tone * (chroma_w * 0.38)
    pad = _padded_shape(a.shape)
    a_lp = apply_frequency_filter(a_out, make_gaussian_kernel(pad, 16.0))
    b_lp = apply_frequency_filter(b_out, make_gaussian_kernel(pad, 18.0))
    lp_w = np.clip(red_map * 0.35, 0.0, 0.35) * safe_skin
    a_out = a_out * (1.0 - lp_w) + a_lp * lp_w
    b_out = b_out * (1.0 - lp_w * 0.5) + b_lp * (lp_w * 0.5)
    L_lift = np.clip(red_map * 0.38, 0.0, 0.38)
    L_clear = np.maximum(L_tone, L)
    L_out = L * (1.0 - L_lift) + L_clear * L_lift
    L_smooth = _fft_local_skin_field(L_out, 10.0, 30.0)
    smooth_w = np.clip(safe_skin * 0.26, 0.0, 0.26)
    L_out = L_out * (1.0 - smooth_w) + L_smooth * smooth_w
    L_low = _fft_lp(L_out, 18.0)
    L_hi = L_out - L_low
    lift = np.clip(safe_skin * 0.48, 0.0, 0.48)
    L_bright = np.clip(L_low * 1.13 + 15.0, 0, 255)
    L_out = L_bright * lift + L_hi * 0.95 * lift + L_out * (1.0 - lift)
    pad = _padded_shape(L_out.shape)
    L_hp = apply_frequency_filter(L_out, make_highpass_kernel(pad, 14.0))
    geom = analysis.geom
    eye_zone = geom.ellipse_mask(shape, 0.34, 0.40, 0.14, 0.08)
    eye_zone = cv2.bitwise_or(eye_zone, geom.ellipse_mask(shape, 0.66, 0.40, 0.14, 0.08))
    glow_w = _soft_mask(eye_zone, 15) * 0.18
    L_out = np.clip(L_out + L_hp * glow_w, 0, 255)
    lab_out = np.stack([
        np.clip(L_out, 0, 255), np.clip(a_out, 0, 255), np.clip(b_out, 0, 255),
    ], axis=2)
    processed = cv2.cvtColor(clip_to_uint8(lab_out), cv2.COLOR_LAB2BGR).astype(np.float64)
    safe3 = np.stack([safe_skin, safe_skin, safe_skin], axis=2)
    result = clip_to_uint8(
        bgr.astype(np.float64) * (1.0 - safe3) + processed * safe3,
    )
    return result, 20.0


# ── WOMAN freckle / porcelain pipeline (updated fixes) ──────────────────────

def _woman_skin_mask_u8(
    geom: FaceGeometry, skin_mask: np.ndarray, shape: Tuple[int, int],
) -> np.ndarray:
    k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    face = _skin_face_oval_u8(shape, geom, skin_mask)
    x1, y1, x2, y2 = geom.bbox
    roi = np.zeros(shape, dtype=np.uint8)
    pad = max(2, int(0.015 * (x2 - x1)))
    roi[max(0, y1 - pad):min(shape[0], y2 + pad),
        max(0, x1 - pad):min(shape[1], x2 + pad)] = 255
    face = cv2.bitwise_and(face, roi)
    inner = cv2.erode(face, k9, iterations=1)
    border = cv2.bitwise_and(face, cv2.bitwise_not(inner))
    skin_near = cv2.dilate(skin_mask, k9, iterations=3)
    border_keep = cv2.bitwise_and(border, skin_near)
    return cv2.bitwise_or(inner, border_keep)


def _woman_skin_weights(
    analysis: FaceAnalysis, shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    geom = analysis.geom
    base_u8 = _woman_skin_mask_u8(geom, analysis.skin_mask, shape)
    composite_w = _soft_mask(base_u8, 41)
    dist = cv2.distanceTransform(base_u8, cv2.DIST_L2, 5)
    t = np.clip(dist / 24.0, 0.0, 1.0)
    edge = t * t * (3.0 - 2.0 * t)
    composite_w = composite_w * edge
    process_w = composite_w.copy()
    lum = cv2.cvtColor(analysis.bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    bg = (lum < 36.0) & (base_u8 == 0)
    composite_w[bg] = 0.0
    process_w[bg] = 0.0
    return np.clip(composite_w, 0.0, 1.0), np.clip(process_w, 0.0, 1.0)


def _woman_brow_mask(
    geom: FaceGeometry, shape: Tuple[int, int], skin_mask: np.ndarray | None = None,
) -> np.ndarray:
    dnx, dny = _skin_norm_offset(geom, skin_mask) if skin_mask is not None else (0.0, 0.0)
    out = np.zeros(shape, dtype=np.float64)
    for nx in (0.34 + dnx, 0.66 + dnx):
        brow = _soft_mask(
            geom.ellipse_mask(shape, nx, 0.334 + dny, 0.078, 0.022), 5,
        )
        out = np.clip(out + brow, 0.0, 1.0)
    return out


def _woman_freckle_mask(bgr: np.ndarray, skin_mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    L = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float64)
    return _freckle_mask_bandpass(L, skin_mask, shape)


def _woman_freckle_target_mask(
    bgr: np.ndarray,
    skin_mask: np.ndarray,
    flaw_mask: np.ndarray,
    geom: FaceGeometry,
    shape: Tuple[int, int],
) -> np.ndarray:
    bp = _woman_freckle_mask(bgr, skin_mask, shape)
    compact = _filter_small_flaw_blobs(flaw_mask, shape, compact_only=False)
    merged = cv2.bitwise_or(compact, bp)
    merged = _exclude_features(merged, geom, shape)
    for nx in (0.34, 0.66):
        brow = geom.ellipse_mask(shape, nx, 0.334, 0.078, 0.028)
        merged = cv2.bitwise_and(merged, cv2.bitwise_not(brow))
    for nx in (0.11, 0.89):
        ear = geom.ellipse_mask(shape, nx, 0.48, 0.055, 0.11)
        merged = cv2.bitwise_and(merged, cv2.bitwise_not(ear))
    dnx, dny = _skin_norm_offset(geom, skin_mask)
    face_inner = geom.ellipse_mask(shape, 0.50 + dnx, 0.52 + dny, 0.40, 0.45)
    merged = cv2.bitwise_and(merged, face_inner)
    return merged


def beautify_woman_fft(bgr: np.ndarray, analysis: FaceAnalysis) -> Tuple[np.ndarray, float]:
    shape = bgr.shape[:2]
    geom = analysis.geom
    dnx, dny = _skin_norm_offset(geom, analysis.skin_mask)
    composite_w, process_w = _woman_skin_weights(analysis, shape)
    skin_px = composite_w > 0.12
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    pad = _padded_shape(L.shape)
    h_bp = make_bandpass_kernel(pad, 4.0, 15.0)
    L_mid = apply_frequency_filter(L, h_bp)
    freckle_e = np.abs(L_mid)
    skin_proc = process_w > 0.08
    f_thresh = float(np.percentile(freckle_e[skin_proc], 72)) if np.any(skin_proc) else 999.0
    freckle_w = np.clip((freckle_e - f_thresh) / max(f_thresh * 0.55, 1.0), 0.0, 1.0)
    freckle_w = cv2.GaussianBlur(freckle_w.astype(np.float32), (0, 0), 2.2).astype(np.float64)
    freckle_w = np.clip(freckle_w * process_w * 0.62, 0.0, 0.62)
    L_clear = np.maximum(_fft_local_skin_field(L, 7.0, 22.0), L - 3.0)
    a_clear = _fft_local_skin_field(a, 7.0, 22.0)
    b_clear = _fft_local_skin_field(b, 7.0, 24.0)
    L = L * (1.0 - freckle_w) + L_clear * freckle_w
    a = a * (1.0 - freckle_w * 0.50) + a_clear * (freckle_w * 0.50)
    b = b * (1.0 - freckle_w * 0.45) + b_clear * (freckle_w * 0.45)
    L_ref, a_ref, b_ref = L.copy(), a.copy(), b.copy()
    if np.any(skin_px):
        med_l = float(np.median(L[skin_px]))
        med_a = float(np.median(a[skin_px]))
        med_b = float(np.median(b[skin_px]))
        L_ref[~skin_px], a_ref[~skin_px], b_ref[~skin_px] = med_l, med_a, med_b
    a_tone = _fft_local_skin_field(a_ref, 8.0, 24.0)
    b_tone = _fft_local_skin_field(b_ref, 8.0, 26.0)
    sigma_lp = 20.0
    L_low = _fft_lp(L, sigma_lp)
    L_high = L - L_low
    L_bright = np.clip(L_low * 1.18 + 22.0, 0, 255)
    bright_w = np.clip(process_w * 0.74, 0.0, 0.74)
    L_out = (L_bright + L_high * 0.96) * bright_w + L * (1.0 - bright_w)
    a_tgt = a_tone * 0.72 + 131.0 * 0.28
    b_tgt = b_tone * 0.70 + 134.0 * 0.30
    chroma_w = np.clip(process_w * 0.50, 0.0, 0.50)
    a_out = a * (1.0 - chroma_w) + a_tgt * chroma_w
    b_out = b * (1.0 - chroma_w) + b_tgt * chroma_w
    L_smooth = _fft_local_skin_field(L_out, 11.0, 32.0)
    smooth_w = np.clip(process_w * 0.40, 0.0, 0.40)
    L_out = L_out * (1.0 - smooth_w) + L_smooth * smooth_w
    lower_face = geom.ellipse_mask(shape, 0.50 + dnx, 0.78 + dny, 0.34, 0.12)
    lower_px = (lower_face > 0) & (process_w > 0.10)
    target_l = (
        float(np.percentile(L_out[lower_px], 72))
        if np.any(lower_px)
        else float(np.percentile(L_out[skin_px], 76)) if np.any(skin_px) else 200.0
    )
    L_coarse = _fft_local_skin_field(L_out, 16.0, 38.0)
    even_w = np.clip(process_w * 0.20, 0.0, 0.20)
    L_out = L_out * (1.0 - even_w) + (L_coarse * 0.48 + target_l * 0.52) * even_w
    L_hi_orig = L - _fft_lp(L, 5.5)
    L_out = L_out + L_hi_orig * process_w * 0.34
    eye_sharp = np.zeros(shape, dtype=np.float64)
    for nx in (0.34 + dnx, 0.66 + dnx):
        eye_sharp = np.clip(
            eye_sharp + _soft_mask(
                geom.ellipse_mask(shape, nx, 0.408 + dny, 0.062, 0.034), 5,
            ),
            0.0, 1.0,
        )
    L_hp = apply_frequency_filter(L_out, make_highpass_kernel(pad, 9.0))
    L_out = np.clip(L_out + L_hp * eye_sharp * 0.14, 0, 255)
    brow_zone = _woman_brow_mask(geom, shape, analysis.skin_mask)
    L_orig = lab[:, :, 0]
    L_local_o = cv2.GaussianBlur(L_orig, (0, 0), 7.0)
    hair_w = np.clip((L_local_o - L_orig + 2.5) / 9.0, 0.0, 1.0) * brow_zone
    hair_w = cv2.GaussianBlur(hair_w.astype(np.float32), (0, 0), 2.0).astype(np.float64)
    hair_w = np.clip(hair_w * 0.85, 0.0, 0.85)
    L_dark = np.clip(L_orig * 0.40 + 10.0, 0, 255)
    L_out = L_out * (1.0 - hair_w) + L_dark * hair_w
    a_dark = a * 0.65 + 127.0 * 0.35
    b_dark = b * 0.65 + 102.0 * 0.35
    a_out = a_out * (1.0 - hair_w * 0.50) + a_dark * (hair_w * 0.50)
    b_out = b_out * (1.0 - hair_w * 0.50) + b_dark * (hair_w * 0.50)
    lab_out = np.stack([
        np.clip(L_out, 0, 255), np.clip(a_out, 0, 255), np.clip(b_out, 0, 255),
    ], axis=2)
    processed = cv2.cvtColor(clip_to_uint8(lab_out), cv2.COLOR_LAB2BGR).astype(np.float64)
    w3 = np.stack([composite_w, composite_w, composite_w], axis=2)
    delta = processed - bgr.astype(np.float64)
    result = clip_to_uint8(bgr.astype(np.float64) + delta * w3)
    return result, sigma_lp


# ── Subject orchestration ───────────────────────────────────────────────────

def _build_woman1_before_dict(analysis: FaceAnalysis) -> dict:
    bgr = analysis.bgr
    profile = analysis.profile
    shape = bgr.shape[:2]
    restored, sigma_lp = restore_woman1_before(bgr, analysis)
    ycrcb = cv2.cvtColor(restored, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0].astype(np.float64)
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    pad = _padded_shape(y.shape)
    y_lp = apply_frequency_filter(y, make_gaussian_kernel(pad, sigma_lp))
    return {
        "image": restored,
        "lp": _ycrcb_to_bgr(y_lp, cr, cb),
        "H_LP": make_gaussian_kernel(shape, sigma_lp),
        "profile": profile,
        "analysis": analysis,
    }


def _build_after_dict_custom(
    original: np.ndarray,
    result: np.ndarray,
    analysis: FaceAnalysis,
    fft_sigma: float,
    mode: str = "lp",
) -> dict:
    from dataclasses import replace as dc_replace

    profile = analysis.profile
    shape = original.shape[:2]
    skin_mask = analysis.skin_mask
    flaw_mask = analysis.flaw_mask
    wrinkle_map = analysis.wrinkle_map
    ycrcb = cv2.cvtColor(original, cv2.COLOR_BGR2YCrCb).astype(np.float64)
    y, cr, cb = ycrcb[:, :, 0], ycrcb[:, :, 1], ycrcb[:, :, 2]
    viz_profile = dc_replace(profile, fft_sigma=fft_sigma)
    layers = decompose_frequency_layers(y, viz_profile)
    y_smooth = layers["low"]
    if profile.mid_attenuate > 0:
        y_smooth = y_smooth - profile.mid_attenuate * layers["mid_residual"]
    hp_vis = layers["detail"] - layers["mid_residual"]
    h_lp = make_gaussian_kernel(shape, fft_sigma)
    h_hp = make_highpass_kernel(shape, fft_sigma if mode == "hp" else 10.0)
    return {
        "image": result,
        "lp": _ycrcb_to_bgr(layers["low"], cr, cb),
        "bp": _ycrcb_to_bgr(layers["mid"] + 128, cr, cb),
        "hp": _ycrcb_to_bgr(np.clip(hp_vis + 128, 0, 255), cr, cb),
        "smooth_layer": _ycrcb_to_bgr(y_smooth, cr, cb),
        "flaw_mask": flaw_mask,
        "skin_mask": skin_mask,
        "wrinkle_map": (
            clip_to_uint8(wrinkle_map * 255) if wrinkle_map is not None else skin_mask
        ),
        "H_LP": h_lp if mode == "lp" else make_gaussian_kernel(shape, profile.before_sigma_lp),
        "H_BP": make_bandpass_kernel(shape, profile.sigma_mid_lo, profile.sigma_mid_hi),
        "H_HP": h_hp,
        "profile": profile,
        "analysis": analysis,
    }


def _build_subjects() -> list:
    subjects: list = []
    woman1_path = os.path.join(BASE_DIR, "images", "WOMAN1.jpg")
    if not os.path.isfile(woman1_path):
        woman1_path = os.path.join(BASE_DIR, "WOMAN1.jpg")
    if os.path.isfile(woman1_path):
        subjects.append((
            "WOMAN1", woman1_path, PROFILES["red_blemish"], beautify_woman1_fft, "lp",
        ))
    woman_path = os.path.join(BASE_DIR, "images", "WOMAN.jpg")
    if not os.path.isfile(woman_path):
        woman_path = os.path.join(BASE_DIR, "WOMAN.jpg")
    if os.path.isfile(woman_path):
        subjects.append((
            "WOMAN", woman_path, PROFILES["red_blemish"], beautify_woman_fft, "lp",
        ))
    return subjects


def process_subject_custom(
    stem: str,
    image_path: str,
    profile_override: BeautyProfile | None,
    beautify_fn,
    fft_mode: str = "lp",
) -> dict:
    print(f"\nAnalyzing: {stem} ({os.path.basename(image_path)}) ...")
    original = _load_image(image_path)
    analysis = analyze_portrait(original, stem)
    if profile_override is not None and analysis.profile != profile_override:
        flaw_mask = detect_skin_flaws(
            analysis.bgr, analysis.skin_mask, profile_override, analysis.geom,
        )
        wrinkle_map = (
            build_wrinkle_strength_map(
                analysis.bgr, analysis.skin_mask, profile_override, analysis.geom,
            )
            if is_line_flaw(profile_override)
            else None
        )
        analysis = FaceAnalysis(
            bgr=analysis.bgr,
            geom=analysis.geom,
            skin_mask=analysis.skin_mask,
            profile=profile_override,
            flaw_mask=flaw_mask,
            wrinkle_map=wrinkle_map,
        )
        boxes, color = collect_enhancement_boxes(analysis)
        analysis.enhancement_boxes = boxes
        analysis.annotate_bgr = color
    if stem == "WOMAN1":
        flaw_mask = _woman1_acne_mask(
            analysis.bgr, analysis.skin_mask, analysis.geom, analysis.bgr.shape[:2],
        )
        analysis = FaceAnalysis(
            bgr=analysis.bgr,
            geom=analysis.geom,
            skin_mask=analysis.skin_mask,
            profile=analysis.profile,
            flaw_mask=flaw_mask,
            wrinkle_map=analysis.wrinkle_map,
        )
        boxes, color = collect_enhancement_boxes(analysis)
        analysis.enhancement_boxes = boxes
        analysis.annotate_bgr = color
    if stem == "WOMAN":
        flaw_mask = _woman_freckle_target_mask(
            analysis.bgr,
            analysis.skin_mask,
            analysis.flaw_mask,
            analysis.geom,
            analysis.bgr.shape[:2],
        )
        analysis = FaceAnalysis(
            bgr=analysis.bgr,
            geom=analysis.geom,
            skin_mask=analysis.skin_mask,
            profile=analysis.profile,
            flaw_mask=flaw_mask,
            wrinkle_map=analysis.wrinkle_map,
        )
        boxes, color = collect_enhancement_boxes(analysis)
        analysis.enhancement_boxes = boxes
        analysis.annotate_bgr = color
    profile = analysis.profile
    n_targets = len(analysis.enhancement_boxes)
    print(
        f"  Face {analysis.geom.fw}x{analysis.geom.fh}px  mode={profile.flaw_mode}"
        f"  targets={n_targets}",
    )
    targets_img = render_before_preview(analysis.bgr, analysis)
    if stem == "WOMAN1":
        before_d = _build_woman1_before_dict(analysis)
    else:
        before_d = enhance_before(analysis)
    after_img, fft_sigma = beautify_fn(original, analysis)
    after_d = _build_after_dict_custom(original, after_img, analysis, fft_sigma, mode=fft_mode)
    cv2.imwrite(os.path.join(OUT_DIR, f"{stem}_original.png"), original)
    cv2.imwrite(os.path.join(OUT_DIR, f"{stem}_before.png"), before_d["image"])
    cv2.imwrite(os.path.join(OUT_DIR, f"{stem}_targets.png"), targets_img)
    cv2.imwrite(os.path.join(OUT_DIR, f"{stem}_after.png"), after_d["image"])
    triptych = save_triptych_figure(stem, original, targets_img, after_d["image"], OUT_DIR)
    mb = compute_metrics(original, before_d["image"])
    ma = compute_metrics(original, after_d["image"])
    print(f"  Before - PSNR:{mb['PSNR']:.2f} dB  SSIM:{mb['SSIM']:.4f}")
    print(f"  After  - PSNR:{ma['PSNR']:.2f} dB  SSIM:{ma['SSIM']:.4f}  std:{ma['StdDev_enh']:.1f}")
    return {
        "name": stem,
        "profile_name": profile.name,
        "metrics_before": mb,
        "metrics_after": ma,
        "fig_professor": save_professor_figure(stem, original, after_d, OUT_DIR),
        "fig_comparison": save_comparison_figure(stem, original, before_d, after_d, OUT_DIR),
        "fig_pipeline": save_pipeline_figure(stem, original, after_d, OUT_DIR),
        "fig_filters": save_filter_figure(stem, before_d, after_d, OUT_DIR),
        "fig_hist": save_hist_figure(stem, original, before_d["image"], after_d["image"], OUT_DIR),
        "fig_diff": save_difference_map(
            stem, original, before_d["image"], after_d["image"], OUT_DIR,
            analysis=after_d.get("analysis"),
        ),
        "fig_spec": save_spectrum_figure(stem, original, before_d["image"], after_d["image"], OUT_DIR),
        "fig_rgb": save_rgb_channel_figure(stem, original, before_d["image"], after_d["image"], OUT_DIR),
        "fig_triptych": triptych,
    }


def main():
    print("=" * 60)
    print("HW2 — Frequency-Domain Facial Beautification (All Subjects)")
    print("  WOMAN1: restoration (before) + acne tone / smooth skin (after)")
    print("  WOMAN:  freckle FFT clear + porcelain lighten (after)")
    print("=" * 60)

    subjects = _build_subjects()
    if not subjects:
        print("ERROR: No images found. Place WOMAN1.jpg / WOMAN.jpg in")
        print(f"       the same folder as this script, or in: {IMG_DIR}")
        return

    face_data_list: list = []
    for stem, path, prof_override, fn, fft_mode in subjects:
        try:
            fd = process_subject_custom(stem, path, prof_override, fn, fft_mode)
            face_data_list.append(fd)
        except Exception as exc:
            print(f"  SKIP {stem}: {exc}")

    if face_data_list:
        print("\nBuilding PDF report...")
        pdf = build_report(face_data_list, OUT_DIR)
        print(f"Report saved: {pdf}")

    print(f"\nDone! All outputs in: {OUT_DIR}")



if __name__ == "__main__":
    main()