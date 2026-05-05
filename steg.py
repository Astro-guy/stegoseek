
import argparse
import math
import os
import struct
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy.stats import chi2

THRESHOLDS = {
    "rs_difference":         0.10,   # |Rm - Sm| ratio threshold
    "spa_embedding_rate":    0.05,   # estimated embedding fraction
    "dct_lsb_chi2_p_value":  0.05,
    "double_compress_score": 0.30,
    "histogram_anomaly":     0.15,
    "noise_residual_zscore": 3.5,
    "entropy_delta":         0.25,   # bits per symbol difference
}

WEIGHT = {                           # used in overall score (must sum to 1.0)
    "rs":               0.25,
    "spa":              0.20,
    "dct_lsb":          0.15,
    "double_compress":  0.12,
    "histogram":        0.12,
    "lsb_plane":        0.08,
    "noise_residual":   0.05,
    "entropy":          0.03,
}

VERDICT_THRESHOLD = 0.35             # weighted score → likely steganography



def load_image(path: str) -> Tuple[Image.Image, np.ndarray]:
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    return img, arr


def shannon_entropy(data: np.ndarray) -> float:
    counts = Counter(data.ravel().tolist())
    total = data.size
    return -sum((c / total) * math.log2(c / total) for c in counts.values() if c)


def lsb_plane(channel: np.ndarray) -> np.ndarray:
    return channel & 1


# ═══════════════════════════════════════════════════════════════════════════
# Test 1 – RS Analysis
# ═══════════════════════════════════════════════════════════════════════════

def _rs_discriminant(group: np.ndarray) -> int:
    """Smoothness measure: sum of |f(i+1) - f(i)|."""
    return int(np.sum(np.abs(np.diff(group.astype(np.int32)))))


def _flip_lsb(pixels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply F1 flipping mask to pixel values."""
    out = pixels.copy().astype(np.int32)
    for i, m in enumerate(mask):
        if m == 1:
            out[i] ^= 1          # flip LSB
        elif m == -1:            # F-1 (flip between -1 and 0 neighbourhood)
            out[i] = out[i] - 1 if out[i] % 2 == 1 else out[i] + 1
    return np.clip(out, 0, 255).astype(np.uint8)


def test_rs_analysis(arr: np.ndarray) -> Dict:
    """
    RS (Regular-Singular) analysis on luminance channel.
    Compares R, S counts with and without flipping to estimate embedding rate.
    """
    gray = np.mean(arr, axis=2).astype(np.uint8)
    h, w = gray.shape
    group_len = 4
    mask = np.array([1, 0, 1, 0])  # standard F1 mask

    Rm = Sm = Rm_ = Sm_ = 0
    groups_analysed = 0

    for row in range(h):
        for col in range(0, w - group_len + 1, group_len):
            g = gray[row, col:col + group_len]
            d_orig = _rs_discriminant(g)

            g_flip = _flip_lsb(g, mask)
            d_flip = _rs_discriminant(g_flip)

            g_neg = _flip_lsb(g, -mask)
            d_neg = _rs_discriminant(g_neg)

            if d_flip > d_orig:
                Rm += 1
            elif d_flip < d_orig:
                Sm += 1

            if d_neg > d_orig:
                Rm_ += 1
            elif d_neg < d_orig:
                Sm_ += 1

            groups_analysed += 1

    if groups_analysed == 0:
        return {"name": "RS Analysis", "suspicious": False, "score": 0.0, "detail": {}}

    rm = Rm / groups_analysed
    sm = Sm / groups_analysed
    rm_ = Rm_ / groups_analysed
    sm_ = Sm_ / groups_analysed

    diff = abs(rm - rm_) + abs(sm - sm_)
    suspicious = diff > THRESHOLDS["rs_difference"]
    score = min(1.0, diff / (THRESHOLDS["rs_difference"] * 2))

    return {
        "name": "RS Analysis",
        "suspicious": suspicious,
        "score": float(score),
        "detail": {
            "Rm": rm, "Sm": sm, "Rm_": rm_, "Sm_": sm_,
            "diff": diff
        }
    }


# ═══════════════════════════════════════════════════════════════════════════
# Test 2 – Sample Pair Analysis (SPA)
# ═══════════════════════════════════════════════════════════════════════════

def test_spa(arr: np.ndarray) -> Dict:
    """
    Sample Pair Analysis: estimates embedding rate from the asymmetry between
    pairs where (u mod 2, v mod 2) = (0,1) vs (1,0).
    """
    scores = []
    details = {}
    suspicious = False

    for c, name in enumerate(["R", "G", "B"]):
        ch = arr[:, :, c].astype(np.int32)
        u = ch[:, :-1].ravel()
        v = ch[:, 1:].ravel()

        # Count pairs in each category
        W  = np.sum((u % 2 == 0) & (v % 2 == 0))
        X  = np.sum((u % 2 == 1) & (v % 2 == 0))
        Y  = np.sum((u % 2 == 0) & (v % 2 == 1))
        Z  = np.sum((u % 2 == 1) & (v % 2 == 1))
        total = W + X + Y + Z + 1e-10

        # Embedding rate estimate (Dumitrescu et al.)
        p = X + Y
        q = W + Z
        if (2 * (p - q)) != 0:
            rate = (2 * Y - p) / (2 * (p - q)) if (2 * (p - q)) != 0 else 0.0
        else:
            rate = 0.0
        rate = max(0.0, min(1.0, abs(float(rate))))

        flag = rate > THRESHOLDS["spa_embedding_rate"]
        details[name] = {"estimated_rate": rate, "flag": flag}
        scores.append(rate / max(THRESHOLDS["spa_embedding_rate"], 0.01))
        if flag:
            suspicious = True

    score = min(1.0, float(np.mean(scores)))
    return {"name": "Sample Pair Analysis", "suspicious": suspicious, "score": score, "detail": details}


# ═══════════════════════════════════════════════════════════════════════════
# Test 3 – DCT Coefficient LSB Test (JPEG-aware)
# ═══════════════════════════════════════════════════════════════════════════


def test_dct_lsb(arr: np.ndarray) -> Dict:

    gray = (0.299 * arr[:, :, 0]
            + 0.587 * arr[:, :, 1]
            + 0.114 * arr[:, :, 2]).astype(np.float64)

    h, w = gray.shape

    # Pre-build 8×8 DCT matrix for fast transforms
    n = 8
    dct_mat = np.zeros((n, n))
    for u in range(n):
        for x in range(n):
            cu = (1 / math.sqrt(2)) if u == 0 else 1.0
            dct_mat[u, x] = cu * math.cos((2 * x + 1) * u * math.pi / 16) * 0.5
    # Full 2-D DCT: D @ block @ D.T
    dct_mat_t = dct_mat.T

    block_coords = [
        (r, c)
        for r in range(0, h - 7, 8)
        for c in range(0, w - 7, 8)
    ]
    rng = np.random.default_rng(42)
    if len(block_coords) > 2000:
        idxs = rng.choice(len(block_coords), 2000, replace=False)
        block_coords = [block_coords[i] for i in idxs]

    # Collect rounded AC coefficient magnitudes (skip DC [0,0])
    magnitudes = []
    for r, c in block_coords:
        block = gray[r:r + 8, c:c + 8] - 128.0
        dct_block = dct_mat @ block @ dct_mat_t
        for i in range(n):
            for j in range(n):
                if i == 0 and j == 0:
                    continue
                m = abs(int(round(dct_block[i, j])))
                if m >= 2:          # only use coefficients where LSB is meaningful
                    magnitudes.append(m)

    if len(magnitudes) < 200:
        return {
            "name": "DCT LSB Test",
            "suspicious": False,
            "score": 0.0,
            "detail": {"note": "Too few usable AC coefficients (image may be too small or flat)"},
        }

    mags = np.array(magnitudes)

    # Self-calibrated baseline: fraction of odd magnitudes in this image
    # (this IS the natural LSB=1 rate for coefficients of magnitude ≥ 2)
    natural_odd_rate = float(np.mean(mags % 2))

    # Observed LSB=1 rate
    observed_odd_rate = natural_odd_rate   # same data — we need a *split* sample

    # --- Split-sample calibration ---
    # Use first half to estimate baseline, second half to test.
    # This lets us detect a *shift* introduced by embedding on top of the
    # natural distribution.
    half = len(mags) // 2
    baseline_rate = float(np.mean(mags[:half] % 2))
    test_lsbs     = (mags[half:] % 2).astype(np.float64)
    n_test        = len(test_lsbs)
    observed_rate = float(np.mean(test_lsbs))

    # Binomial z-test: is the observed rate significantly different from baseline?
    std_err = math.sqrt(baseline_rate * (1 - baseline_rate) / n_test + 1e-12)
    z = abs(observed_rate - baseline_rate) / std_err

    # Two-tailed p-value from normal approximation (valid for large n)
    from scipy.stats import norm as sp_norm
    p_value = float(2 * (1 - sp_norm.cdf(z)))

    suspicious = p_value < THRESHOLDS["dct_lsb_chi2_p_value"]
    score = min(1.0, max(0.0, 1.0 - p_value / THRESHOLDS["dct_lsb_chi2_p_value"]))

    return {
        "name": "DCT LSB Test",
        "suspicious": suspicious,
        "score": float(score),
        "detail": {
            "usable_ac_coeffs":  len(mags),
            "baseline_odd_rate": round(baseline_rate, 5),
            "observed_odd_rate": round(observed_rate, 5),
            "z_score":           round(z, 4),
            "p_value":           round(p_value, 6),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# Test 4 – JPEG Double-Compression Ghost
# ═══════════════════════════════════════════════════════════════════════════

def test_double_compression(path: str, arr: np.ndarray) -> Dict:
    """
    Re-compress the image at two different quality levels and measure
    DCT histogram double-quantisation artefacts.  High scores suggest the
    image was previously compressed, then re-saved (common in JPEG stego).
    """
    import io
    from PIL import Image as PILImage

    img = PILImage.fromarray(arr)
    scores_q = []

    for q in [70, 85]:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        recomp = np.array(PILImage.open(buf).convert("RGB"), dtype=np.float64)
        orig_f = arr.astype(np.float64)
        diff = np.abs(orig_f - recomp).mean()
        # Natural single-compression: diff should be low if q matches original
        # Double-compression leaves ghost peaks → diff is erratic across q
        scores_q.append(diff)

    # Variance between quality-level diffs indicates double compression
    variance = float(np.var(scores_q))
    # Normalise to 0-1
    score = min(1.0, variance / (THRESHOLDS["double_compress_score"] * 50))
    suspicious = score > THRESHOLDS["double_compress_score"]

    return {
        "name": "JPEG Double-Compression",
        "suspicious": suspicious,
        "score": score,
        "detail": {"recompression_diffs": scores_q, "variance": variance}
    }


# ═══════════════════════════════════════════════════════════════════════════
# Test 5 – Histogram Anomaly
# ═══════════════════════════════════════════════════════════════════════════

def test_histogram_anomaly(arr: np.ndarray) -> Dict:
    """
    In a clean image, adjacent pixel values usually have smoothly varying
    counts.  LSB substitution equalises adjacent-pair counts → detectable as
    a comb-like pattern in the histogram.
    """
    results = {}
    suspicious = False
    scores = []

    for c, name in enumerate(["R", "G", "B"]):
        ch = arr[:, :, c].ravel()
        hist, _ = np.histogram(ch, bins=256, range=(0, 255))
        hist = hist.astype(np.float64)

        # Compute ratio of even-odd pairs
        even = hist[::2]
        odd  = hist[1::2]
        diffs = np.abs(even - odd) / (even + odd + 1e-10)
        comb_score = float(np.mean(diffs))

        flag = comb_score > THRESHOLDS["histogram_anomaly"]
        results[name] = {"comb_score": comb_score, "flag": flag}
        scores.append(min(1.0, comb_score / THRESHOLDS["histogram_anomaly"]))
        if flag:
            suspicious = True

    score = float(np.mean(scores))
    return {"name": "Histogram Anomaly", "suspicious": suspicious, "score": score, "detail": results}


# ═══════════════════════════════════════════════════════════════════════════
# Test 6 – LSB Plane Visualisation (optional debug output)
# ═══════════════════════════════════════════════════════════════════════════

def test_lsb_plane_visual(arr: np.ndarray, debug_dir: Optional[str]) -> Dict:
    """
    Isolate LSB planes.  Structured patterns (text, images, banding) are
    visible in stego images; clean images show near-random noise.
    Quantifies randomness via entropy.
    """
    results = {}
    suspicious = False
    scores = []

    for c, name in enumerate(["R", "G", "B"]):
        plane = lsb_plane(arr[:, :, c]) * 255
        ent = shannon_entropy(plane)
        # Random LSB plane entropy ≈ 1.0 (binary)
        # Structured content drops entropy significantly
        deviation = abs(ent - 1.0)
        flag = deviation > 0.15
        results[name] = {"lsb_entropy": float(ent), "deviation": float(deviation), "flag": flag}
        scores.append(min(1.0, deviation / 0.15))
        if flag:
            suspicious = True

        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            lsb_img = Image.fromarray(plane.astype(np.uint8), mode="L")
            lsb_img.save(os.path.join(debug_dir, f"lsb_plane_{name}.png"))

    score = float(np.mean(scores))
    return {"name": "LSB Plane Visual", "suspicious": suspicious, "score": score, "detail": results}


# ═══════════════════════════════════════════════════════════════════════════
# Test 7 – Noise Residual Analysis
# ═══════════════════════════════════════════════════════════════════════════

def test_noise_residual(arr: np.ndarray) -> Dict:
    """
    Compute noise residual by subtracting a median-filtered version of the
    image.  In smooth regions, stego noise is structured → high Z-score.
    """
    from scipy.ndimage import median_filter

    results = {}
    suspicious = False
    scores = []

    for c, name in enumerate(["R", "G", "B"]):
        ch = arr[:, :, c].astype(np.float64)
        smoothed = median_filter(ch, size=3).astype(np.float64)
        residual = ch - smoothed

        # Identify smooth regions (low local variance)
        h, w = ch.shape
        block_var = []
        for r in range(0, h - 7, 8):
            for col in range(0, w - 7, 8):
                block_var.append(float(np.var(ch[r:r+8, col:col+8])))
        if not block_var:
            continue
        bv = np.array(block_var)
        smooth_thresh = float(np.percentile(bv, 25))

        smooth_residuals = []
        for r in range(0, h - 7, 8):
            for col in range(0, w - 7, 8):
                if float(np.var(ch[r:r+8, col:col+8])) <= smooth_thresh:
                    smooth_residuals.extend(residual[r:r+8, col:col+8].ravel().tolist())

        if len(smooth_residuals) < 10:
            continue

        sr = np.array(smooth_residuals)
        z = float(np.abs(np.mean(sr)) / (np.std(sr) + 1e-10))
        flag = z > THRESHOLDS["noise_residual_zscore"]
        results[name] = {"z_score": z, "flag": flag}
        scores.append(min(1.0, z / THRESHOLDS["noise_residual_zscore"]))
        if flag:
            suspicious = True

    score = float(np.mean(scores)) if scores else 0.0
    return {"name": "Noise Residual", "suspicious": suspicious, "score": score, "detail": results}


# ═══════════════════════════════════════════════════════════════════════════
# Test 8 – Channel Entropy Comparison
# ═══════════════════════════════════════════════════════════════════════════

def test_channel_entropy(arr: np.ndarray) -> Dict:
    """
    LSB embedding raises entropy in channels that carry hidden data.
    Abnormally high or unbalanced channel entropy is a soft indicator.
    """
    entropies = {}
    for c, name in enumerate(["R", "G", "B"]):
        entropies[name] = shannon_entropy(arr[:, :, c])

    vals = list(entropies.values())
    mean_ent = float(np.mean(vals))
    delta = float(max(vals) - min(vals))

    # Natural images: channels very similar in entropy, all near 7-8 bits
    # Stego: may show one channel with elevated entropy
    suspicious = delta > THRESHOLDS["entropy_delta"] or mean_ent > 7.98
    score = min(1.0, delta / THRESHOLDS["entropy_delta"])

    return {
        "name": "Channel Entropy",
        "suspicious": suspicious,
        "score": float(score),
        "detail": {**entropies, "delta": delta, "mean": mean_ent}
    }


# ═══════════════════════════════════════════════════════════════════════════
# Test 9 – EOF / Appended Data
# ═══════════════════════════════════════════════════════════════════════════

def test_eof_data(path: str) -> Dict:
    """
    Check for data appended after the image's end-of-file marker.
    Works for JPEG (FF D9), PNG (IEND), and GIF (3B).
    """
    with open(path, "rb") as f:
        data = f.read()

    ext = Path(path).suffix.lower()
    suspicious = False
    appended_bytes = 0

    if ext in (".jpg", ".jpeg"):
        eof_marker = b"\xff\xd9"
        idx = data.rfind(eof_marker)
        if idx != -1:
            appended_bytes = len(data) - idx - 2
    elif ext == ".png":
        iend = b"IEND\xaeB`\x82"
        idx = data.rfind(iend)
        if idx != -1:
            appended_bytes = len(data) - idx - len(iend)
    elif ext == ".gif":
        # GIF terminator is 0x3B
        idx = data.rfind(b"\x3b")
        if idx != -1:
            appended_bytes = len(data) - idx - 1
    else:
        appended_bytes = 0

    suspicious = appended_bytes > 0
    score = 1.0 if suspicious else 0.0

    return {
        "name": "EOF / Appended Data",
        "suspicious": suspicious,
        "score": score,
        "detail": {"appended_bytes": appended_bytes}
    }


# ═══════════════════════════════════════════════════════════════════════════
# Aggregate & Report
# ═══════════════════════════════════════════════════════════════════════════

def compute_overall(results: List[Dict]) -> Tuple[float, str]:
    key_map = {
        "RS Analysis":            "rs",
        "Sample Pair Analysis":   "spa",
        "DCT LSB Test":           "dct_lsb",
        "JPEG Double-Compression":"double_compress",
        "Histogram Anomaly":      "histogram",
        "LSB Plane Visual":       "lsb_plane",
        "Noise Residual":         "noise_residual",
        "Channel Entropy":        "entropy",
        "EOF / Appended Data":    "eof",
    }
    weighted_sum = 0.0
    weight_total = 0.0
    for r in results:
        k = key_map.get(r["name"])
        # EOF gets boosted weight when triggered (high-confidence indicator)
        if r["name"] == "EOF / Appended Data" and r["suspicious"]:
            w = 0.50
        else:
            w = WEIGHT.get(k, 0.05)
        weighted_sum += r["score"] * w
        weight_total += w

    if weight_total == 0:
        return 0.0, "CLEAN"


    strong_tests = {
    "RS Analysis",
    "Sample Pair Analysis",
    "DCT LSB Test",
    "EOF / Appended Data"
    }

    any_strong_flag = any(
        r["suspicious"] and r["name"] in strong_tests
        for r in results
    )


    overall = weighted_sum / weight_total
    if overall >= VERDICT_THRESHOLD:
        verdict = "HIGH POSSIBILITY OF STEGANOGRAPHY"
    elif overall >= 0.16:
        verdict = "SUSPICIOUS"
    elif any_strong_flag:
        verdict = "SUSPICIOUS"
    else:
        verdict = "CLEAN"
    return overall, verdict


def print_report(path: str, results: List[Dict], overall: float, verdict: str) -> None:
    W = 70
    SEP = "─" * W
    print(f"\n{'═' * W}")
    print(f"  STEGANOGRAPHY DETECTION REPORT")
    print(f"  File: {path}")
    print(f"{'═' * W}")
    print(f"{'Test':<35} {'Score':>7}  {'Flags'}")
    print(SEP)
    for r in results:
        flag = " SUSPICIOUS" if r["suspicious"] else "  ok"
        print(f"  {r['name']:<33} {r['score']:>6.3f}  {flag}")
    print(SEP)
    verdict_line = f"  OVERALL SCORE: {overall:.3f}   VERDICT: {verdict}"
    print(verdict_line)
    print(f"{'═' * W}\n")

   

# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def analyse(image_path: str, debug_dir: Optional[str] = None) -> Tuple[float, str]:
    if not os.path.isfile(image_path):
        print(f"[ERROR] File not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Loading image: {image_path}")
    img, arr = load_image(image_path)
    print(f"    Size: {img.width}×{img.height}  Mode: {img.mode}")
    print("[*] Running tests …")

    results = []

    print("    1/9  RS Analysis …")
    results.append(test_rs_analysis(arr))

    print("    2/9  Sample Pair Analysis …")
    results.append(test_spa(arr))

    print("    3/9  DCT LSB Test …")
    results.append(test_dct_lsb(arr))

    print("    4/9  JPEG Double-Compression …")
    results.append(test_double_compression(image_path, arr))

    print("    5/9  Histogram Anomaly …")
    results.append(test_histogram_anomaly(arr))

    print("    6/9  LSB Plane Visualisation …")
    results.append(test_lsb_plane_visual(arr, debug_dir))

    print("    7/9  Noise Residual …")
    results.append(test_noise_residual(arr))

    print("    8/9  Channel Entropy …")
    results.append(test_channel_entropy(arr))

    print("    9/9  EOF / Appended Data …")
    results.append(test_eof_data(image_path))

    overall, verdict = compute_overall(results)
    print_report(image_path, results, overall, verdict)
    return overall, verdict






import random

def analyse_directory(folder_path: str, limit: int = None, random_sample: bool = False, debug_dir: Optional[str] = None):
    if not os.path.isdir(folder_path):
        print(f"[ERROR] Not a directory: {folder_path}")
        return

    files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ]

    if not files:
        print("[INFO] No valid image files found.")
        return

    # 🔹 Apply limit
    if limit is not None:
        if random_sample:
            files = random.sample(files, min(limit, len(files)))
        else:
            files = files[:limit]

    print(f"\n=== BATCH STEGANALYSIS ({len(files)} files) ===\n")

    for file in sorted(files):
        path = os.path.join(folder_path, file)

        try:
            img, arr = load_image(path)

            results = []
            results.append(test_rs_analysis(arr))
            results.append(test_spa(arr))
            results.append(test_dct_lsb(arr))
            results.append(test_double_compression(path, arr))
            results.append(test_histogram_anomaly(arr))
            results.append(test_lsb_plane_visual(arr, None))
            results.append(test_noise_residual(arr))
            results.append(test_channel_entropy(arr))
            results.append(test_eof_data(path))

            overall, verdict = compute_overall(results)

            # Extract reasons
            reasons = []
            for r in results:
                if r["suspicious"]:
                    reasons.append(r["name"])

            # Output
            print(f"{file}")
            print(f"  Verdict: {verdict} ({overall:.3f})")

            if reasons:
                print("  Reasons:")
                for r in reasons:
                    print(f"   - {r}")
            else:
                print("  Reasons: None")

            print("-" * 50)

        except Exception as e:
            print(f"[ERROR] {file}: {e}")







def main():
    parser = argparse.ArgumentParser(description="Steganography detection tool (no ML)")

    parser.add_argument("path", help="Path to image file OR directory")
    parser.add_argument("--debug-dir", help="Directory to save debug outputs", default=None)

    parser.add_argument("--limit", type=int, help="Limit number of files (for directory mode)", default=None)
    parser.add_argument("--random", action="store_true", help="Randomly sample files")

    args = parser.parse_args()

    if os.path.isdir(args.path):
        # Directory mode
        analyse_directory(
            args.path,
            limit=args.limit,
            random_sample=args.random,
            debug_dir=args.debug_dir
        )
    else:
        # Single image mode
        analyse(args.path, args.debug_dir)

if __name__ == "__main__":
    main()
