"""Cut the model out of the studio background (ISNet) and remove the white fringe."""
import sys, numpy as np, onnxruntime as ort, cv2
from PIL import Image

T = "/tmp/claude-0/-home-user--/803224e4-c7b3-564a-9a74-0c2f64db4169/scratchpad/"
sess = ort.InferenceSession(T + "tools/isnet.onnx", providers=["CPUExecutionProvider"])

def matte(rgb):
    x = cv2.resize(rgb, (1024, 1024), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / 1.0
    pred = sess.run(None, {sess.get_inputs()[0].name: x.transpose(2, 0, 1)[None]})[0][0, 0]
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-9)
    h, w = rgb.shape[:2]
    m = cv2.resize(pred, (w, h), interpolation=cv2.INTER_CUBIC)
    m = np.clip((m - 0.25) / 0.5, 0, 1)
    m = m * m * (3 - 2 * m)                                  # smoothstep for crisper edge
    return m

def guided(I, p, r, eps):
    """Grey guided filter (He et al.) using box filters."""
    mI = cv2.boxFilter(I, -1, (r, r)); mp = cv2.boxFilter(p, -1, (r, r))
    cov = cv2.boxFilter(I * p, -1, (r, r)) - mI * mp
    var = cv2.boxFilter(I * I, -1, (r, r)) - mI * mI
    A = cov / (var + eps); B = mp - A * mI
    return cv2.boxFilter(A, -1, (r, r)) * I + cv2.boxFilter(B, -1, (r, r))

def refine(rgb, a):
    """Snap the coarse matte to real edges at full resolution, then pull it in slightly."""
    s = rgb.shape[1] / 1000
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    r = max(3, int(6 * s)) | 1
    a2 = np.clip(guided(g, a.astype(np.float32), r, 1e-3), 0, 1)
    a2 = np.clip(guided(g, a2, max(3, int(2 * s)) | 1, 2e-4), 0, 1)
    k = max(1, int(1.0 * s))
    a2 = cv2.erode(a2, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    a2 = np.clip((a2 - 0.15) / 0.7, 0, 1)
    return cv2.GaussianBlur(a2, (0, 0), 0.6 * s)

def decontaminate(rgb, a):
    """Replace edge colours with colour bled outward from the solid interior (no halo, no dark rim)."""
    I = rgb.astype(np.float32)
    inner = (a > 0.98).astype(np.float32)
    F = I.copy()
    for sig in (2, 6, 18):
        s = sig * rgb.shape[1] / 1000
        num = cv2.GaussianBlur(I * inner[..., None], (0, 0), s)
        den = cv2.GaussianBlur(inner, (0, 0), s)[..., None]
        est = num / np.maximum(den, 1e-4)
        fill = (den > 0.02) & (F is not None)
        F = np.where((a[..., None] <= 0.98) & fill & ~np.isnan(est), est, F) if sig == 2 else \
            np.where((a[..., None] <= 0.98) & (np.abs(F - I).sum(-1, keepdims=True) == 0) & fill, est, F)
    return np.clip(F, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    im = np.asarray(Image.open(src).convert("RGB"))
    a = refine(im, matte(im))
    F = decontaminate(im, a)
    Image.fromarray(np.dstack([F, (a * 255 + 0.5).astype(np.uint8)])).save(dst)
    print(dst, im.shape)
