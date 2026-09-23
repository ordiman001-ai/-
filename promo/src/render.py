"""RUSSIA track-suit promo — vertical 9:16, beat-synced to music.wav (125 BPM).

usage: python3 render.py <scale> <out.mp4> [t0 t1]
  scale 1.0 -> 4320x7680 (8K vertical), 0.125 -> 540x960 preview
"""
import sys, os, math, subprocess, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from concurrent.futures import ProcessPoolExecutor

T = "/tmp/claude-0/-home-user--/803224e4-c7b3-564a-9a74-0c2f64db4169/scratchpad/"
SC = float(sys.argv[1]) if len(sys.argv) > 1 else 0.125
W, H = int(round(4320 * SC / 2) * 2), int(round(7680 * SC / 2) * 2)
FPS = 30
BEAT, BAR = 0.48, 1.92
DUR = 25.2
FONT_H = T + "tools/RussoOne.ttf"
FONT_B = T + "tools/Montserrat%5Bwght%5D.ttf"
cv2.setNumThreads(2)

# ------------------------------------------------------------------ utils
def clamp(x, a=0.0, b=1.0): return max(a, min(b, x))
def ease_out(p): p = clamp(p); return 1 - (1 - p) ** 3
def ease_in(p): p = clamp(p); return p ** 3
def ease_io(p): p = clamp(p); return 3 * p * p - 2 * p * p * p
def ease_back(p):
    p = clamp(p); c = 1.9; return 1 + (c + 1) * (p - 1) ** 3 + c * (p - 1) ** 2
def hexc(h): h = h.lstrip('#'); return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.float32)

def grade(rgb):
    """Contrast S-curve, cool navy shadows, a touch more saturation."""
    x = np.arange(256, dtype=np.float32) / 255
    s = x + 0.10 * np.sin(2 * np.pi * x) * -0.5 * (1)   # gentle S
    s = np.clip(0.5 + (s - 0.5) * 1.06, 0, 1)
    luts = [np.clip((s * 255) + o, 0, 255).astype(np.uint8) for o in (-2, 0, 4)]
    out = cv2.merge([cv2.LUT(c, l) for c, l in zip(cv2.split(rgb), luts)])
    hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.12, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

# ------------------------------------------------------------------ assets
PHOTOS = {}
CUTS = {}
CUT_LO = {"01": "lofull/01_front_full.png", "02": "lofull/02_front_jacket.png",
          "03": "lofull/03_back_full.png", "08": "lo/08_pants_side.png"}
def load_assets():
    # photos are loaded so their height is 1.35*H: enough for a 9:16 fill with room to push in
    for n in ["01_front_full", "02_front_jacket", "03_back_full", "04_chest_closeup", "05_back_print",
              "06_cuff", "07_waistband", "08_pants_side", "09_jacket_pocket", "10_shoulder_flag"]:
        im = np.asarray(Image.open(T + os.environ.get("UPDIR", "up/") + f"{n}.png").convert("RGB"))
        lo = np.asarray(Image.open(T + f"lo/{n}.png").convert("RGB"))
        if lo.shape[0] < im.shape[0]:   # keep 30% of the original texture under the AI upscale
            lo = cv2.resize(lo, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_CUBIC)
            im = cv2.addWeighted(im, 0.7, lo, 0.3, 0)
        f = min(1.0, 1.6 * H / im.shape[0])
        if f < 1: im = cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        PHOTOS[n[:2]] = grade(im)
    for n in ["01", "02", "03", "08"]:
        rgba = np.asarray(Image.open(T + os.environ.get("CUTDIR", "cut/") + f"{n}.png"))
        f = min(1.0, 1.1 * H / rgba.shape[0])
        if f < 1: rgba = cv2.resize(rgba, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        rgb = np.ascontiguousarray(rgba[..., :3])
        lo = np.asarray(Image.open(T + CUT_LO[n]).convert("RGB"))
        if lo.shape[0] < rgb.shape[0]:   # 35% original texture against the waxy upscale look
            lo = cv2.resize(lo, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
            rgb = cv2.addWeighted(rgb, 0.65, lo, 0.35, 0)
        rgb = grade(rgb); a = rgba[..., 3]
        ys, xs = np.where(a > 128)
        CUTS[n] = dict(rgb=rgb, a=a, box=(xs.min(), ys.min(), xs.max(), ys.max()))

TEXT_CACHE = {}
def text_img(s, size_frac, font=FONT_H, color=(255, 255, 255), weight=None, stroke=0, stroke_color=None,
             hollow=False, tracking=0.0):
    key = (repr(s), size_frac, font, tuple(color), weight, stroke, hollow, tracking)
    if key in TEXT_CACHE: return TEXT_CACHE[key]
    px = max(8, int(size_frac * W))
    f = ImageFont.truetype(font, px)
    if weight: f.set_variation_by_axes([weight])
    sw = int(stroke * px)
    # per-character layout for tracking and multi-colour runs  [(text, colour), ...]
    runs = s if isinstance(s, list) else [(s, color)]
    track = tracking * px
    widths = []
    for txt, _ in runs:
        for ch in txt: widths.append(f.getlength(ch) + track)
    tw = int(sum(widths) - track + 2 * sw + 4); asc, desc = f.getmetrics(); th = asc + desc + 2 * sw + 4
    img = Image.new("RGBA", (max(tw, 1), th), (0, 0, 0, 0)); d = ImageDraw.Draw(img)
    x = sw + 2; i = 0
    for txt, col in runs:
        for ch in txt:
            if hollow:
                d.text((x, sw + 2), ch, font=f, fill=(0, 0, 0, 0), stroke_width=sw, stroke_fill=tuple(col) + (255,))
            else:
                d.text((x, sw + 2), ch, font=f, fill=tuple(col) + (255,), stroke_width=sw,
                       stroke_fill=(tuple(stroke_color) + (255,)) if stroke_color else None)
            x += widths[i]; i += 1
    a = np.asarray(img)
    if hollow:   # PIL draws fill over stroke; rebuild pure outline alpha
        a = a.copy()
    TEXT_CACHE[key] = (np.ascontiguousarray(a[..., :3]), a[..., 3].copy())
    return TEXT_CACHE[key]

# ------------------------------------------------------------------ compositing
def blend_into(canvas, rgb, a, x0, y0, opacity=1.0):
    """Alpha-blend rgb/a (uint8) with top-left at x0,y0 (ints), clipped to canvas."""
    h, w = a.shape
    cx0, cy0 = max(0, x0), max(0, y0); cx1, cy1 = min(W, x0 + w), min(H, y0 + h)
    if cx1 <= cx0 or cy1 <= cy0 or opacity <= 0: return
    sa = a[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0].astype(np.float32) * (opacity / 255.0)
    sr = rgb[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0].astype(np.float32)
    dst = canvas[cy0:cy1, cx0:cx1].astype(np.float32)
    sa = sa[..., None]
    canvas[cy0:cy1, cx0:cx1] = (dst + (sr - dst) * sa).astype(np.uint8)

def place(canvas, rgb, a, cx, cy, scale, opacity=1.0, rot=0.0, blur_x=0):
    """Place a layer centred at (cx, cy) px with uniform scale (and optional rotation)."""
    h, w = a.shape
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    if rot == 0:
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        # crop to the visible part first so we never resize off-screen pixels
        x0 = int(round(cx - nw / 2)); y0 = int(round(cy - nh / 2))
        vx0, vy0 = max(0, x0), max(0, y0); vx1, vy1 = min(W, x0 + nw), min(H, y0 + nh)
        if vx1 <= vx0 or vy1 <= vy0: return
        sx0 = int((vx0 - x0) / scale); sy0 = int((vy0 - y0) / scale)
        sx1 = min(w, int(math.ceil((vx1 - x0) / scale)) + 1); sy1 = min(h, int(math.ceil((vy1 - y0) / scale)) + 1)
        sub_r = rgb[sy0:sy1, sx0:sx1]; sub_a = a[sy0:sy1, sx0:sx1]
        M = np.float32([[scale, 0, (sx0 * scale + x0) - vx0], [0, scale, (sy0 * scale + y0) - vy0]])
        size = (vx1 - vx0, vy1 - vy0)
        r = cv2.warpAffine(sub_r, M, size, flags=interp, borderMode=cv2.BORDER_REPLICATE)
        al = cv2.warpAffine(sub_a, M, size, flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if blur_x > 1:
            k = int(blur_x) | 1; r = cv2.blur(r, (k, 1)); al = cv2.blur(al, (k, 1))
        blend_into(canvas, r, al, vx0, vy0, opacity)
    else:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rot, scale)
        M[0, 2] += cx - w / 2; M[1, 2] += cy - h / 2
        r = cv2.warpAffine(rgb, M, (W, H), flags=cv2.INTER_LINEAR)
        al = cv2.warpAffine(a, M, (W, H), flags=cv2.INTER_LINEAR)
        blend_into(canvas, r, al, 0, 0, opacity)

def fill_photo(key, fx, fy, zoom, dx=0.0, dy=0.0, rot=0.0):
    """Full-bleed crop of a photo: focus point (fx, fy) in [0,1] image coords, zoom >= 1."""
    im = PHOTOS[key]; ih, iw = im.shape[:2]
    s = max(W / iw, H / ih) * zoom
    # clamp focus so the frame never leaves the image
    half_w, half_h = W / 2 / s, H / 2 / s
    cx = clamp(fx * iw + dx * iw, half_w, iw - half_w); cy = clamp(fy * ih + dy * ih, half_h, ih - half_h)
    M = cv2.getRotationMatrix2D((cx, cy), rot, s)
    M[0, 2] += W / 2 - cx; M[1, 2] += H / 2 - cy
    return cv2.warpAffine(im, M, (W, H), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)

# ------------------------------------------------------------------ backgrounds
BG_STYLES = {
    "navy":  dict(c1="#0b1a3a", c2="#030713", glow="#3563b8", g=1.0, streak=[("#ffffff", .10), ("#2f6fe0", .16), ("#e0323c", .14)]),
    "red":   dict(c1="#1c0a14", c2="#060309", glow="#b3222f", g=0.9, streak=[("#ff3b47", .30), ("#ffffff", .10)]),
    "blue":  dict(c1="#0d2a63", c2="#040b1f", glow="#4f93ff", g=1.0, streak=[("#8cc4ff", .20), ("#ffffff", .10)]),
    "tri":   dict(c1="#0b1a3a", c2="#030713", glow="#3563b8", g=0.9, streak=[("#ffffff", .30), ("#2f6fe0", .34), ("#e0323c", .34)]),
}
_bg_grid = None
def background(style, t, speed=1.0, glow_y=0.42):
    global _bg_grid
    st = BG_STYLES[style]
    w, h = W // 4 + 1, H // 4 + 1
    if _bg_grid is None:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        _bg_grid = (xx / w, yy / h * (H / W))
    X, Y = _bg_grid
    ny = Y / (H / W)
    img = hexc(st["c1"]) * (1 - ny[..., None]) + hexc(st["c2"]) * ny[..., None]
    r = np.sqrt((X - 0.5) ** 2 + (Y - glow_y * H / W) ** 2)
    g = np.exp(-(r / 0.55) ** 2) * st["g"] * (0.92 + 0.08 * math.sin(t * 2.1))
    img = img + (hexc(st["glow"]) - img) * g[..., None] * 0.85
    # diagonal light streaks drifting
    u = X * math.cos(0.45) + Y * math.sin(0.45)
    for i, (col, amp) in enumerate(st["streak"]):
        pos = ((t * 0.18 * speed + i * 0.37) % 1.6) - 0.3 + 0.35 * i
        band = np.exp(-((u - pos) / (0.035 + 0.02 * i)) ** 2) * amp
        img = img + (hexc(col) - img) * band[..., None]
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_CUBIC)

def floor_shadow(canvas, cx, cy, width):
    h = max(4, int(width * 0.12)); w = max(8, int(width))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    a = np.exp(-(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) * 2.5) * 150
    blend_into(canvas, np.zeros((h, w, 3), np.uint8), a.astype(np.uint8), int(cx - w / 2), int(cy - h / 2))

def put_cut(canvas, key, cx, bottom, height, opacity=1.0, blur_x=0, shadow=True):
    """Place a cut-out so the person's bbox is `height` px tall with feet at `bottom`."""
    c = CUTS[key]; x0, y0, x1, y1 = c["box"]
    s = height / (y1 - y0)
    ih, iw = c["a"].shape
    ccx = cx - ((x0 + x1) / 2 - iw / 2) * s
    ccy = bottom - (y1 - ih / 2) * s
    if shadow: floor_shadow(canvas, cx, bottom - 0.004 * H, (x1 - x0) * s * 0.9)
    place(canvas, c["rgb"], c["a"], ccx, ccy, s, opacity, blur_x=blur_x)

# ------------------------------------------------------------------ text helpers
def put_text(canvas, s, size, cx, cy, opacity=1.0, scale=1.0, anchor="c", **kw):
    rgb, a = text_img(s, size, **kw)
    if scale != 1.0:
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=interp)
        a = cv2.resize(a, None, fx=scale, fy=scale, interpolation=interp)
    h, w = a.shape
    x = cx - w / 2 if anchor == "c" else (cx if anchor == "l" else cx - w)
    blend_into(canvas, rgb, a, int(x), int(cy - h / 2), opacity)
    return w, h

def wipe_text(canvas, s, size, cx, cy, p, anchor="c", **kw):
    """Left-to-right reveal (p in 0..1)."""
    rgb, a = text_img(s, size, **kw)
    h, w = a.shape
    edge = int((w + 0.3 * w) * clamp(p)) - int(0.15 * w)
    ramp = np.clip((edge - np.arange(w)) / (0.15 * w + 1), 0, 1).astype(np.float32)
    a2 = (a.astype(np.float32) * ramp[None, :]).astype(np.uint8)
    x = cx - w / 2 if anchor == "c" else (cx if anchor == "l" else cx - w)
    blend_into(canvas, rgb, a2, int(x), int(cy - h / 2))

def bar(canvas, x, y, w, h, color, opacity=1.0):
    x, y, w, h = int(x), int(y), max(1, int(w)), max(1, int(h))
    blend_into(canvas, np.full((h, w, 3), color, np.uint8), np.full((h, w), 255, np.uint8), x, y, opacity)

def tricolor_line(canvas, x, y, w, p=1.0):
    th = max(2, int(0.006 * W)); seg = w * clamp(p) / 3
    for i, c in enumerate([(255, 255, 255), (47, 111, 224), (224, 50, 60)]):
        bar(canvas, x + i * seg, y, seg, th, c)

def dark_band(canvas, y0, y1, strength=0.55):
    y0, y1 = int(max(0, y0)), int(min(H, y1))
    g = np.linspace(0, 1, y1 - y0, dtype=np.float32) ** 0.8
    canvas[y0:y1] = (canvas[y0:y1].astype(np.float32) * (1 - strength * g)[:, None, None]).astype(np.uint8)

def logo(canvas, cx, cy, size, opacity=1.0, scale=1.0):
    return put_text(canvas, [("R", (230, 45, 55)), ("USSIA", (255, 255, 255))], size, cx, cy,
                    opacity=opacity, scale=scale, tracking=0.08)

# ------------------------------------------------------------------ scenes
def kick_punch(t, t0, t1, amt=0.025):
    """Scale bump on every beat inside [t0, t1)."""
    if not (t0 <= t < t1): return 1.0
    ph = (t - t0) % BEAT
    return 1 + amt * math.exp(-ph / 0.09)

def sc_intro(t):
    # 0 - 1.92: slow push on the back print, title reveal
    p = t / BAR
    fr = fill_photo("05", 0.5, 0.42 + 0.08 * p, 1.35 + 0.15 * p)
    fr = (fr.astype(np.float32) * (0.55 + 0.25 * ease_out(p))).astype(np.uint8)
    dark_band(fr, 0.55 * H, H, 0.7)
    wipe_text(fr, "СОЗДАН", 0.15, W / 2, 0.70 * H, (t - 0.25) / 0.6)
    wipe_text(fr, "ДЛЯ ДВИЖЕНИЯ", 0.085, W / 2, 0.78 * H, (t - 0.55) / 0.7, font=FONT_H, color=(235, 60, 70))
    return fr

DETAIL_FLASH = [("10", 0.33, 0.52, 1.25, 0.10, 0.0), ("04", 0.78, 0.52, 1.45, -0.05, 0.0),
                ("06", 0.42, 0.58, 1.30, 0.0, 0.05), ("07", 0.45, 0.35, 1.25, 0.0, -0.04)]
def sc_teaser(t):
    # 1.92 - 3.84: four beat cuts, push-ins, build to the drop
    i = min(3, int(t / BEAT)); lt = t - i * BEAT
    k, fx, fy, z, dx, dy = DETAIL_FLASH[i]
    q = lt / BEAT
    fr = fill_photo(k, fx, fy, z * (1 + 0.12 * ease_out(q)), dx * q, dy * q)
    if t > 1.2:   # riser: brighten toward the drop
        g = ((t - 1.2) / 0.72) ** 2
        fr = cv2.addWeighted(fr, 1 - 0.6 * g, np.full_like(fr, 255), 0.6 * g, 0)
    return fr

def sc_hero(t):
    # 3.84 - 5.76 (drop)
    fr = background("navy", t + 3.84)
    # giant hollow word behind
    put_text(fr, "RUSSIA", 0.34, W / 2 + (0.25 - 0.18 * t / BAR) * W, 0.30 * H, opacity=0.22,
             hollow=True, stroke=0.018, color=(255, 255, 255))
    s = kick_punch(t, 0, BAR, 0.02) * (1.08 - 0.08 * ease_out(t / 0.5))
    put_cut(fr, "01", W / 2, 0.80 * H, 0.64 * H * s)
    dark_band(fr, 0.70 * H, H, 0.75)
    p = ease_out((t - 0.15) / 0.4)
    put_text(fr, "КОСТЮМ", 0.075, W / 2, (0.855 + 0.03 * (1 - p)) * H, opacity=p, color=(230, 235, 245))
    logo(fr, W / 2, (0.915 + 0.03 * (1 - p)) * H, 0.14, opacity=p)
    p2 = ease_out((t - 0.55) / 0.4)
    put_text(fr, "олимпийка + брюки", 0.045, W / 2, 0.965 * H, opacity=p2, font=FONT_B, weight=600,
             color=(200, 212, 235))
    return fr

FEATURES = [  # key, focus x, y, zoom start, zoom end, pan dx, title, subtitle
    ("04", 0.72, 0.52, 1.30, 1.55, 0.06, "ВЫШИТЫЙ ГЕРБ", "и надпись RUS на груди"),
    ("10", 0.45, 0.55, 1.15, 1.30, 0.22, "ФЛАГ НА РУКАВЕ", "лампасы в цветах триколора"),
    ("05", 0.50, 0.40, 1.10, 1.25, 0.00, "ПРИНТ НА СПИНЕ", "вертикальная надпись RUSSIA"),
    ("09", 0.52, 0.55, 1.15, 1.30, -0.04, "КАРМАНЫ НА МОЛНИИ", "всё нужное — при себе"),
    ("06", 0.42, 0.58, 1.15, 1.30, 0.04, "МАНЖЕТЫ-РЕЗИНКИ", "рукава не сползают"),
    ("07", 0.45, 0.35, 1.15, 1.30, 0.00, "ПОЯС СО ШНУРКОМ", "посадка под тебя"),
]
def sc_features(t):
    # 5.76 - 11.52: six details, 2 beats each
    i = min(5, int(t / (2 * BEAT))); lt = t - i * 2 * BEAT; q = lt / (2 * BEAT)
    k, fx, fy, z0, z1, pdx, title, sub = FEATURES[i]
    dy = 0.18 * q if k == "05" else 0.0
    fr = fill_photo(k, fx, fy, (z0 + (z1 - z0) * ease_io(q)) * kick_punch(lt, 0, 2 * BEAT, 0.012), pdx * q, dy)
    dark_band(fr, 0.66 * H, H, 0.8)
    x = 0.08 * W
    put_text(fr, f"0{i + 1}", 0.05, x, 0.775 * H, anchor="l", opacity=ease_out(lt / 0.2), color=(235, 60, 70))
    tricolor_line(fr, x + 0.12 * W, 0.772 * H, 0.2 * W, ease_out(lt / 0.3))
    tsz = 0.083 * min(1.0, 0.84 * W / text_img(title, 0.083)[1].shape[1])
    wipe_text(fr, title, tsz, x, 0.83 * H, lt / 0.28, anchor="l")
    p = ease_out((lt - 0.2) / 0.3)
    put_text(fr, sub, 0.045, x, (0.885 + 0.01 * (1 - p)) * H, anchor="l", opacity=p, font=FONT_B, weight=600,
             color=(210, 220, 240))
    return fr

def top_shade(fr, y1, strength=0.7):
    y1 = int(y1); g = (1 - np.linspace(0, 1, y1, dtype=np.float32)) ** 1.2
    fr[:y1] = (fr[:y1].astype(np.float32) * (1 - strength * g)[:, None, None]).astype(np.uint8)

def scenario_title(fr, word, sub, lt, accent, shade=0.7):
    top_shade(fr, 0.34 * H, shade)
    p = ease_back(lt / 0.22)
    sc = 1.35 - 0.35 * p
    put_text(fr, word, 0.105, W / 2, 0.12 * H, opacity=clamp(lt / 0.1), scale=sc)
    tricolor_line(fr, W / 2 - 0.15 * W, 0.165 * H, 0.3 * W, ease_out((lt - 0.1) / 0.3))
    p2 = ease_out((lt - 0.25) / 0.3)
    put_text(fr, sub, 0.048, W / 2, (0.20 + 0.01 * (1 - p2)) * H, opacity=p2, font=FONT_B, weight=600, color=accent)

def sc_scenarios(t):
    # 11.52 - 19.20: four use cases, one bar each
    i = min(3, int(t / BAR)); lt = t - i * BAR
    if i == 0:   # training
        fr = background("red", lt, speed=3.0, glow_y=0.6)
        for j in range(7):       # speed lines
            y = (0.30 + 0.09 * j) * H; L = (0.25 + 0.1 * (j % 3)) * W
            x = W - ((lt * (2.2 + 0.4 * j) * W + j * 0.3 * W) % (W + L))
            bar(fr, x, y, L, max(2, 0.004 * W), (255, 255, 255), 0.18)
        p = ease_out(lt / 0.35)
        put_cut(fr, "08", W * (0.52 + 0.5 * (1 - p)), 1.03 * H, 1.0 * H * kick_punch(lt, 0.35, BAR, 0.015),
                blur_x=(1 - p) * 0.15 * W)
        scenario_title(fr, "НА ТРЕНИРОВКУ", "свобода движения", lt, (255, 170, 175), shade=0.97)
    elif i == 1:  # travel
        fr = background("blue", lt + 5, glow_y=0.55)
        p = ease_out(lt / 0.4)
        put_cut(fr, "02", W / 2, H * (1.0 + 0.25 * (1 - p)), 0.62 * H * kick_punch(lt, 0.4, BAR, 0.012), shadow=False)
        scenario_title(fr, "В ДОРОГУ", "удобно в пути и в аэропорту", lt, (170, 205, 255))
    elif i == 2:  # tribune / fans
        fr = background("tri", lt + 9, speed=1.5, glow_y=0.5)
        put_text(fr, "RUSSIA", 0.34, W / 2 - (0.1 + 0.2 * lt / BAR) * W, 0.55 * H, opacity=0.14,
                 hollow=True, stroke=0.018)
        p = ease_out(lt / 0.35)
        put_cut(fr, "03", W / 2, 0.97 * H, 0.70 * H * (1.1 - 0.1 * p) * kick_punch(lt, 0.35, BAR, 0.015), opacity=p)
        scenario_title(fr, "НА ТРИБУНУ", "болей за своих", lt, (255, 255, 255))
    else:         # every day: three panels on beats
        fr = background("navy", lt + 13)
        panels = [("01", 0.5, 0.42, 1.15), ("10", 0.55, 0.55, 1.2), ("08", 0.45, 0.45, 1.1)]
        pw = W // 3
        for j, (k, fx, fy, z) in enumerate(panels):
            pj = ease_out((lt - j * BEAT * 0.5) / 0.3)
            if pj <= 0: continue
            im = PHOTOS[k]; ih, iw = im.shape[:2]
            ph = int(0.68 * H); s = max(pw / iw, ph / ih) * z * (1 + 0.05 * lt / BAR)
            cx, cy = fx * iw, fy * ih
            M = np.float32([[s, 0, pw / 2 - cx * s], [0, s, ph / 2 - cy * s]])
            panel = cv2.warpAffine(im, M, (pw, ph), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT)
            y = int(0.27 * H + (1 - pj) * 0.4 * H)
            blend_into(fr, panel, np.full((ph, pw), 255, np.uint8), j * pw, y, pj)
        for j in (1, 2): bar(fr, j * pw - 0.003 * W, 0.27 * H, 0.006 * W, 0.68 * H, (4, 9, 22))
        scenario_title(fr, "КАЖДЫЙ ДЕНЬ", "в городе, дома и на прогулке", lt, (200, 212, 235))
    return fr

MONTAGE = [("01", 0.5, 0.30, 1.6), ("04", 0.78, 0.50, 1.5), ("10", 0.85, 0.62, 1.8), ("05", 0.5, 0.5, 1.3),
           ("08", 0.36, 0.28, 1.8), ("03", 0.52, 0.25, 1.9), ("02", 0.62, 0.30, 1.6), ("09", 0.5, 0.55, 1.2),
           ("06", 0.42, 0.58, 1.3), ("04", 0.18, 0.47, 2.0), ("07", 0.45, 0.35, 1.3), ("10", 0.28, 0.52, 1.6)]
def montage_index(t):
    # beats 0-1: 1/2-beat cuts (4), beats 2-3: 1/4-beat cuts (8)
    if t < 2 * BEAT: return int(t / (BEAT / 2)), (t % (BEAT / 2)) / (BEAT / 2)
    u = t - 2 * BEAT; return 4 + min(7, int(u / (BEAT / 4))), (u % (BEAT / 4)) / (BEAT / 4)
def sc_montage(t):
    i, q = montage_index(t)
    k, fx, fy, z = MONTAGE[i % len(MONTAGE)]
    fr = fill_photo(k, fx, fy, z * (1 + 0.08 * q), rot=(1.5 if i % 2 else -1.5) * (1 - q))
    if t > 1.4:
        g = ((t - 1.4) / 0.52) ** 2
        fr = cv2.addWeighted(fr, 1 - 0.7 * g, np.full_like(fr, 255), 0.7 * g, 0)
    return fr

def sc_final(t):
    # 21.12 - end: pack shot + logo + CTA
    fr = background("navy", t + 20, glow_y=0.40)
    p = ease_out(t / 0.8)
    put_cut(fr, "03", W * (0.66 + 0.1 * (1 - p)), 0.80 * H, 0.56 * H, opacity=p * 0.95)
    put_cut(fr, "01", W * (0.36 - 0.1 * (1 - p)), 0.82 * H, 0.60 * H * (1.04 - 0.04 * p))
    dark_band(fr, 0.66 * H, H, 0.85)
    pl = ease_back((t - 0.25) / 0.45)
    logo(fr, W / 2, 0.10 * H, 0.2, opacity=clamp((t - 0.25) / 0.15), scale=0.8 + 0.2 * pl)
    tricolor_line(fr, W / 2 - 0.2 * W, 0.158 * H, 0.4 * W, ease_out((t - 0.5) / 0.4))
    p2 = ease_out((t - 0.7) / 0.5)
    put_text(fr, "СОЗДАН ДЛЯ ДВИЖЕНИЯ", 0.062, W / 2, (0.855 + 0.01 * (1 - p2)) * H, opacity=p2)
    put_text(fr, "И КОМФОРТА", 0.062, W / 2, (0.895 + 0.01 * (1 - p2)) * H, opacity=p2, color=(235, 60, 70))
    p3 = ease_back((t - 1.3) / 0.4)
    if t > 1.3:
        bw, bh = 0.62 * W, 0.045 * H
        s = 0.8 + 0.2 * p3
        pulse = 1 + 0.03 * math.sin((t - 1.3) * 2 * math.pi / (2 * BEAT)) if t > 1.8 else 1
        bar(fr, W / 2 - bw * s * pulse / 2, 0.945 * H - bh * s * pulse / 2, bw * s * pulse, bh * s * pulse,
            (225, 40, 52), clamp((t - 1.3) / 0.15))
        put_text(fr, "ВЫБИРАЙ СВОЙ РАЗМЕР", 0.040, W / 2, 0.945 * H, opacity=clamp((t - 1.35) / 0.15),
                 scale=s * pulse)
    if t > 2.9:   # fade out with the music
        g = ease_in((t - 2.9) / (DUR - 21.12 - 2.9))
        fr = (fr.astype(np.float32) * (1 - g)).astype(np.uint8)
    return fr

SCENES = [(0.0, sc_intro), (BAR, sc_teaser), (2 * BAR, sc_hero), (3 * BAR, sc_features),
          (6 * BAR, sc_scenarios), (10 * BAR, sc_montage), (11 * BAR, sc_final)]

# cut transitions: time, kind ('whip+' / 'whip-' / 'flash' / 'punch'), strength
CUTS_FX = [(BAR, "flash", 0.5), (2 * BAR, "flash", 1.0), (6 * BAR, "flash", 0.7), (11 * BAR, "flash", 1.0)]
for j in range(1, 4): CUTS_FX.append((BAR + j * BEAT, "flash", 0.35))
for j in range(1, 6): CUTS_FX.append((3 * BAR + j * 2 * BEAT, "whip+" if j % 2 else "whip-", 1.0))
CUTS_FX.append((3 * BAR, "whip-", 1.0))
for j in range(1, 4): CUTS_FX.append((6 * BAR + j * BAR, "whip+" if j % 2 else "whip-", 1.0))
CUTS_FX.append((10 * BAR, "flash", 0.6))

def post(fr, t):
    shift = 0.0; blur = 0.0; flash = 0.0; ca = 0.0
    for tc, kind, s in CUTS_FX:
        d = t - tc
        if kind == "flash" and 0 <= d < 0.35:
            flash = max(flash, s * math.exp(-d / 0.07)); ca = max(ca, s * math.exp(-d / 0.05))
        if kind.startswith("whip") and -0.12 <= d < 0.12:
            sg = 1 if kind == "whip+" else -1
            if d < 0: p = 1 + d / 0.12; shift = -sg * p * p * 0.35 * W; blur = max(blur, p * 0.22 * W)
            else: p = 1 - d / 0.12; shift = sg * p * p * 0.35 * W; blur = max(blur, p * 0.22 * W)
            ca = max(ca, p)
    if shift or blur > 2:
        M = np.float32([[1, 0, shift], [0, 1, 0]])
        fr = cv2.warpAffine(fr, M, (W, H), borderMode=cv2.BORDER_REFLECT)
        if blur > 2: fr = cv2.blur(fr, (int(blur) | 1, 1))
    if ca > 0.02:
        o = max(1, int(ca * 0.006 * W))
        fr[:, o:, 0] = fr[:, :-o, 0].copy(); fr[:, :-o, 2] = fr[:, o:, 2].copy()
    if flash > 0.01:
        fr = cv2.addWeighted(fr, 1 - flash, np.full_like(fr, 255), flash, 0)
    return fr

VIGN = None
def vignette(fr):
    global VIGN
    if VIGN is None:
        yy, xx = np.mgrid[0:H // 8, 0:W // 8].astype(np.float32)
        r = np.sqrt(((xx / (W / 8) - 0.5) * 1.0) ** 2 + ((yy / (H / 8) - 0.5) * 0.8) ** 2)
        v = 1 - 0.35 * np.clip((r - 0.35) / 0.45, 0, 1) ** 1.5
        VIGN = cv2.resize(v, (W, H), interpolation=cv2.INTER_CUBIC)[..., None]
    return (fr.astype(np.float32) * VIGN).astype(np.uint8)

def frame(n):
    t = n / FPS
    t0, fn = [s for s in SCENES if s[0] <= t + 1e-9][-1]
    fr = fn(t - t0)
    fr = post(fr, t)
    if t < 0.4: fr = (fr.astype(np.float32) * ease_out(t / 0.4)).astype(np.uint8)
    return vignette(fr)

_init = False
def worker(n):
    global _init
    if not _init: load_assets(); _init = True
    fd = os.environ.get("FRAMEDIR")
    if fd:   # stage 1 of the 8K path: frames to disk, encoded separately
        cv2.imwrite(f"{fd}/{n:05d}.jpg", cv2.cvtColor(frame(n), cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 96, cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444])
        return n, None
    return n, frame(n).tobytes()

if __name__ == "__main__":
    out = sys.argv[2] if len(sys.argv) > 2 else T + "preview.mp4"
    ta = float(sys.argv[3]) if len(sys.argv) > 3 else 0; tb = float(sys.argv[4]) if len(sys.argv) > 4 else DUR
    frames = list(range(int(ta * FPS), int(tb * FPS)))
    if out.endswith(".png"):   # contact sheet of given times
        load_assets(); imgs = [frame(int(float(x) * FPS)) for x in sys.argv[3:]]
        cols = 6; rows = (len(imgs) + cols - 1) // cols
        sheet = np.zeros((rows * H, cols * W, 3), np.uint8)
        for i, im in enumerate(imgs): sheet[(i // cols) * H:(i // cols + 1) * H, (i % cols) * W:(i % cols + 1) * W] = im
        Image.fromarray(sheet).save(out); sys.exit()
    big = W >= 3000
    venc = (["-c:v", "libx265", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-tag:v", "hvc1",
             "-x265-params", "vbv-maxrate=30000:vbv-bufsize=60000:log-level=error:frame-threads=2:rc-lookahead=12:bframes=4:pools=4"] if big else
            ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p"])
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-ss", str(ta), "-i", T + "music.wav", "-t", str(tb - ta),
           "-map", "0:v", "-map", "1:a", *venc, "-c:a", "aac", "-b:a", "256k",
           "-af", "loudnorm=I=-14:TP=-1:LRA=9", "-ar", "48000", "-movflags", "+faststart", out]
    if os.environ.get("FRAMEDIR"):
        from concurrent.futures import as_completed
        t0 = __import__("time").time()
        with ProcessPoolExecutor(int(os.environ.get("WORKERS", "3"))) as ex:
            for k, fu in enumerate(as_completed([ex.submit(worker, f) for f in frames])):
                fu.result()
                if k % 30 == 0: print(f"frames {k}/{len(frames)}  {__import__('time').time() - t0:.0f}s", flush=True)
        print("frames done"); sys.exit()
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    workers = int(os.environ.get("WORKERS", "3"))
    import time; t0 = time.time()
    from collections import deque
    with ProcessPoolExecutor(workers) as ex:   # bounded window: never hold more than a few 8K frames
        pending = deque(); it = iter(frames); k = 0
        for _ in range(workers + 1):
            f = next(it, None)
            if f is not None: pending.append(ex.submit(worker, f))
        while pending:
            n, buf = pending.popleft().result()
            f = next(it, None)
            if f is not None: pending.append(ex.submit(worker, f))
            enc.stdin.write(buf); del buf
            if k % 30 == 0: print(f"frame {n}/{frames[-1]}  {time.time() - t0:.0f}s", flush=True)
            k += 1
    enc.stdin.close(); enc.wait()
    print("done", out, W, H, f"{time.time() - t0:.0f}s")
