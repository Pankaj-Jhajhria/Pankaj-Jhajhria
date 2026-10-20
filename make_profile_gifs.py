#!/usr/bin/env python3
"""
Regenerates the animated GIFs used by the GitHub profile README.

    python3 tools/make_profile_gifs.py            # all three
    python3 tools/make_profile_gifs.py hero       # or: focus | divider

Outputs (written next to the static SVG assets, which are left untouched):
    assets/profile/hero-loop.gif      1200 x 300   ~7.4 s loop
    assets/profile/focus-cycle.gif     900 x  90   ~9.1 s loop
    assets/profile/flow-divider.gif   1200 x  24   ~4.2 s loop

Requirements: Python 3, Pillow, numpy, ffmpeg (used only to build an optimised
shared palette and write the GIF), and the Inter font family (falls back to
DejaVu Sans if Inter is not installed).

Every frame is drawn from geometric primitives and text at 2x (hero, focus)
or 4x (divider) supersampling, then downsampled for clean edges. Everything is
a pure function of the loop time, so the first and last frames join seamlessly.
"""
import math
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(ROOT, "assets", "profile")

# ----------------------------------------------------------------- palette --
BG0 = (7, 11, 26)
BG1 = (12, 18, 48)
INK = (7, 11, 26)
PERI = (91, 108, 255)
EDGE = (108, 139, 255)
CYAN = (34, 211, 238)
CYAN_L = (138, 234, 248)
BLUE = (79, 139, 255)
BLUE_L = (169, 195, 255)
INDIGO = (124, 108, 255)
VIOLET = (160, 107, 255)
VIOLET_L = (205, 180, 255)
AMBER = (251, 191, 36)
WHITE = (255, 255, 255)
MATTE = (13, 17, 23)  # GitHub dark canvas, used to anti-alias rounded corners


# ----------------------------------------------------------------- helpers --
def clamp01(x):
    return max(0.0, min(1.0, x))


def smooth(x):
    x = clamp01(x)
    return x * x * (3 - 2 * x)


def ease_out(x):
    x = clamp01(x)
    return 1 - (1 - x) ** 3


def ease_io(x):
    x = clamp01(x)
    return 0.5 - 0.5 * math.cos(math.pi * x)


def lerp(a, b, t):
    return a + (b - a) * t


def lerpc(c1, c2, t):
    return tuple(int(round(lerp(a, b, t))) for a, b in zip(c1, c2))


def find_font(names, size):
    dirs = ["/usr/share/fonts/opentype/inter", "/usr/share/fonts/truetype/inter",
            "/usr/share/fonts/truetype/dejavu", os.path.expanduser("~/.fonts"),
            "C:/Windows/Fonts", "/Library/Fonts", "/System/Library/Fonts/Supplemental"]
    for n in names:
        for d in dirs:
            p = os.path.join(d, n)
            if os.path.exists(p):
                return ImageFont.truetype(p, size)
    return ImageFont.truetype("DejaVuSans-Bold.ttf", size)


def inter(weight, size):
    fallback = {"Regular": "DejaVuSans.ttf", "Medium": "DejaVuSans.ttf",
                "SemiBold": "DejaVuSans-Bold.ttf", "Bold": "DejaVuSans-Bold.ttf",
                "ExtraBold": "DejaVuSans-Bold.ttf"}[weight]
    return find_font([f"Inter-{weight}.otf", f"Inter-{weight}.ttf", fallback], size)


def mono(size):
    return find_font(["DejaVuSansMono.ttf", "Inter-Medium.otf"], size)


def fit_size(weight, text, target_w, probe=200):
    return probe * target_w / inter(weight, probe).getlength(text)


def tracked_width(text, font, tracking):
    return sum(font.getlength(c) for c in text) + tracking * (len(text) - 1)


def draw_tracked(d, x, y, text, font, fill, tracking):
    for ch in text:
        d.text((x, y), ch, font=font, fill=fill, anchor="ls")
        x += font.getlength(ch) + tracking


def gradient_bg(W, H, S, glows=True, grid=40, grid_alpha=0.08):
    """The banner background: diagonal navy gradient, soft glows, faint grid."""
    w, h = W * S, H * S
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    t = ((xs / w + ys / h) / 2)[..., None]
    img = np.array(BG0, np.float32) + (np.array(BG1, np.float32) - np.array(BG0, np.float32)) * t
    if glows:
        for cx, cy, r, col, a0 in ((0.08, 0.2, 0.5, (47, 107, 255), 0.30),
                                   (0.86, 0.5, 0.42, (109, 92, 255), 0.34)):
            dd = np.hypot((xs / w - cx) / r, (ys / h - cy) / r)
            a = (a0 * (1 - np.clip(dd, 0, 1)))[..., None]
            img = img * (1 - a) + np.array(col, np.float32) * a
    if grid:
        g = ((xs.astype(int) % (grid * S)) < S) | ((ys.astype(int) % (grid * S)) < S)
        a = (g * grid_alpha)[..., None]
        img = img * (1 - a) + np.array((108, 125, 255), np.float32) * a
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB")


def corner_mask(W, H, radius):
    m = Image.new("L", (W * 4, H * 4), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, W * 4 - 1, H * 4 - 1), radius=radius * 4, fill=255)
    return m.resize((W, H), Image.LANCZOS)


SENT = (255, 0, 255)  # reserved colour: becomes the GIF's transparent index (rounded corners)


def finish_rgb(img2x, W, H, cmask):
    """Downsample, anti-alias the rounded corner against a dark matte, and paint the
    pixels that lie fully outside the corner with the reserved sentinel colour.
    (ffmpeg only delta-encodes frames that have no alpha channel, so transparency is
    restored afterwards by patch_transparency().)"""
    img = img2x.resize((W, H), Image.LANCZOS)
    img = Image.composite(img, Image.new("RGB", (W, H), MATTE), cmask)
    outside = cmask.point(lambda v: 255 if v < 128 else 0)
    img.paste(Image.new("RGB", (W, H), SENT), mask=outside)
    return img


_sprites = {}


def glow_sprite(radius, color, peak, power=2.0):
    key = (radius, color, peak, power)
    if key not in _sprites:
        n = 2 * radius + 1
        ys, xs = np.mgrid[0:n, 0:n]
        d = np.hypot(xs - radius, ys - radius) / radius
        a = (peak * 255 * (1 - np.clip(d, 0, 1)) ** power).astype(np.uint8)
        _sprites[key] = (Image.new("RGB", (n, n), color), Image.fromarray(a, "L"))
    return _sprites[key]


def paste_glow(img, x, y, radius, color, peak, power=2.0):
    col, mask = glow_sprite(int(radius), color, peak, power)
    img.paste(col, (int(round(x - radius)), int(round(y - radius))), mask)


def temporal_filter(frames, thr):
    """Lossy delta: a pixel whose colour moved by <= thr (per channel) keeps the value
    it already had on screen. Unchanged pixels then quantise to identical palette
    indices, so the encoder's changed-rectangle shrinks and LZW sees long flat runs.
    thr is small enough to be visually indistinguishable."""
    prev = None
    for fr in frames:
        a = np.array(fr.convert("RGB"))
        if prev is not None:
            d = np.abs(a.astype(np.int16) - prev.astype(np.int16)).max(axis=2)
            a[d <= thr] = prev[d <= thr]
        prev = a.copy()
        yield Image.fromarray(a, "RGB")


def patch_transparency(path, index):
    """Mark `index` as the transparent colour of the FIRST frame only.
    Later frames are delta frames: ffmpeg already makes their unchanged pixels
    (including the corners) transparent, so the corners are painted exactly once,
    in frame 0, where this patch makes the sentinel colour see-through."""
    data = bytearray(open(path, "rb").read())
    i = 13
    if data[10] & 0x80:
        i += 3 * (2 ** ((data[10] & 7) + 1))
    n = 0
    while i < len(data):
        b = data[i]
        if b == 0x3B:
            break
        if b == 0x21:
            label = data[i + 1]
            if label == 0xF9 and data[i + 2] == 4:
                data[i + 3] |= 0x01      # transparency flag
                data[i + 6] = index      # transparent colour index
                n += 1
                break
            i += 2
            while data[i] != 0:
                i += data[i] + 1
            i += 1
        elif b == 0x2C:
            flags = data[i + 9]
            i += 10
            if flags & 0x80:
                i += 3 * (2 ** ((flags & 7) + 1))
            i += 1
            while data[i] != 0:
                i += data[i] + 1
            i += 1
        else:
            raise ValueError(f"unexpected GIF block 0x{b:02x} at {i}")
    open(path, "wb").write(data)
    return n


def global_palette(path):
    d = open(path, "rb").read(13 + 768)
    n = 2 ** ((d[10] & 7) + 1)
    return [tuple(d[13 + 3 * k: 16 + 3 * k]) for k in range(n)]


def encode_gif(frames_dir, out_path, delay_cs, colors=128, dither="bayer:bayer_scale=5"):
    """RGB frames (corners = SENT) -> optimised GIF with a shared palette,
    changed-rectangle frames and transparent rounded corners."""
    tmp = tempfile.mkdtemp(prefix="pal_")
    pal_png = os.path.join(tmp, "pal.png")
    src = os.path.join(frames_dir, "f%03d.png")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vf",
                    f"palettegen=max_colors={colors - 1}:stats_mode=full", pal_png], check=True)
    pal = Image.open(pal_png).convert("RGB")
    px = list(pal.get_flattened_data() if hasattr(pal, 'get_flattened_data') else pal.getdata())
    # palettegen pads unused slots by repeating a colour (and may return fewer colours than
    # asked for), so de-duplicate, make the entry closest to the sentinel exactly the sentinel,
    # and fill the remaining slots with unique far-away greens that are never chosen.
    uniq = list(dict.fromkeys(px))
    k = min(range(len(uniq)), key=lambda j: sum((uniq[j][c] - SENT[c]) ** 2 for c in range(3)))
    uniq[k] = SENT
    px = uniq + [(0, 255, j % 256) for j in range(256 - len(uniq))]
    pal.putdata(px)
    pal.save(pal_png)
    vf = f"[0:v][1:v]paletteuse=dither={dither}:diff_mode=rectangle"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"100/{delay_cs}", "-i", src,
                    "-i", pal_png, "-lavfi", vf, "-loop", "0", out_path], check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    # the transparent index is whatever frame 0 actually uses for the corner pixels
    im = Image.open(out_path)
    W, H = im.size
    idxs = {im.getpixel(p) for p in ((0, 0), (W - 1, 0), (0, H - 1), (W - 1, H - 1))}
    assert len(idxs) == 1, f"corner pixels use several palette indices: {idxs}"
    idx = idxs.pop()
    assert tuple(im.getpalette()[3 * idx:3 * idx + 3]) == SENT, "corner index is not the sentinel colour"
    im.close()
    patch_transparency(out_path, idx)


def encode_gif_alpha(frames_dir, out_path, delay_cs, colors=64, dither="none"):
    """RGBA frames (thin divider): ffmpeg's alpha path is fine for a 24 px strip."""
    vf = (f"split[a][b];[a]palettegen=max_colors={colors}:reserve_transparent=1:stats_mode=full[p];"
          f"[b][p]paletteuse=dither={dither}:alpha_threshold=128:diff_mode=rectangle")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"100/{delay_cs}",
                    "-i", os.path.join(frames_dir, "f%03d.png"), "-vf", vf, "-loop", "0", out_path], check=True)


# =========================================================== 1. HERO LOOP ==
class Hero:
    W, H, S = 1200, 300, 2
    DELAY = 8  # centiseconds -> 12.5 fps
    N = 92
    DT = DELAY / 100.0
    T = N * DT  # 7.36 s
    THR, COLORS, DITHER = 6, 192, "bayer:bayer_scale=4"
    C0 = 4.0  # loop phase of frame 0 (name fully shown on first paint)

    SCAN_START, SCAN_DUR = 0.10, 1.50
    FADE_START, FADE_DUR = 6.85, 0.40
    HOP = 0.45

    LAYERS = [
        [(770, 78), (770, 128), (770, 178), (770, 228)],
        [(890, 60), (890, 105), (890, 150), (890, 195), (890, 240)],
        [(1010, 70), (1010, 115), (1010, 160), (1010, 205), (1010, 250)],
        [(1130, 95), (1130, 150), (1130, 205)],
    ]
    LAYER_COL = [CYAN, BLUE, INDIGO, VIOLET]
    COLS = [770, 890, 1010, 1130]
    OUT = (1130, 150)
    BASE_LEVEL = 0.58
    # (start time, colour, light colour, path input -> hidden -> hidden -> output)
    FLOWS = [
        (1.85, CYAN, (200, 250, 255), [(770, 128), (890, 105), (1010, 160), (1130, 150)]),
        (3.15, VIOLET, VIOLET_L, [(770, 78), (890, 150), (1010, 115), (1130, 150)]),
        (4.45, CYAN, (200, 250, 255), [(770, 228), (890, 195), (1010, 205), (1130, 150)]),
    ]

    def __init__(self):
        S = self.S
        self.cmask = corner_mask(self.W, self.H, 18)
        self.base = self._build_base()
        # name layout
        self.name = "PANKAJ JHAJHRIA"
        fs = fit_size("ExtraBold", self.name, 600)
        self.nfont = inter("ExtraBold", fs * S)
        self.nx0, self.nbase = 62.0, 154.0
        self.char_x = [self.nx0 + self.nfont.getlength(self.name[:i]) / S for i in range(len(self.name))]
        self.name_right = self.nx0 + self.nfont.getlength(self.name) / S
        gx0, gx1 = self.char_x[7], self.name_right
        self.chars = []
        for i, ch in enumerate(self.name):
            if ch == " ":
                self.chars.append(None)
                continue
            l, t, r, b = self.nfont.getbbox(ch, anchor="ls")
            mask = Image.new("L", (r - l + 4, b - t + 4), 0)
            ImageDraw.Draw(mask).text((2 - l, 2 - t), ch, font=self.nfont, fill=255, anchor="ls")
            xs = (self.char_x[i] * S + l - 2 + np.arange(mask.width)) / S
            if i < 7:
                col = np.tile(np.array(WHITE, np.uint8), (mask.height, mask.width, 1))
            else:
                tt = np.clip((xs - gx0) / (gx1 - gx0), 0, 1)[None, :, None]
                c = np.array(CYAN_L, np.float32) + (np.array(BLUE, np.float32) - np.array(CYAN_L, np.float32)) * tt
                col = np.broadcast_to(c, (mask.height, mask.width, 3)).astype(np.uint8)
            self.chars.append((mask, Image.fromarray(col, "RGB"), l - 2, t - 2))
        # scan-line sprite (bright hairline + trailing glow, fades top/bottom)
        tw = 190
        sx = np.arange(tw)[None, :]
        sy = np.arange(self.H * S)[:, None]
        vert = np.clip(np.minimum(sy, self.H * S - sy) / (36 * S), 0, 1) ** 0.8
        trail = 0.17 * np.exp(-(tw - 1 - sx) / 42.0) * (sx < tw - 4)
        core = 0.95 * (sx >= tw - 4)
        a = np.clip((trail + core) * vert, 0, 1)
        self.scan_mask = Image.fromarray((a * 255).astype(np.uint8), "L")
        self.scan_col = Image.new("RGB", (tw, self.H * S), CYAN)
        self.scan_tw = tw

    # ---- static layer ---------------------------------------------------
    def _build_base(self):
        S, W, H = self.S, self.W, self.H
        img = gradient_bg(W, H, S)
        d = ImageDraw.Draw(img, "RGBA")
        d.rounded_rectangle((S, S, W * S - S - 1, H * S - S - 1), radius=17 * S, outline=PERI + (82,), width=2 * S)
        # chips
        f = inter("Bold", 12.5 * S)
        for x, y, w, text, col, tcol in ((64, 48, 188, "AI & DATA SCIENCE", CYAN, CYAN_L),
                                         (264, 48, 204, "MACHINE LEARNING", BLUE, BLUE_L),
                                         (480, 48, 192, "COMPUTER VISION", VIOLET, VIOLET_L)):
            d.rounded_rectangle((x * S, y * S, (x + w) * S, (y + 30) * S), radius=15 * S,
                                fill=col + (26,), outline=col + (178,), width=int(1.5 * S))
            tw = tracked_width(text, f, 1.2 * S)
            draw_tracked(d, (x + w / 2) * S - tw / 2, (y + 20) * S, text, f, tcol + (255,), 1.2 * S)
        # tagline + amber marker
        tag = "Building and understanding practical ML systems."
        tf = inter("Regular", fit_size("Regular", tag, 470) * S)
        d.text((64 * S, 204 * S), tag, font=tf, fill=(180, 191, 220, 255), anchor="ls")
        d.rounded_rectangle((64 * S, 226 * S, 120 * S, 230 * S), radius=2 * S, fill=AMBER + (255,))
        d.rounded_rectangle((128 * S, 226 * S, 148 * S, 230 * S), radius=2 * S, fill=AMBER + (115,))
        # network edges (low contrast)
        for a, b in zip(self.LAYERS, self.LAYERS[1:]):
            for p in a:
                for q in b:
                    d.line([(p[0] * S, p[1] * S), (q[0] * S, q[1] * S)], fill=EDGE + (62,), width=2)
        # dotted reference path (as in the static banner)
        path = self.FLOWS[0][3]
        for p, q in zip(path, path[1:]):
            length = math.hypot(q[0] - p[0], q[1] - p[1])
            n = int(length / 9)
            for k in range(n + 1):
                t = k * 9 / length
                x, y = lerp(p[0], q[0], t), lerp(p[1], q[1], t)
                d.ellipse(((x - 1.4) * S, (y - 1.4) * S, (x + 1.4) * S, (y + 1.4) * S), fill=CYAN + (170,))
        # direction axis under the network
        ay = 283
        d.line([(770 * S, ay * S), (1132 * S, ay * S)], fill=EDGE + (70,), width=2)
        d.polygon([(1146 * S, ay * S), (1136 * S, (ay - 5) * S), (1136 * S, (ay + 5) * S)], fill=EDGE + (110,))
        return img

    # ---- timeline helpers -----------------------------------------------
    def t_pass(self, x):
        return self.SCAN_START + self.SCAN_DUR * (x + 20) / 1240.0

    def level(self, c, col_x):
        rise = smooth((c - self.t_pass(col_x)) / 0.35)
        fall = smooth((c - self.FADE_START) / (self.FADE_DUR - 0.05))
        return self.BASE_LEVEL + (1 - self.BASE_LEVEL) * rise * (1 - fall)

    def _node(self, img, d, x, y, color, level, flash, amber=False):
        S = self.S
        col = lerpc((14, 20, 50), color, level)
        if flash > 0:
            col = lerpc(col, (240, 252, 255), min(1, flash) * 0.6)
        r = 11 if amber else 8
        d.ellipse(((x - r - 1.5) * S, (y - r - 1.5) * S, (x + r + 1.5) * S, (y + r + 1.5) * S), fill=INK + (255,))
        d.ellipse(((x - r + 1.5) * S, (y - r + 1.5) * S, (x + r - 1.5) * S, (y + r - 1.5) * S), fill=col + (255,))

    def _ring(self, d, x, y, rad, color, alpha, width=2.0):
        S = self.S
        if alpha <= 0.01:
            return
        d.ellipse(((x - rad) * S, (y - rad) * S, (x + rad) * S, (y + rad) * S),
                  outline=color + (int(255 * alpha),), width=max(1, int(width * S)))

    # ---- frame ----------------------------------------------------------
    def render(self, c):
        S = self.S
        img = self.base.copy()
        d = ImageDraw.Draw(img, "RGBA")
        flash = {}
        rings = []

        # packet flows: lit edges + arrival events
        packets = []
        for start, color, light, path in self.FLOWS:
            raw = (c - start) / self.HOP
            for h in range(3):
                p, q = path[h], path[h + 1]
                done_t = start + (h + 1) * self.HOP
                frac, a = None, 0.0
                if raw >= h + 1:
                    frac, a = 1.0, 0.8 * (1 - clamp01((c - done_t) / 0.9)) ** 1.5
                elif raw > h:
                    f = raw - h
                    frac, a = 0.5 * f + 0.5 * ease_io(f), 0.9
                if frac is not None and a > 0.01:
                    e = (lerp(p[0], q[0], frac) * S, lerp(p[1], q[1], frac) * S)
                    d.line([(p[0] * S, p[1] * S), e], fill=color + (int(60 * a),), width=7)
                    d.line([(p[0] * S, p[1] * S), e], fill=lerpc(color, WHITE, 0.25) + (int(255 * a),), width=3)
                # arrival event at path[h+1]
                age = c - done_t
                if 0 <= age < 0.8:
                    big = (h == 2)
                    k = age / (0.8 if big else 0.55)
                    if k < 1:
                        rings.append((q[0], q[1], (12.5 if big else 9.5) + (28 if big else 15) * ease_out(k),
                                      AMBER if big else color, 0.6 * (1 - k)))
                    flash[q] = max(flash.get(q, 0), 1 - age / (0.5 if big else 0.3))
            age0 = c - start
            if 0 <= age0 < 0.5:
                rings.append((path[0][0], path[0][1], 9.5 + 12 * ease_out(age0 / 0.5), color, 0.55 * (1 - age0 / 0.5)))
                flash[path[0]] = max(flash.get(path[0], 0), 1 - age0 / 0.3)
            if -0.0 < raw < 3 + 0.25 / self.HOP:
                packets.append((start, color, light, path, raw))

        # scan ignition rings
        for j, col_x in enumerate(self.COLS):
            age = c - self.t_pass(col_x)
            if 0 <= age < 0.5:
                for (x, y) in self.LAYERS[j]:
                    rings.append((x, y, 9.5 + 9 * ease_out(age / 0.5), self.LAYER_COL[j], 0.40 * (1 - age / 0.5)))
        for x, y, rad, col, a in rings:
            self._ring(d, x, y, rad, col, a)

        # nodes
        for j, layer in enumerate(self.LAYERS):
            lv = self.level(c, self.COLS[j])
            for (x, y) in layer:
                if (x, y) == self.OUT:
                    continue
                self._node(img, d, x, y, self.LAYER_COL[j], lv, flash.get((x, y), 0))
        lv = self.level(c, 1130)
        ox, oy = self.OUT
        self._ring(d, ox, oy, 19, AMBER, 0.45 * (0.35 + 0.65 * (lv - self.BASE_LEVEL) / (1 - self.BASE_LEVEL)))
        self._node(img, d, ox, oy, AMBER, 0.55 + 0.45 * (lv - self.BASE_LEVEL) / (1 - self.BASE_LEVEL),
                   flash.get(self.OUT, 0), amber=True)
        f_out = flash.get(self.OUT, 0)
        if f_out > 0:
            paste_glow(img, ox * S, oy * S, 46 * S, AMBER, 0.30 * f_out)

        # axis ticks (light up with their column)
        for j, col_x in enumerate(self.COLS):
            lv = (self.level(c, col_x) - self.BASE_LEVEL) / (1 - self.BASE_LEVEL)
            col = lerpc(EDGE, self.LAYER_COL[j], 0.5 + 0.5 * lv)
            d.line([(col_x * S, 276 * S), (col_x * S, 290 * S)], fill=col + (int(255 * (0.30 + 0.60 * lv)),), width=3)

        # packets
        for start, color, light, path, raw in packets:
            hop = min(int(max(raw, 0)), 2)
            f = clamp01(raw - hop) if raw < 3 else 1.0
            u = hop + (0.5 * f + 0.5 * ease_io(f)) if raw < 3 else 3.0

            def pos(uu):
                uu = max(0.0, min(3.0, uu))
                h = min(int(uu), 2)
                fr = uu - h
                p, q = path[h], path[h + 1]
                return lerp(p[0], q[0], fr), lerp(p[1], q[1], fr)

            vis = clamp01(raw * self.HOP / 0.12) * (1 - clamp01((raw - 3) * self.HOP / 0.2))
            if vis <= 0.01:
                continue
            for k in range(9, 0, -1):
                tx, ty = pos(u - k * 0.05)
                rr = 4.0 * (1 - k / 11.0)
                d.ellipse(((tx - rr) * S, (ty - rr) * S, (tx + rr) * S, (ty + rr) * S),
                          fill=color + (int(150 * vis * (1 - k / 10.0)),))
            px, py = pos(u)
            paste_glow(img, px * S, py * S, 20 * S, color, 0.50 * vis)
            d.ellipse(((px - 4.6) * S, (py - 4.6) * S, (px + 4.6) * S, (py + 4.6) * S), fill=color + (int(255 * vis),))
            d.ellipse(((px - 2.6) * S, (py - 2.6) * S, (px + 2.6) * S, (py + 2.6) * S), fill=light + (int(255 * vis),))
            # matching dot travelling along the axis
            d.ellipse(((px - 2.6) * S, (283 - 2.6) * S, (px + 2.6) * S, (283 + 2.6) * S), fill=light + (int(220 * vis),))

        # name (per-character reveal as the scan passes, letter-spacing settle)
        g = ease_out((c - 0.15) / 1.9)
        out = smooth((c - self.FADE_START) / self.FADE_DUR)
        for i, item in enumerate(self.chars):
            if item is None:
                continue
            mask, col, ox_, oy_ = item
            u = ease_out((c - self.t_pass(self.char_x[i])) / 0.45)
            alpha = u * (1 - out)
            if alpha <= 0.01:
                continue
            dx = -(1 - u) * 8 + (1 - g) * i * 1.6 + out * i * 1.0
            lut = [int(v * alpha) for v in range(256)]
            img.paste(col, (int(round((self.char_x[i] + dx) * S + ox_)), int(round(self.nbase * S + oy_))), mask.point(lut))

        # cyan scan line
        sx = -20 + 1240 * clamp01((c - self.SCAN_START) / self.SCAN_DUR)
        if 0 < (c - self.SCAN_START) < self.SCAN_DUR + 0.08:
            img.paste(self.scan_col, (int(round(sx * S)) - self.scan_tw, 0), self.scan_mask)

        return finish_rgb(img, self.W, self.H, self.cmask)

    def frames(self):
        for k in range(self.N):
            yield self.render((self.C0 + k * self.DT) % self.T)


# ========================================================== 2. FOCUS CYCLE ==
class Focus:
    W, H, S = 900, 90, 2
    DELAY = 8
    THR, COLORS, DITHER = 5, 96, "bayer:bayer_scale=4"
    PHRASES = ["Machine Learning Systems", "Computer Vision", "Multimodal AI",
               "Python & Data Engineering", "Model Evaluation", "Software Engineering"]
    P = 1.52  # seconds per phrase (19 frames)
    N = 6 * 19
    DT = DELAY / 100.0
    T = N * DT
    C0 = 0.45
    TX, BASELINE = 158, 56
    WIN = (150, 14, 700, 66)  # clip window (x0, y0, x1, y1) in 1x px

    def __init__(self):
        S, W, H = self.S, self.W, self.H
        self.cmask = corner_mask(W, H, 14)
        img = gradient_bg(W, H, S, glows=False, grid=30, grid_alpha=0.05)
        d = ImageDraw.Draw(img, "RGBA")
        d.rounded_rectangle((S, S, W * S - S - 1, H * S - S - 1), radius=13 * S, outline=PERI + (82,), width=2 * S)
        d.ellipse((37 * S, 43 * S, 43 * S, 49 * S), fill=CYAN + (255,))
        draw_tracked(d, 52 * S, 51 * S, "FOCUS", inter("Bold", 12 * S), CYAN_L + (235,), 2.4 * S)
        d.line([(132 * S, 24 * S), (132 * S, 66 * S)], fill=EDGE + (70,), width=2)
        self.base = img
        self.pfont = inter("SemiBold", 36 * S)
        self.cfont = mono(15 * S)

    def _text_layer(self, text, font, x, baseline, dy, alpha, color, win):
        S = self.S
        x0, y0, x1, y1 = win
        m = Image.new("L", ((x1 - x0) * S, (y1 - y0) * S), 0)
        ImageDraw.Draw(m).text(((x - x0) * S, (baseline - y0 + dy) * S), text, font=font, fill=255, anchor="ls")
        lut = [int(v * alpha) for v in range(256)]
        return Image.new("RGB", m.size, color), m.point(lut), (x0 * S, y0 * S)

    def render(self, c):
        S = self.S
        img = self.base.copy()
        d = ImageDraw.Draw(img, "RGBA")
        idx = int(c // self.P) % 6
        t = c - idx * self.P
        IN, OUT = 0.30, 0.20
        # phrase: previous one leaves while the next arrives
        layers = []
        a_in = ease_out(t / IN)
        layers.append((idx, 16 * (1 - a_in), a_in))
        if t < IN * 0.7:  # previous phrase (still leaving)
            pass
        prev = (idx - 1) % 6
        t_prev = t + self.P
        a_out = clamp01((t_prev - (self.P - OUT)) / OUT)
        if a_out < 1:
            layers.append((prev, -16 * ease_io(a_out), 1 - a_out))
        for i, dy, al in layers:
            if al <= 0.01:
                continue
            col, mask, pos = self._text_layer(self.PHRASES[i], self.pfont, self.TX, self.BASELINE, dy, al,
                                              (240, 244, 255), self.WIN)
            img.paste(col, pos, mask)
        # counter
        cwin = (760, 30, 880, 64)
        for i, dy, al in layers:
            if al <= 0.01:
                continue
            txt = f"{i + 1:02d} / 06"
            tw = self.cfont.getlength(txt) / S
            col, mask, pos = self._text_layer(txt, self.cfont, 864 - tw, 52, dy * 0.35, al * 0.75, BLUE_L, cwin)
            img.paste(col, pos, mask)
        # progress ticks
        wrap = smooth((c - (self.T - 0.35)) / 0.35)
        for j in range(6):
            x0 = self.TX + j * 44
            if j < idx:
                a = lerp(0.45, 0.15, wrap)
                fill_w = 1.0
            elif j == idx:
                fill_w = clamp01(t / self.P)
                a = 0.15
            else:
                a, fill_w = 0.15, 0.0
            if j == 5 and idx == 5:
                fill_w_show = fill_w
            d.rounded_rectangle((x0 * S, 73 * S, (x0 + 36) * S, 76 * S), radius=int(1.5 * S), fill=EDGE + (int(255 * a),))
            if j == idx:
                fw = max(0.0, fill_w * (1 - wrap)) if idx == 5 else fill_w
                if fw > 0.02:
                    wpx = 36 * fw
                    d.rounded_rectangle((x0 * S, 73 * S, (x0 + wpx) * S, 76 * S), radius=int(1.5 * S), fill=CYAN + (235,))
                    d.ellipse(((x0 + wpx - 2.2) * S, 72.2 * S, (x0 + wpx + 2.2) * S, 76.8 * S), fill=AMBER + (255,))
            elif j < idx:
                d.rounded_rectangle((x0 * S, 73 * S, (x0 + 36) * S, 76 * S), radius=int(1.5 * S), fill=CYAN + (int(255 * lerp(0.55, 0.0, wrap) * (1 if idx < 6 else 0)),))
        return finish_rgb(img, self.W, self.H, self.cmask)

    def frames(self):
        for k in range(self.N):
            yield self.render((self.C0 + k * self.DT) % self.T)


# ======================================================== 3. FLOW DIVIDER ==
class Divider:
    W, H, S = 1200, 24, 4
    DELAY = 8
    THR, COLORS, DITHER = 0, 64, "none"
    N = 52
    DT = DELAY / 100.0
    T = N * DT
    TRAVEL = 3.4
    C0 = 0.45
    NODES = [(560, 5, BLUE), (600, 7, AMBER), (640, 5, CYAN)]
    STOPS = [(0.0, CYAN), (0.3, BLUE), (0.5, INDIGO), (0.7, BLUE), (1.0, CYAN)]

    def _line_col(self, x):
        u = x / self.W
        for (a, ca), (b, cb) in zip(self.STOPS, self.STOPS[1:]):
            if a <= u <= b:
                return lerpc(ca, cb, (u - a) / (b - a))
        return CYAN

    def render(self, c):
        S, W, H = self.S, self.W, self.H
        img = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        cy = 12
        # line: solid core that dissolves into dots at both ends
        for x in range(110, 1090):
            d.rectangle((x * S, (cy - 1) * S, (x + 1) * S - 1, (cy + 1) * S - 1), fill=self._line_col(x) + (255,))
        gap, x = 8, 104
        for k in range(6):
            for sgn in (-1, 1):
                px = x if sgn < 0 else W - x
                d.ellipse(((px - 1.2) * S, (cy - 1.2) * S, (px + 1.2) * S, (cy + 1.2) * S), fill=self._line_col(px) + (255,))
            x -= gap
            gap += 4
        # connectors between the three nodes
        for a, b in ((565, 593), (607, 635)):
            d.rectangle((a * S, int((cy - 0.75) * S), b * S, int((cy + 0.75) * S)), fill=(124, 140, 196, 255))
        # packet position
        px = -30 + (W + 60) * clamp01(c / self.TRAVEL)
        # nodes swell as the packet passes
        for nx, r, col in self.NODES:
            bump = math.exp(-(((px - nx) / 26.0) ** 2))
            rr = r * (1 + 0.30 * bump)
            cc = lerpc(col, (235, 252, 255), 0.45 * bump)
            d.ellipse(((nx - rr - 1.5) * S, (cy - rr - 1.5) * S, (nx + rr + 1.5) * S, (cy + rr + 1.5) * S), fill=(11, 16, 48, 255))
            d.ellipse(((nx - rr + 0.0) * S, (cy - rr + 0.0) * S, (nx + rr - 0.0) * S, (cy + rr - 0.0) * S), fill=cc + (255,))
        # packet with a short solid-colour tail
        if 0 <= c <= self.TRAVEL:
            for k in range(7, 0, -1):
                tx = px - k * 7
                rr = 2.7 - 0.28 * k
                col = lerpc(CYAN, INDIGO, k / 7.0)
                d.ellipse(((tx - rr) * S, (cy - rr) * S, (tx + rr) * S, (cy + rr) * S), fill=col + (255,))
            d.ellipse(((px - 4.6) * S, (cy - 4.6) * S, (px + 4.6) * S, (cy + 4.6) * S), fill=CYAN + (255,))
            d.ellipse(((px - 2.9) * S, (cy - 2.9) * S, (px + 2.9) * S, (cy + 2.9) * S), fill=(200, 250, 255, 255))
        out = img.resize((W, H), Image.LANCZOS)
        a = out.getchannel("A").point(lambda v: 255 if v >= 128 else 0)
        out.putalpha(a)
        return out

    def frames(self):
        for k in range(self.N):
            yield self.render((self.C0 + k * self.DT) % self.T)


# ------------------------------------------------------------------- main --
JOBS = {
    "hero": (Hero, "hero-loop.gif"),
    "focus": (Focus, "focus-cycle.gif"),
    "divider": (Divider, "flow-divider.gif"),
}


def build(name, keep_frames=None):
    cls, fname = JOBS[name]
    obj = cls()
    tmp = keep_frames or tempfile.mkdtemp(prefix=f"gif_{name}_")
    os.makedirs(tmp, exist_ok=True)
    frames = obj.frames()
    if cls.THR > 0:
        frames = temporal_filter(frames, cls.THR)
    for k, fr in enumerate(frames, start=1):
        fr.save(os.path.join(tmp, f"f{k:03d}.png"), compress_level=1)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, fname)
    enc = encode_gif_alpha if name == "divider" else encode_gif
    enc(tmp, out, cls.DELAY, cls.COLORS, cls.DITHER)
    if not keep_frames:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"{fname}: {cls.N} frames x {cls.DELAY * 10} ms = {cls.N * cls.DELAY / 100:.2f} s, "
          f"{os.path.getsize(out) / 1024:.0f} KiB")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    keep = None
    for a in sys.argv[1:]:
        if a.startswith("--keep-frames="):
            keep = a.split("=", 1)[1]
    targets = args or list(JOBS)
    for t in targets:
        if keep:
            build(t, os.path.join(keep, t))
        else:
            build(t)
