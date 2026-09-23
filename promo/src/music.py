"""Original soundtrack for the RUSSIA track-suit promo. 125 BPM, A minor."""
import numpy as np
from scipy import signal
import wave

SR = 48000
BPM = 125
BEAT = 60 / BPM            # 0.48 s
BAR = 4 * BEAT             # 1.92 s
NBARS = 12
DUR = NBARS * BAR + 2.2    # tail for the last chord
N = int(DUR * SR)
rng = np.random.default_rng(7)

def z(): return np.zeros(N)
def idx(t): return int(round(t * SR))
def midi(m): return 440.0 * 2 ** ((m - 69) / 12)

def env_adsr(n, a, d, s, r, gate):
    t = np.arange(n) / SR
    e = np.where(t < a, t / max(a, 1e-4), s + (1 - s) * np.exp(-(t - a) / max(d, 1e-4)))
    rel = t > gate
    e[rel] = e[rel] * np.exp(-(t[rel] - gate) / max(r, 1e-4))
    return e

_tables = {}
def saw_table(f):
    """Band-limited saw single period (additive, harmonics below 16 kHz)."""
    key = round(f, 2)
    if key not in _tables:
        L = 2048; ph = np.arange(L) / L * 2 * np.pi
        k = np.arange(1, max(2, int(16000 / f)) + 1)
        _tables[key] = (np.sin(np.outer(ph, k)) @ ((-1) ** (k + 1) / k)) * (2 / np.pi)
    return _tables[key]

def osc_saw(f, n, phase0=0.0):
    tab = saw_table(f); L = len(tab)
    ph = (phase0 + f * np.arange(n) / SR) % 1.0
    return tab[(ph * L).astype(int)]

def lp(x, fc, order=2):
    b, a = signal.butter(order, min(fc, SR / 2 - 100) / (SR / 2)); return signal.lfilter(b, a, x)
def hp(x, fc, order=2):
    b, a = signal.butter(order, fc / (SR / 2), 'high'); return signal.lfilter(b, a, x)
def bp(x, lo, hi, order=2):
    b, a = signal.butter(order, [lo / (SR / 2), hi / (SR / 2)], 'band'); return signal.lfilter(b, a, x)

def add(buf, t, x, g=1.0):
    i = idx(t); j = min(N, i + len(x)); buf[i:j] += g * x[: j - i]

def sweep_lp(x, f_from, f_to, block=256):
    """Time-varying low-pass (block-wise one-pole cascade)."""
    y = np.zeros_like(x); s1 = s2 = 0.0
    nb = int(np.ceil(len(x) / block))
    fs = np.geomspace(f_from, f_to, nb)
    for b in range(nb):
        a = 1 - np.exp(-2 * np.pi * fs[b] / SR)
        seg = x[b * block:(b + 1) * block]
        z1 = signal.lfilter([a], [1, a - 1], seg, zi=[s1 * (1 - a)])
        o1, s1v = z1; s1 = o1[-1] if len(o1) else s1
        z2 = signal.lfilter([a], [1, a - 1], o1, zi=[s2 * (1 - a)])
        o2, _ = z2; s2 = o2[-1] if len(o2) else s2
        y[b * block:(b + 1) * block] = o2
    return y

# ---------- instruments ----------
def kick():
    n = idx(0.45); t = np.arange(n) / SR
    f = 45 + 110 * np.exp(-t / 0.035)
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(ph) * np.exp(-t / 0.22)
    click = hp(rng.standard_normal(n), 2500) * np.exp(-t / 0.004) * 0.35
    return np.tanh(1.6 * (body + click))

def clap():
    n = idx(0.35); t = np.arange(n) / SR
    nz = bp(rng.standard_normal(n), 900, 5000)
    e = np.zeros(n)
    for off in (0, 0.011, 0.022):
        m = t >= off; e[m] += np.exp(-(t[m] - off) / 0.006)
    m = t >= 0.03; e[m] += 0.7 * np.exp(-(t[m] - 0.03) / 0.09)
    return nz * e * 0.9

def hat(open_=False):
    n = idx(0.25 if open_ else 0.06); t = np.arange(n) / SR
    return hp(rng.standard_normal(n), 7500, 4) * np.exp(-t / (0.08 if open_ else 0.015)) * 0.5

def impact(len_s=2.5):
    n = idx(len_s); t = np.arange(n) / SR
    f = 30 + 70 * np.exp(-t / 0.15)
    boom = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.8)
    crash = hp(rng.standard_normal(n), 3000) * np.exp(-t / 0.7) * 0.25
    return np.tanh(1.3 * boom) + crash

def riser(len_s):
    n = idx(len_s); t = np.arange(n) / SR
    nz = rng.standard_normal(n)
    y = sweep_lp(nz, 300, 12000) * (t / len_s) ** 2
    f = 220 * 2 ** (2 * t / len_s)
    tone = np.sin(2 * np.pi * np.cumsum(f) / SR) * (t / len_s) ** 3 * 0.15
    return (hp(y, 200) * 0.6 + tone)

def whoosh(len_s=0.5, up=True):
    n = idx(len_s); t = np.arange(n) / SR
    nz = rng.standard_normal(n)
    y = sweep_lp(nz, 400 if up else 9000, 9000 if up else 400)
    e = np.sin(np.pi * np.clip(t / len_s, 0, 1)) ** 2
    return hp(y, 250) * e * 0.8

def supersaw(freqs, dur, voices=5, detune=0.12):
    n = idx(dur); out = np.zeros(n)
    for f in freqs:
        for v in range(voices):
            d = (v - (voices - 1) / 2) / ((voices - 1) / 2) * detune  # semitones
            out += osc_saw(f * 2 ** (d / 12), n, rng.random())
    return out / (len(freqs) * voices) * 2.2

def pluck(f, dur=0.22):
    n = idx(dur); t = np.arange(n) / SR
    x = osc_saw(f, n) + 0.5 * osc_saw(f * 2.003, n)
    return lp(x, 3500) * np.exp(-t / 0.09)

def bass_note(f, dur):
    n = idx(dur); t = np.arange(n) / SR
    x = osc_saw(f, n) * 0.6 + np.sin(2 * np.pi * f * t)
    return lp(np.tanh(1.5 * x), 900) * env_adsr(n, 0.005, 0.12, 0.6, 0.03, dur - 0.03)

# ---------- arrangement ----------
# Am - F - C - G, one chord per bar
CH = [(57, 60, 64), (53, 57, 60), (48, 52, 55), (55, 59, 62)]
ROOT = [33, 29, 36, 31]            # bass MIDI (A1, F1, C2, G1)
drums, bass, pads, pads2, arp, lead, fx = z(), z(), z(), z(), z(), z(), z()
K, C = kick(), clap()

DROP = 2          # bar where the full beat starts
FINAL = 11        # bar with the closing hit
for b in range(NBARS):
    t0 = b * BAR; c = b % 4
    chord = CH[c]
    # pads: all bars, brighter from the drop
    cut = 1800 if b < DROP else 7000
    for dst in (pads, pads2):   # independent L/R voices for width
        pad = supersaw([midi(m) for m in chord] + [midi(chord[0] + 12), midi(chord[2] + 12)], BAR + (2.2 if b == FINAL else 0.05))
        pad = lp(pad, cut) * env_adsr(len(pad), 0.02 if b >= DROP else 0.4, 0.5, 0.8, 0.9 if b == FINAL else 0.08,
                                      len(pad) / SR - (1.6 if b == FINAL else 0.08))
        add(dst, t0, pad, 0.9 if b >= DROP else 0.6)
    if b == FINAL:
        add(drums, t0, K, 1.0)
        continue
    # arp: 16ths, chord tones over two octaves
    tones = [chord[0], chord[1], chord[2], chord[0] + 12, chord[2], chord[1] + 12, chord[0] + 12, chord[2] + 12]
    for s in range(16):
        add(arp, t0 + s * BEAT / 4, pluck(midi(tones[s % 8] + 12)), 0.5 if b >= DROP else 0.35)
    if b < DROP:
        for s in range(8):                                # building hats in the intro
            if b == 1 or s % 2: add(drums, t0 + s * BEAT / 2, hat(), 0.5 + 0.2 * b)
        continue
    # four-on-the-floor + clap + hats
    for q in range(4):
        add(drums, t0 + q * BEAT, K, 0.6)
        add(drums, t0 + q * BEAT + BEAT / 2, hat(open_=True), 0.7)
        add(drums, t0 + q * BEAT + BEAT / 4, hat(), 0.5)
        add(drums, t0 + q * BEAT + 3 * BEAT / 4, hat(), 0.5)
        if q in (1, 3): add(drums, t0 + q * BEAT, C, 1.0)
        # off-beat bass
        add(bass, t0 + q * BEAT + BEAT / 2, bass_note(midi(ROOT[c] + 12), BEAT / 2 - 0.01), 0.45)
        add(bass, t0 + q * BEAT, bass_note(midi(ROOT[c]), BEAT / 2 - 0.02), 0.22)
    # clap roll into the final hit
    if b == FINAL - 1:
        for s in range(8):
            add(drums, t0 + 2 * BEAT + s * BEAT / 4, C, 0.25 + 0.08 * s)
    # lead hook from bar 6
    if b >= 6:
        hook = {0: [(0, 76, 1), (1, 74, .5), (1.5, 72, .5), (2, 74, 1), (3, 76, 1)],
                1: [(0, 77, 1.5), (1.5, 76, .5), (2, 72, 2)],
                2: [(0, 76, 1), (1, 79, 1), (2, 76, .5), (2.5, 74, .5), (3, 72, 1)],
                3: [(0, 74, 1.5), (1.5, 71, .5), (2, 74, 1), (3, 79, 1)]}[c]
        for beat, m, ln in hook:
            n = supersaw([midi(m)], ln * BEAT, voices=3, detune=0.08)
            n = lp(n, 6000) * env_adsr(len(n), 0.01, 0.2, 0.7, 0.05, ln * BEAT - 0.05)
            add(lead, t0 + beat * BEAT, n, 0.6)

# FX: riser into the drop, impacts, whooshes on section changes
add(fx, DROP * BAR - 3.2, riser(3.2), 0.55)
add(fx, DROP * BAR, impact(), 0.8)
add(fx, 6 * BAR - 0.45, whoosh(0.5), 0.5)
add(fx, 6 * BAR, impact(1.5), 0.35)
add(fx, FINAL * BAR - 1.6, riser(1.6), 0.35)
add(fx, FINAL * BAR, impact(3.0), 0.9)

# sidechain pump from the kick grid (drop section only)
sc = np.ones(N)
for b in range(DROP, FINAL):
    for q in range(4):
        i = idx(b * BAR + q * BEAT); n = idx(BEAT)
        t = np.arange(n) / SR
        sc[i:i + n] = np.minimum(sc[i:i + n], 1 - 0.65 * np.exp(-t / 0.09))

def reverb(x, rt=1.6, mix=0.25):
    n = idx(rt); t = np.arange(n) / SR
    irL = rng.standard_normal(n) * np.exp(-t / (rt / 6.9)); irR = rng.standard_normal(n) * np.exp(-t / (rt / 6.9))
    irL, irR = lp(irL, 6000), lp(irR, 6000)
    wl = signal.fftconvolve(x, irL)[:N]; wr = signal.fftconvolve(x, irR)[:N]
    wl /= np.max(np.abs(wl)) + 1e-9; wr /= np.max(np.abs(wr)) + 1e-9
    s = np.max(np.abs(x)) + 1e-9
    return x + mix * s * wl, x + mix * s * wr

def pan(x, p):  # p in [-1, 1]
    return x * np.sqrt((1 - p) / 2), x * np.sqrt((1 + p) / 2)

padL, _ = reverb(pads * sc, 2.2, 0.35); _, padR = reverb(pads2 * sc, 2.2, 0.35)
arpL, arpR = reverb(arp * sc, 1.2, 0.3)
# stereo arp ping-pong feel
d = idx(BEAT * 0.75)
arpR = np.concatenate([np.zeros(d), arpR[:-d]]) * 0.8 + arpR * 0.4
leadL, leadR = reverb(lead * sc, 1.8, 0.3)
fxL, fxR = reverb(fx, 2.5, 0.3)
dL, dR = drums, drums
bL, bR = bass * sc, bass * sc

L = padL + arpL * 0.9 + leadL + fxL + dL + bL
R = padR + arpR * 0.9 + leadR + fxR + dR + bR
mix = np.stack([L, R], 1)
mix = hp(mix.T, 28).T
b_, a_ = signal.butter(1, 3500 / (SR / 2), 'high'); mix = mix + 0.6 * signal.lfilter(b_, a_, mix.T).T   # air shelf
mix /= np.max(np.abs(mix))
mix = np.tanh(1.8 * mix) / np.tanh(1.8)               # glue / soft clip
fade = idx(1.2); mix[-fade:] *= np.linspace(1, 0, fade)[:, None] ** 2
mix *= 10 ** (-1 / 20) / np.max(np.abs(mix))

with wave.open('music.wav', 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((mix * 32767).astype('<i2').tobytes())
print(f"music.wav {DUR:.2f}s  bar={BAR}s  drop@{DROP*BAR:.2f}s  final@{FINAL*BAR:.2f}s")
