"""Render the built-in drum kits to samples/<kit>/<instrument>.wav.

Everything is synthesized here, so the samples are original (no licensing).
Run: python3 tools/make_samples.py   (needs numpy + scipy)
"""
import os
import wave

import numpy as np
from scipy import signal

SR = 44100
OUT = os.path.join(os.path.dirname(__file__), '..', 'samples')
rng = np.random.default_rng(7)


def tt(d):
    return np.arange(int(SR * d)) / SR


def env(t, decay, attack=0.0015):
    e = np.exp(-t / decay)
    a = int(SR * attack)
    if a:
        e[:a] *= np.linspace(0, 1, a)
    return e


def filt(x, kind, f, order=2):
    f = np.atleast_1d(f) / (SR / 2)
    sos = signal.butter(order, f if len(f) > 1 else f[0], kind, output='sos')
    return signal.sosfilt(sos, x)


def sat(x, drive):
    return np.tanh(x * drive) / np.tanh(drive)


def noise(n):
    return rng.uniform(-1, 1, n)


def reverb(x, tail=0.25, mix=0.15):
    n = int(SR * tail)
    ir = noise(n) * np.exp(-np.arange(n) / (SR * tail / 5))
    ir[0] = 0
    wet = signal.fftconvolve(x, ir)[: len(x)]
    wet *= np.max(np.abs(x)) / (np.max(np.abs(wet)) + 1e-9)
    return x * (1 - mix) + wet * mix


def crush(x, bits):
    q = 2 ** (bits - 1)
    return np.round(x * q) / q


def finish(x, peak=0.89, fade_ms=8):
    x = x - np.mean(x)
    x = x / (np.max(np.abs(x)) + 1e-9) * peak
    n = int(SR * fade_ms / 1000)
    x[-n:] *= np.linspace(1, 0, n)
    return x


def kick(f0, f1, tau, decay, drive, click=0.25, dur=0.6, lowpass=None):
    t = tt(dur)
    f = f1 + (f0 - f1) * np.exp(-t / tau)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * env(t, decay)
    c = filt(noise(len(t)), 'high', 2500) * env(t, 0.004) * click
    x = sat(body + c, drive)
    return filt(x, 'low', lowpass) if lowpass else x


def snare(tone_f, tone_decay, noise_decay, lo, hi, noise_amt=0.9, dur=0.45, drive=1.5):
    t = tt(dur)
    tone = (np.sin(2 * np.pi * tone_f * t) + 0.5 * np.sin(2 * np.pi * tone_f * 1.9 * t)) * env(t, tone_decay)
    nz = filt(noise(len(t)), 'band', [lo, hi]) * env(t, noise_decay)
    return sat(tone * 0.7 + nz * noise_amt, drive)


def clap(dur=0.5, lo=900, hi=3800, tail=0.11):
    t = tt(dur)
    n = noise(len(t))
    out = np.zeros(len(t))
    for i, off in enumerate([0, 0.011, 0.023, 0.036]):
        o = int(SR * off)
        seg = t[: len(t) - o]
        out[o:] += n[: len(seg)] * env(seg, 0.006 if i < 3 else tail)
    return sat(filt(out, 'band', [lo, hi]), 2.0)


def hat(decay, dur=0.4, metal=0.6):
    t = tt(dur)
    freqs = np.array([205.3, 304.4, 369.6, 522.7, 540.0, 800.0]) * 1.7
    sq = sum(np.sign(np.sin(2 * np.pi * f * t)) for f in freqs) / len(freqs)
    x = sq * metal + noise(len(t)) * (1 - metal)
    return filt(filt(x, 'high', 6500, 3), 'low', 15000) * env(t, decay, 0.0008)


def bass808(root_hz=55.0, decay=1.0, drive=2.5, glide=0.025, dur=1.6):
    t = tt(dur)
    f = root_hz * (1 + 0.8 * np.exp(-t / glide))
    x = np.sin(2 * np.pi * np.cumsum(f) / SR)
    e = env(t, decay, 0.003)
    return sat(x * e, drive) * 0.9


KITS = {
    'trap': {
        'kick': lambda: kick(185, 46, 0.03, 0.42, 3.0, click=0.2, dur=0.8),
        'snare': lambda: reverb(snare(190, 0.07, 0.18, 1400, 9000, 1.0), 0.3, 0.2),
        'clap': lambda: reverb(clap(), 0.35, 0.22),
        'hat': lambda: hat(0.03),
        'ohat': lambda: hat(0.22, metal=0.5),
        'bass': lambda: bass808(decay=1.1, drive=3.0),
    },
    'boombap': {
        'kick': lambda: kick(150, 52, 0.035, 0.22, 2.2, click=0.12, dur=0.5, lowpass=3500),
        'snare': lambda: filt(snare(205, 0.08, 0.13, 900, 7000, 0.9, drive=2.2), 'low', 8000),
        'clap': lambda: filt(clap(lo=800, hi=3000, tail=0.09), 'low', 7000),
        'hat': lambda: filt(hat(0.035, metal=0.4), 'low', 11000),
        'ohat': lambda: filt(hat(0.2, metal=0.4), 'low', 10000),
        'bass': lambda: bass808(decay=0.5, drive=1.8, dur=1.0),
    },
    'pop': {
        'kick': lambda: kick(210, 54, 0.02, 0.28, 1.6, click=0.35, dur=0.5),
        'snare': lambda: reverb(snare(220, 0.06, 0.2, 1800, 10000, 0.85, drive=1.3), 0.4, 0.28),
        'clap': lambda: reverb(clap(lo=1200, hi=4500), 0.45, 0.3),
        'hat': lambda: hat(0.028, metal=0.55),
        'ohat': lambda: hat(0.3, metal=0.5),
        'bass': lambda: bass808(decay=0.7, drive=1.5, dur=1.2),
    },
    'lofi': {
        'kick': lambda: crush(kick(140, 50, 0.04, 0.2, 1.8, click=0.05, dur=0.5, lowpass=1800), 10),
        'snare': lambda: crush(filt(snare(200, 0.09, 0.17, 700, 5000, 0.8, drive=1.8), 'low', 5000), 10),
        'clap': lambda: crush(filt(clap(lo=700, hi=2500), 'low', 4500), 10),
        'hat': lambda: crush(filt(hat(0.04, metal=0.3), 'low', 8000), 9),
        'ohat': lambda: crush(filt(hat(0.18, metal=0.3), 'low', 7500), 9),
        'bass': lambda: filt(bass808(decay=0.6, drive=1.4, dur=1.0), 'low', 900),
    },
    'latin': {
        'kick': lambda: kick(175, 50, 0.025, 0.3, 2.4, click=0.3, dur=0.6),
        'snare': lambda: snare(240, 0.045, 0.1, 2200, 10000, 1.0, dur=0.3, drive=2.0),
        'clap': lambda: reverb(clap(lo=1000, hi=4200, tail=0.08), 0.3, 0.2),
        'hat': lambda: hat(0.03),
        'ohat': lambda: hat(0.2),
        'bass': lambda: bass808(decay=0.8, drive=2.4, dur=1.2),
    },
}


def write_wav(path, x):
    pcm = np.int16(np.clip(x, -1, 1) * 32767)
    with wave.open(path, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


if __name__ == '__main__':
    for kit, insts in KITS.items():
        d = os.path.join(OUT, kit)
        os.makedirs(d, exist_ok=True)
        for name, fn in insts.items():
            x = finish(fn())
            write_wav(os.path.join(d, name + '.wav'), x)
            print(f'{kit}/{name}: {len(x) / SR:.2f}s peak {np.max(np.abs(x)):.2f}')
