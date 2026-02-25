"""
scripts/make_icon.py — Genere assets/trado.ico si absent.
Appelee par build.bat.
"""
import struct
import zlib
from pathlib import Path

ASSETS = Path(__file__).parent.parent / "assets"
ICO    = ASSETS / "trado.ico"

if ICO.exists():
    print("assets/trado.ico existe deja, skip.")
    raise SystemExit(0)

ASSETS.mkdir(exist_ok=True)

W, H = 32, 32

def make_png_32x32() -> bytes:
    pixels = []
    cx, cy = W // 2, H // 2
    r_outer = W // 2 - 2
    r_inner = W // 2 - 8
    for y in range(H):
        row = b"\x00"  # filtre PNG None
        for x in range(W):
            dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            if r_inner <= dist <= r_outer:
                row += bytes([0x26, 0xA6, 0x9A, 255])  # teal opaque
            else:
                row += bytes([0, 0, 0, 0])              # transparent
        pixels.append(row)

    raw = b"".join(pixels)
    compressed = zlib.compress(raw, 9)

    def chunk(name: bytes, data: bytes) -> bytes:
        length = struct.pack(">I", len(data))
        crc    = struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)
        return length + name + data + crc

    ihdr = struct.pack(">IIBBBBB", W, H, 8, 6, 0, 0, 0)
    png  = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", ihdr)
    png += chunk(b"IDAT", compressed)
    png += chunk(b"IEND", b"")
    return png

png_data = make_png_32x32()

# Format ICO : 1 image PNG embarquee
ico_header = struct.pack("<HHH", 0, 1, 1)          # magic, type=1(ICO), count=1
ico_dir    = struct.pack(
    "<BBBBHHII",
    32, 32,            # width, height
    0,                 # color count (0 = >256)
    0,                 # reserved
    1,                 # planes
    32,                # bit count
    len(png_data),     # taille image
    6 + 16,            # offset = header(6) + 1 dir entry(16)
)

with open(ICO, "wb") as f:
    f.write(ico_header + ico_dir + png_data)

print(f"assets/trado.ico cree ({ICO.stat().st_size} octets)")
