"""MathWriting (tracés InkML) → images de lignes + index jsonl, sans extraire l'archive.

Lit le .tgz en flux, rend chaque tracé en niveaux de gris (encre noire sur blanc,
comme les découpes de crop_scans.py), recadré sur l'encre et ramené à la hauteur
HEIGHT. La cible est l'étiquette brute (label) passée dans normalize_latex, et non
le normalizedLabel de Google, qui suit d'autres règles (\\sinh → sinh, par exemple) :
les cibles doivent avoir la même forme canonique que nos lots.

Sortie : data/mathwriting/img/<split>/<id>.png et data/mathwriting/<split>.jsonl
(id, image, latex, w), plus rejets.jsonl (étiquettes refusées par le normaliseur).
Le dossier symbols/ (glyphes isolés) est ignoré.

Usage : python scripts/rasterize_mathwriting.py [--limit N] [--splits train,valid]
"""
import argparse
import json
import os
import sys
import tarfile
import xml.etree.ElementTree as ET
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from normalize_latex import LatexError, normalize  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = ROOT / "data/mathwriting/mathwriting-2024.tgz"
OUT = ROOT / "data/mathwriting"
SPLITS = ("train", "valid", "test", "synthetic")
HEIGHT = 96
MARGIN = 4         # px de blanc autour de l'encre
SS = 4             # suréchantillonnage, pour un trait anticrénelé comme sur un scan
PEN = 0.1          # épaisseur du trait / hauteur médiane d'un tracé (stylo sur lettre de 5 mm)
NS = "{http://www.w3.org/2003/InkML}"


def parse_inkml(data):
    """Octets InkML → (annotations, liste de tracés (n, 2) en float)."""
    root = ET.fromstring(data)
    ann = {e.get("type"): e.text for e in root.iter(NS + "annotation")}
    strokes = []
    for tr in root.iter(NS + "trace"):
        pts = [p.split()[:2] for p in tr.text.strip().split(",") if p.strip()]
        if pts:
            strokes.append(np.array(pts, dtype=float))
    return ann, strokes


def render(strokes, height=HEIGHT):
    """Tracés → image uint8 de hauteur `height`, encre noire (0) sur blanc (255)."""
    pts = np.concatenate(strokes)
    x0, y0 = pts.min(0)
    ink_h = max(np.ptp(pts[:, 1]), 1e-6)
    scale = (height - 2 * MARGIN) / ink_h
    width = int(np.ceil(np.ptp(pts[:, 0]) * scale)) + 2 * MARGIN
    # L'épaisseur suit la taille des glyphes, pas celle de l'image : une fraction
    # à trois étages ramenée à 96 px a des lettres plus petites, donc un trait plus fin.
    glyph = np.median([max(np.ptp(s[:, 1]), np.ptp(s[:, 0])) for s in strokes])
    thick = float(np.clip(PEN * glyph * scale, 1.0, 4.0))

    canvas = np.full((height * SS, width * SS), 255, np.uint8)
    t = max(1, round(thick * SS))
    for s in strokes:
        p = np.round(((s - (x0, y0)) * scale + MARGIN) * SS).astype(np.int32)
        if len(p) == 1:
            cv2.circle(canvas, tuple(int(v) for v in p[0]), max(1, t // 2), 0, -1, cv2.LINE_AA)
        else:
            cv2.polylines(canvas, [p], False, 0, t, cv2.LINE_AA)
    return cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA)


def process(item):
    """(split, id, octets) → ("ok", entrée d'index, png) ou ("rejet", info, None)."""
    split, sid, data = item
    ann, strokes = parse_inkml(data)
    label = ann.get("label") or ""
    if not strokes:
        return "rejet", {"id": sid, "split": split, "label": label, "err": "aucun tracé"}, None
    try:
        latex = normalize(r"\[" + label + r"\]")
    except LatexError as e:
        return "rejet", {"id": sid, "split": split, "label": label, "err": str(e)}, None
    img = render(strokes)
    ok, png = cv2.imencode(".png", img)
    entry = {"id": sid, "image": f"img/{split}/{sid}.png", "latex": latex, "w": img.shape[1]}
    return "ok", entry, png.tobytes()


def read_archive(path, splits, limit):
    """Parcourt le .tgz en flux et renvoie les (split, id, octets) des splits voulus."""
    seen = Counter()
    with tarfile.open(path, "r|gz") as tf:
        for m in tf:
            parts = m.name.split("/")
            if not m.isfile() or len(parts) != 3 or not parts[2].endswith(".inkml"):
                continue
            split, sid = parts[1], parts[2][:-6]
            if limit and all(seen[s] >= limit for s in splits):
                return
            if split not in splits or (limit and seen[split] >= limit):
                continue
            seen[split] += 1
            yield split, sid, tf.extractfile(m).read()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, default=ARCHIVE)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--splits", default=",".join(SPLITS))
    ap.add_argument("--limit", type=int, default=0, help="nombre max d'exemples par split")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args(argv)
    splits = args.splits.split(",")
    for s in splits:
        (args.out / "img" / s).mkdir(parents=True, exist_ok=True)

    # Index écrits sous .part puis renommés : un run interrompu ne laisse pas
    # un index tronqué qui aurait l'air complet.
    index = {s: open(args.out / f"{s}.jsonl.part", "w", encoding="utf-8") for s in splits}
    rejects = open(args.out / "rejets.jsonl.part", "w", encoding="utf-8")
    n_ok, n_rej = Counter(), Counter()
    with Pool(args.workers) as pool:
        items = read_archive(args.archive, splits, args.limit)
        for status, info, png in pool.imap(process, items, chunksize=64):
            s = info["split"] if status == "rejet" else info["image"].split("/")[1]
            if status == "ok":
                (args.out / info["image"]).write_bytes(png)
                index[s].write(json.dumps(info, ensure_ascii=False) + "\n")
                n_ok[s] += 1
            else:
                rejects.write(json.dumps(info, ensure_ascii=False) + "\n")
                n_rej[s] += 1
            total = sum(n_ok.values()) + sum(n_rej.values())
            if total % 5000 == 0:
                print(f"\r{total} tracés ({dict(n_ok)})", end="", flush=True)
    print()
    for s, f in index.items():
        f.close()
        os.replace(args.out / f"{s}.jsonl.part", args.out / f"{s}.jsonl")
    rejects.close()
    os.replace(args.out / "rejets.jsonl.part", args.out / "rejets.jsonl")
    for s in splits:
        print(f"{s:10s} {n_ok[s]:7d} images, {n_rej[s]:5d} rejets")


if __name__ == "__main__":
    main()
