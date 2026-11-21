"""Scans de gabarits remplis → paires (image de ligne, LaTeX) sans annotation.

Pour chaque image (ou chaque page d'un PDF sorti de l'appli scanner) :
  1. lit le QR code pour savoir quelle page du manifest c'est ;
  2. trouve les marqueurs ArUco et redresse la page par homographie, à partir de
     leurs coins (positions connues en mm dans le manifest) ;
  3. découpe l'intérieur de chaque cadre, un peu en retrait du trait.

Sortie : data/lines/<lot>/<page>_<nn>.png (niveaux de gris, PX_PER_MM px/mm) et
data/lines/<lot>/<page>.json (latex, type, cadre vide ou non, scan d'origine).
Rescanner une page écrase ses découpes précédentes.

Usage : python scripts/crop_scans.py scan1.jpg scans.pdf ... [--debug]
        (--debug : écrit la page redressée avec les cadres en rouge dans debug/)
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PX_PER_MM = 12      # ≈ 305 dpi après redressement
INSET_MM = 1.2      # retrait sous le trait du cadre
MIN_MARKERS = 3     # 12 coins : l'homographie reste robuste à un marqueur caché


class ScanError(Exception):
    pass


def load_manifests(root=ROOT / "templates"):
    """ID de page → (manifest, page)."""
    index = {}
    for f in Path(root).glob("*/manifest.json"):
        m = json.loads(f.read_text(encoding="utf-8"))
        for p in m["pages"]:
            index[p["id"]] = (m, p)
    return index


def read_images(path):
    """Une image, ou chaque page d'un PDF rendue à 300 dpi."""
    path = Path(path)
    if path.suffix.lower() != ".pdf":
        img = cv2.imread(str(path))
        if img is None:
            raise ScanError(f"image illisible : {path}")
        return [img]
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["gs", "-q", "-dNOPAUSE", "-dBATCH", "-sDEVICE=png16m", "-r300",
                        f"-sOutputFile={tmp}/p%03d.png", str(path)], check=True)
        return [cv2.imread(str(f)) for f in sorted(Path(tmp).glob("p*.png"))]


def _gray(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _decode_qr(gray):
    text, _, _ = cv2.QRCodeDetector().detectAndDecode(gray)
    return text[4:] if text.startswith("m2t:") else None


def rectify(img, manifest, ppmm=PX_PER_MM):
    """Page redressée dans le repère du gabarit (1 px = 1/ppmm mm)."""
    gray = _gray(img)
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    corners, ids, _ = det.detectMarkers(gray)
    known = manifest["aruco"]["top_left_mm"]
    s = manifest["aruco"]["size_mm"]
    src, dst = [], []
    for c, i in zip(corners, [] if ids is None else ids.ravel()):
        if str(i) not in known:
            continue
        x, y = known[str(i)]
        src += c.reshape(4, 2).tolist()  # coins dans l'ordre HG, HD, BD, BG du marqueur
        dst += [[x * ppmm, y * ppmm], [(x + s) * ppmm, y * ppmm],
                [(x + s) * ppmm, (y + s) * ppmm], [x * ppmm, (y + s) * ppmm]]
    found = len(src) // 4
    if found < MIN_MARKERS:
        raise ScanError(f"{found} marqueur(s) ArUco trouvé(s), il en faut {MIN_MARKERS}")
    H, _ = cv2.findHomography(np.float32(src), np.float32(dst), cv2.RANSAC, 3.0)
    w, h = (round(v * ppmm) for v in manifest["page_mm"])
    return cv2.warpPerspective(gray, H, (w, h), flags=cv2.INTER_AREA, borderValue=255)


def identify(img, manifests, ppmm=PX_PER_MM):
    """(ID de page, page redressée). Le QR est cherché sur l'image brute, puis,
    à défaut, sur la page redressée (tous les lots partagent la même mise en page)."""
    gray = _gray(img)
    pid = _decode_qr(gray)
    if pid is None:
        any_manifest = next(iter(manifests.values()))[0]
        page = rectify(gray, any_manifest, ppmm)
        x, y, s = any_manifest["qr_mm"]
        m = 5  # mm de marge autour du QR
        roi = page[round((y - m) * ppmm):round((y + s + m) * ppmm),
                   round((x - m) * ppmm):round((x + s + m) * ppmm)]
        pid = _decode_qr(roi)
    if pid is None:
        raise ScanError("QR code illisible")
    if pid not in manifests:
        raise ScanError(f"page {pid} absente des manifests de templates/")
    return pid, rectify(gray, manifests[pid][0], ppmm)


def is_empty(crop):
    """Cadre laissé vide : presque aucun pixel nettement plus sombre que le papier."""
    paper = np.median(crop)
    return np.mean(crop < paper - 60) < 0.002


def crop_zones(page_img, page, ppmm=PX_PER_MM, inset=INSET_MM):
    out = []
    for z in page["zones"]:
        x, y, w, h = z["box_mm"]
        crop = page_img[round((y + inset) * ppmm):round((y + h - inset) * ppmm),
                        round((x + inset) * ppmm):round((x + w - inset) * ppmm)]
        out.append((z, crop))
    return out


def debug_image(page_img, page, ppmm=PX_PER_MM):
    img = cv2.cvtColor(page_img, cv2.COLOR_GRAY2BGR)
    for z in page["zones"]:
        x, y, w, h = z["box_mm"]
        cv2.rectangle(img, (round((x + INSET_MM) * ppmm), round((y + INSET_MM) * ppmm)),
                      (round((x + w - INSET_MM) * ppmm), round((y + h - INSET_MM) * ppmm)), (0, 0, 255), 3)
    return img


def save(pid, page_img, manifests, scan_name, out_root, debug=False):
    manifest, page = manifests[pid]
    out = Path(out_root) / manifest["batch"]
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob(f"{pid}_*.png"):
        old.unlink()
    records = []
    for z, crop in crop_zones(page_img, page):
        empty = bool(is_empty(crop))
        name = f"{pid}_{z['n']:02d}.png"
        if not empty:
            cv2.imwrite(str(out / name), crop)
        records.append({"image": None if empty else name, "latex": z["latex"],
                        "kind": z["kind"], "n": z["n"], "empty": empty, "scan": scan_name})
    (out / f"{pid}.json").write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    if debug:
        (out / "debug").mkdir(exist_ok=True)
        cv2.imwrite(str(out / "debug" / f"{pid}.jpg"), debug_image(page_img, page))
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("scans", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "lines")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    manifests = load_manifests()
    if not manifests:
        sys.exit("aucun manifest dans templates/")
    failures = 0
    for path in args.scans:
        for k, img in enumerate(read_images(path)):
            name = f"{path.name}#{k + 1}" if path.suffix.lower() == ".pdf" else path.name
            try:
                pid, page_img = identify(img, manifests)
            except ScanError as e:
                print(f"ÉCHEC {name} : {e}")
                failures += 1
                continue
            recs = save(pid, page_img, manifests, name, args.out, args.debug)
            n_empty = sum(r["empty"] for r in recs)
            note = f", {n_empty} cadre(s) vide(s)" if n_empty else ""
            print(f"{name} → {pid} : {len(recs) - n_empty} lignes{note}")
    if failures:
        sys.exit(f"{failures} image(s) non traitée(s)")


if __name__ == "__main__":
    main()
