"""Chaîne gabarit → photo simulée → découpe, sans imprimante.

Le gabarit est un vrai PDF produit par make_templates ; l'« encre » est un
disque posé à une position connue de chaque cadre. On vérifie qu'après une
photo déformée (perspective, rotation, ombre, flou, bruit, JPEG), chaque
disque retombe au même endroit de sa découpe à 1 mm près.
"""
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import crop_scans as cs  # noqa: E402
import make_templates as mt  # noqa: E402

DPI = 200
RENDER_PPMM = DPI / 25.4
BLOB = (0.3, 0.5)  # position de l'encre dans le cadre (fraction de largeur, de hauteur)
EMPTY_ZONE = 2     # ce cadre-là reste vide


def ink_position(z, ppmm, inset=cs.INSET_MM):
    """Centre attendu du disque dans la découpe, en px."""
    x, y, w, h = z["box_mm"]
    return ((BLOB[0] * w - inset) * ppmm, (BLOB[1] * h - inset) * ppmm)


def fake_photo(page, rng, upside_down=False):
    h, w = page.shape
    cw, ch = int(w * 1.3), int(h * 1.3)
    base = np.float32([[0.12, 0.08], [0.88, 0.1], [0.9, 0.92], [0.1, 0.9]]) * [cw, ch]
    dst = base + rng.uniform(-0.03, 0.03, (4, 2)) * [cw, ch]
    if upside_down:
        dst = np.roll(dst, 2, axis=0)
    H = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), np.float32(dst))
    photo = cv2.warpPerspective(page, H, (cw, ch), borderValue=110).astype(np.float32)
    shade = np.linspace(0.65, 1.0, cw)[None, :] * np.linspace(0.85, 1.0, ch)[:, None]
    photo = cv2.GaussianBlur(photo * shade, (5, 5), 1.2) + rng.normal(0, 4, photo.shape)
    ok, jpg = cv2.imencode(".jpg", np.clip(photo, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 70])
    return cv2.imdecode(jpg, cv2.IMREAD_GRAYSCALE)


@unittest.skipUnless(shutil.which("pdflatex") and shutil.which("gs"), "pdflatex ou gs absent")
class TestCropScans(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls.tmp.name)
        srcs = ["Une première ligne de texte assez longue pour un cadre",
                r"\[\int_{0}^{1}f(x)dx=\frac{1}{2}\]",
                "Soit $x\\in\\mathbb{R}$ un réel strictement positif.",
                r"\[A=\begin{pmatrix}1&2\\3&4\end{pmatrix}\]"] * 3
        lines = [mt._make_line(s, "test") for s in srcs]
        pages = mt.layout(lines, "test", 1)
        mt.build_pdf(pages, tmp)
        cls.manifest = mt.make_manifest("test", pages)
        cls.page = pages[0]
        cls.manifests = {cls.page["id"]: (cls.manifest, cls.page)}
        subprocess.run(["gs", "-q", "-dNOPAUSE", "-dBATCH", "-sDEVICE=pnggray", f"-r{DPI}",
                        f"-sOutputFile={tmp}/page.png", str(tmp / "gabarits.pdf")], check=True)
        cls.blank = cv2.imread(str(tmp / "page.png"), cv2.IMREAD_GRAYSCALE)
        cls.filled = cls.blank.copy()
        for z in cls.page["zones"]:
            if z["n"] == EMPTY_ZONE:
                continue
            x, y, w, h = z["box_mm"]
            c = ((x + BLOB[0] * w) * RENDER_PPMM, (y + BLOB[1] * h) * RENDER_PPMM)
            cv2.circle(cls.filled, (round(c[0]), round(c[1])), round(2 * RENDER_PPMM), 20, -1)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_manifest_matches_pdf(self):
        """Le trait de chaque cadre est bien là où le manifest le dit."""
        for z in self.page["zones"]:
            x, y, w, h = (v * RENDER_PPMM for v in z["box_mm"])
            top = self.blank[round(y) - 1:round(y) + 2, round(x + w * 0.2):round(x + w * 0.8)]
            left = self.blank[round(y + h * 0.2):round(y + h * 0.8), round(x) - 1:round(x) + 2]
            self.assertLess(top.min(), 220, f"cadre {z['n']}")
            self.assertLess(left.min(), 220, f"cadre {z['n']}")

    def check(self, photo):
        pid, page_img = cs.identify(photo, self.manifests)
        self.assertEqual(pid, self.page["id"])
        for z, crop in cs.crop_zones(page_img, self.page):
            with self.subTest(zone=z["n"]):
                self.assertEqual(cs.is_empty(crop), z["n"] == EMPTY_ZONE)
                if z["n"] == EMPTY_ZONE:
                    continue
                ys, xs = np.nonzero(crop < 100)
                ex, ey = ink_position(z, cs.PX_PER_MM)
                self.assertLess(abs(xs.mean() - ex), cs.PX_PER_MM, "écart > 1 mm en x")
                self.assertLess(abs(ys.mean() - ey), cs.PX_PER_MM, "écart > 1 mm en y")

    def test_photo(self):
        for seed in range(3):
            with self.subTest(seed=seed):
                self.check(fake_photo(self.filled, np.random.default_rng(seed)))

    def test_photo_upside_down(self):
        self.check(fake_photo(self.filled, np.random.default_rng(7), upside_down=True))

    def test_missing_marker(self):
        """Un marqueur caché (doigt, pli) : les trois autres suffisent."""
        img = self.filled.copy()
        x, y = mt.MARKERS[2]
        p = lambda v: round(v * RENDER_PPMM)  # noqa: E731
        img[p(y - 2):p(y + mt.MARKER + 2), p(x - 2):p(x + mt.MARKER + 2)] = 255
        self.check(fake_photo(img, np.random.default_rng(3)))

    def test_too_few_markers(self):
        img = np.full_like(self.filled, 255)
        with self.assertRaises(cs.ScanError):
            cs.identify(img, self.manifests)


if __name__ == "__main__":
    unittest.main()
