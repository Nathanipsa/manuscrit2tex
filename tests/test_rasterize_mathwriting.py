import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import rasterize_mathwriting as R  # noqa: E402


def inkml(label, traces):
    tr = "".join(f'<trace id="{k}">' + ",".join(f"{x} {y} {10 * j}" for j, (x, y) in enumerate(t))
                 + "</trace>" for k, t in enumerate(traces))
    return (f'<ink xmlns="http://www.w3.org/2003/InkML"><annotation type="label">{label}</annotation>'
            f'<annotation type="normalizedLabel">ignoré</annotation>{tr}</ink>').encode()


# « 1 » vertical, puis un point isolé (tracé d'un seul point) à droite
UN = [[(100, 50), (100, 250)], [(160, 240)]]


class TestRender(unittest.TestCase):
    def test_hauteur_encre_et_marges(self):
        _, strokes = R.parse_inkml(inkml("1", UN))
        img = R.render(strokes)
        self.assertEqual(img.shape[0], R.HEIGHT)
        self.assertEqual(img.dtype, np.uint8)
        ink = np.argwhere(img < 128)
        self.assertGreater(len(ink), 50)
        # l'encre touche presque les marges haut et bas, et le point isolé est rendu
        self.assertLessEqual(ink[:, 0].min(), R.MARGIN + 2)
        self.assertGreaterEqual(ink[:, 0].max(), R.HEIGHT - R.MARGIN - 3)
        self.assertGreaterEqual(ink[:, 1].max(), img.shape[1] - R.MARGIN - 3)
        self.assertEqual(img[0].min(), 255)

    def test_meme_rendu_a_toute_echelle(self):
        # les coordonnées InkML dépendent de l'appareil : seul le rapport compte
        _, s1 = R.parse_inkml(inkml("1", UN))
        s2 = [3 * s + 7 for s in s1]
        self.assertEqual(R.render(s1).shape, R.render(s2).shape)


class TestArchive(unittest.TestCase):
    def test_bout_en_bout(self):
        files = {
            "mathwriting-2024/train/aaaa.inkml": inkml(r"2^\frac{1}{x} \le y", UN),
            "mathwriting-2024/synthetic/bbbb.inkml": inkml(r"\frac{1}", UN),
            "mathwriting-2024/symbols/cccc.inkml": inkml(r"\dot", UN),
            "mathwriting-2024/readme.md": b"x",
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            with tarfile.open(tmp / "mw.tgz", "w:gz") as tf:
                for name, data in files.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    tf.addfile(info, io.BytesIO(data))
            R.main(["--archive", str(tmp / "mw.tgz"), "--out", str(tmp / "out"),
                    "--splits", "train,synthetic", "--workers", "1"])
            train = [json.loads(l) for l in (tmp / "out/train.jsonl").read_text().splitlines()]
            self.assertEqual(len(train), 1)
            self.assertEqual(train[0]["latex"], r"\[2^{\frac{1}{x}}\leq y\]")
            img = cv2.imread(str(tmp / "out" / train[0]["image"]), cv2.IMREAD_GRAYSCALE)
            self.assertEqual(img.shape, (R.HEIGHT, train[0]["w"]))
            self.assertEqual((tmp / "out/synthetic.jsonl").read_text(), "")
            rej = [json.loads(l) for l in (tmp / "out/rejets.jsonl").read_text().splitlines()]
            self.assertEqual([r["id"] for r in rej], ["bbbb"])
            self.assertFalse((tmp / "out/img/symbols").exists())
            self.assertFalse(list((tmp / "out").glob("*.part")))


if __name__ == "__main__":
    unittest.main()
