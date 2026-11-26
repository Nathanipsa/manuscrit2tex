import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from m2t import data as D  # noqa: E402
from m2t.vocab import Vocab  # noqa: E402

LIGNES = [  # (latex, largeur)
    (r"\[x^{2}\]", 40),
    (r"\[\frac{a}{b}\]", 30),
    (r"\[a+b=c\]", 100),
    (r"\[y\]", 2000),  # trop large, écartée
]


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "img").mkdir()
        with open(root / "idx.jsonl", "w", encoding="utf-8") as f:
            for k, (latex, w) in enumerate(LIGNES):
                img = np.full((16, w), 255, np.uint8)
                img[:, 0] = 0  # une colonne d'encre
                cv2.imwrite(str(root / f"img/{k}.png"), img)
                f.write(json.dumps({"image": f"img/{k}.png", "latex": latex, "w": w}) + "\n")
        self.idx = root / "idx.jsonl"
        self.vocab = D.build_vocab([self.idx])
        self.ds = D.LineDataset([self.idx], self.vocab, max_width=1024)

    def tearDown(self):
        self.tmp.cleanup()

    def test_exemple(self):
        self.assertEqual((len(self.ds), self.ds.n_rejected), (3, 1))
        x, seq = self.ds[1]
        self.assertEqual(tuple(x.shape), (1, 16, 30))
        # encre = 1, papier = 0, comme le padding
        self.assertEqual((x[0, 0, 0].item(), x[0, 0, 1].item()), (1.0, 0.0))
        self.assertEqual(self.vocab.decode(seq.tolist()), [r"\[", r"\frac", "{", "a", "}",
                                                          "{", "b", "}", r"\]"])
        self.assertEqual((seq[0].item(), seq[-1].item()), (Vocab.BOS, Vocab.EOS))

    def test_max_len(self):
        ds = D.LineDataset([self.idx], self.vocab, max_len=9)
        self.assertEqual(ds.n_rejected, 2)  # la fraction (11 ids) et la trop large

    def test_collate(self):
        images, widths, tgt_in, tgt_out = D.collate([self.ds[0], self.ds[2]])
        self.assertEqual(tuple(images.shape), (2, 1, 16, 104))  # 100 arrondi à 8
        self.assertEqual(widths.tolist(), [40, 100])
        self.assertEqual(images[0, :, :, 40:].abs().sum().item(), 0)
        s0, s2 = self.ds[0][1], self.ds[2][1]
        self.assertEqual(tgt_in.shape, tgt_out.shape)
        self.assertEqual(tgt_in[1].tolist(), s2[:-1].tolist())
        self.assertEqual(tgt_out[1].tolist(), s2[1:].tolist())
        # séquence courte : EOS à sa place dans tgt_out, PAD ensuite
        self.assertEqual(tgt_out[0, :len(s0) - 1].tolist(), s0[1:].tolist())
        self.assertTrue((tgt_out[0, len(s0) - 1:] == Vocab.PAD).all())

    def test_picklable(self):
        # forkserver picke le Dataset vers chaque worker
        ds = pickle.loads(pickle.dumps(self.ds))
        torch.testing.assert_close(ds[2][0], self.ds[2][0])


class TestSampler(unittest.TestCase):
    def setUp(self):
        self.w = np.random.default_rng(1).integers(20, 1024, 5000)

    def check(self, s):
        batches = list(s)
        flat = sorted(i for b in batches for i in b)
        self.assertEqual(flat, list(range(len(self.w))))  # chaque exemple une fois
        for b in batches:
            self.assertLessEqual(len(b) * self.w[b].max(), s.budget)
            self.assertLessEqual(len(b), s.max_batch)
        return batches

    def test_budget_et_couverture(self):
        s = D.WidthBucketSampler(self.w, 8192, max_batch=64, pool=1000)
        b = self.check(s)
        self.assertEqual(len(s), len(b))
        self.assertLess(D.padding_ratio(self.w, b), 0.05)

    def test_epoques(self):
        s = D.WidthBucketSampler(self.w, 8192, pool=1000)
        e0 = list(s)
        self.assertEqual(e0, list(s))  # même époque, mêmes batchs
        s.set_epoch(1)
        self.assertNotEqual(e0, self.check(s))

    def test_sans_melange(self):
        s = D.WidthBucketSampler(self.w, 8192, shuffle=False)
        b = self.check(s)
        self.assertTrue(np.all(np.diff([self.w[x].max() for x in b]) >= 0))

    def test_ligne_trop_large(self):
        with self.assertRaises(ValueError):
            D.WidthBucketSampler([100, 3000], 2048)


if __name__ == "__main__":
    unittest.main()
