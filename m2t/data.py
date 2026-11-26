"""Données d'entraînement : images de lignes + cibles LaTeX, regroupées par largeur.

Format d'index commun (une ligne JSON par exemple) : {"image": chemin relatif au
dossier de l'index, "latex": ligne normalisée, "w": largeur en px}. C'est celui que
produit scripts/rasterize_mathwriting.py ; les lots scannés devront y être convertis.

Usage : python -m m2t.data   (construit data/mathwriting/vocab.json et affiche
le remplissage obtenu par le regroupement par largeur)
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from normalize_latex import tokenize  # noqa: E402

from m2t.vocab import Vocab  # noqa: E402

MATHWRITING = ROOT / "data/mathwriting"
STRIDE_W = 8  # réduction en largeur du CNNEncoder


def read_index(paths):
    """Lit un ou plusieurs index jsonl → liste de (chemin absolu de l'image, latex, w)."""
    rows = []
    for p in map(Path, paths):
        with open(p, encoding="utf-8") as f:
            for line in f:
                e = json.loads(line)
                rows.append((str(p.parent / e["image"]), e["latex"], int(e["w"])))
    return rows


def build_vocab(paths, min_freq=1):
    return Vocab.build((tokenize(latex) for _, latex, _ in read_index(paths)), min_freq)


class LineDataset(Dataset):
    """Paires (image (1, H, W) en float, encre = 1 et papier = 0 ; ids [BOS, ..., EOS]).

    Les exemples plus larges que max_width ou plus longs que max_len tokens (BOS et
    EOS compris) sont écartés, pas redimensionnés : rétrécir l'image changerait la
    taille des glyphes par rapport aux autres lignes.
    """

    def __init__(self, paths, vocab, max_width=1024, max_len=None):
        rows = read_index(paths)
        self.n_rejected = 0
        files, widths, ids, offsets = [], [], [], [0]
        for path, latex, w in rows:
            seq = vocab.encode(tokenize(latex))
            if w > max_width or (max_len and len(seq) > max_len):
                self.n_rejected += 1
                continue
            files.append(path)
            widths.append(w)
            ids.extend(seq)
            offsets.append(len(ids))
        # Tableaux numpy et pas listes Python : avec forkserver (Python 3.14), le
        # Dataset est picklé vers chaque worker du DataLoader, et 650 000 listes
        # de tokens se picklent très lentement.
        self.files = np.array(files)
        self.widths = np.array(widths, dtype=np.int32)
        self.ids = np.array(ids, dtype=np.int16)
        self.offsets = np.array(offsets, dtype=np.int64)

    def __len__(self):
        return len(self.widths)

    def __getitem__(self, i):
        img = cv2.imread(str(self.files[i]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(self.files[i])
        x = torch.from_numpy(1.0 - img.astype(np.float32) / 255.0)[None]
        seq = torch.from_numpy(self.ids[self.offsets[i]:self.offsets[i + 1]].astype(np.int64))
        return x, seq


def collate(batch):
    """Liste de (image, ids) → images (B, 1, H, W), widths (B,), tgt_in et tgt_out (B, T).

    Les images sont complétées à droite par des 0 (papier) jusqu'à un multiple de 8,
    les séquences par PAD. tgt_in = ids sans le dernier, tgt_out = ids sans le premier.
    """
    imgs, seqs = zip(*batch)
    widths = torch.tensor([x.shape[-1] for x in imgs])
    w = -(-int(widths.max()) // STRIDE_W) * STRIDE_W
    images = torch.zeros(len(imgs), 1, imgs[0].shape[-2], w)
    for k, x in enumerate(imgs):
        images[k, :, :, :x.shape[-1]] = x
    tgt = torch.full((len(seqs), max(map(len, seqs))), Vocab.PAD, dtype=torch.long)
    for k, s in enumerate(seqs):
        tgt[k, :len(s)] = s
    return images, widths, tgt[:, :-1], tgt[:, 1:]


class WidthBucketSampler(Sampler):
    """Batchs de lignes de largeurs voisines, sous un budget de pixels.

    Un batch tient si (nombre de lignes) × (largeur max du batch) ≤ max_width_sum :
    la mémoire suit la largeur paddée, d'où plus de lignes courtes que de longues.

    shuffle=True : on mélange, on découpe en paquets de `pool` exemples, on trie
    chaque paquet par largeur, on coupe en batchs, puis on mélange les batchs. Trier
    tout le jeu d'un coup donnerait les mêmes batchs à chaque époque.
    Appeler set_epoch(e) à chaque époque pour changer le tirage.
    """

    def __init__(self, widths, max_width_sum, max_batch=256, shuffle=True, pool=16384, seed=0):
        self.widths = np.asarray(widths)
        if self.widths.max() > max_width_sum:
            raise ValueError("une ligne dépasse à elle seule max_width_sum")
        self.budget, self.max_batch = max_width_sum, max_batch
        self.shuffle, self.pool, self.seed, self.epoch = shuffle, pool, seed, 0
        self._cache = None

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _batches(self):
        if self._cache and self._cache[0] == self.epoch:
            return self._cache[1]
        rng = np.random.default_rng((self.seed, self.epoch))
        if self.shuffle:
            order = rng.permutation(len(self.widths))
            chunks = [order[k:k + self.pool] for k in range(0, len(order), self.pool)]
        else:
            chunks = [np.arange(len(self.widths))]
        batches = []
        for chunk in chunks:
            chunk = chunk[np.argsort(self.widths[chunk], kind="stable")]
            start, wmax = 0, 0
            for k, i in enumerate(chunk):
                wmax = max(wmax, int(self.widths[i]))
                n = k - start + 1
                if n > self.max_batch or n * wmax > self.budget:
                    batches.append(chunk[start:k].tolist())
                    start, wmax = k, int(self.widths[i])
            batches.append(chunk[start:].tolist())
        if self.shuffle:
            batches = [batches[k] for k in rng.permutation(len(batches))]
        self._cache = (self.epoch, batches)
        return batches

    def __iter__(self):
        return iter(self._batches())

    def __len__(self):
        return len(self._batches())


def padding_ratio(widths, batches):
    """Part des pixels de padding dans les batchs (0 = aucun gaspillage)."""
    useful = sum(int(widths[b].sum()) for b in batches)
    padded = sum(len(b) * int(widths[b].max()) for b in batches)
    return 1 - useful / padded


def main():
    splits = [MATHWRITING / f"{s}.jsonl" for s in ("train", "synthetic")]
    vocab = build_vocab(splits)
    vocab.save(MATHWRITING / "vocab.json")
    print(f"vocab : {len(vocab)} tokens → {MATHWRITING / 'vocab.json'}")
    for name, paths in [("train+synthetic", splits), ("valid", [MATHWRITING / "valid.jsonl"])]:
        ds = LineDataset(paths, vocab)
        print(f"{name:16s} {len(ds):7d} lignes gardées, {ds.n_rejected} écartées (> 1024 px)")
    s = WidthBucketSampler(ds.widths, 16384, shuffle=False)
    b = s._batches()
    rnd = np.random.default_rng(0).permutation(len(ds.widths))
    naive = [rnd[k:k + 32] for k in range(0, len(rnd), 32)]
    print(f"valid : {len(b)} batchs, padding {padding_ratio(ds.widths, b):.1%} "
          f"(batchs de 32 au hasard : {padding_ratio(ds.widths, naive):.1%})")


if __name__ == "__main__":
    main()
