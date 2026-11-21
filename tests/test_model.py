"""Tests du modèle, une classe par étape de docs/architecture.md.

Une étape pas encore codée (NotImplementedError) est sautée, pas en échec.
  venv/bin/python -m unittest tests.test_model -v           # tout
  venv/bin/python -m unittest tests.test_model.Etape4 -v    # une étape
"""
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from m2t import model as M  # noqa: E402
from m2t.vocab import Vocab  # noqa: E402


def build(fn):
    try:
        return fn()
    except NotImplementedError as e:
        raise unittest.SkipTest(f"{e} pas encore codée") from None


def n_params(m):
    return sum(p.numel() for p in m.parameters())


class Etape1Vocab(unittest.TestCase):
    def setUp(self):
        self.v = build(lambda: Vocab.build([["a", "b", "a"], [r"\frac", "{", "a", "}"]]))

    def test_ordre(self):
        # fréquence décroissante, puis ordre alphabétique (ordre des codes Unicode)
        self.assertEqual(self.v.itos, Vocab.SPECIALS + ["a", r"\frac", "b", "{", "}"])
        self.assertEqual(len(self.v), 9)
        self.assertEqual(self.v.stoi[r"\frac"], 5)

    def test_encode_decode(self):
        self.assertEqual(self.v.encode(["a", "inconnu"]), [Vocab.BOS, 4, Vocab.UNK, Vocab.EOS])
        self.assertEqual(self.v.decode([Vocab.BOS, 4, 5, Vocab.EOS, 6]), ["a", r"\frac"])
        self.assertEqual(self.v.decode([4, Vocab.PAD, 5]), ["a", r"\frac"])

    def test_min_freq_et_sauvegarde(self):
        self.assertEqual(Vocab.build([["a", "b", "a"]], min_freq=2).itos[4:], ["a"])
        with tempfile.TemporaryDirectory() as tmp:
            self.v.save(Path(tmp) / "v.json")
            self.assertEqual(Vocab.load(Path(tmp) / "v.json").itos, self.v.itos)


class Etape2(unittest.TestCase):
    def test_basic_block(self):
        b = build(lambda: M.BasicBlock(64, 128, (2, 1)))
        y = b(torch.randn(2, 64, 32, 40))
        self.assertEqual(y.shape, (2, 128, 16, 40))
        self.assertGreaterEqual(y.min().item(), 0, "ReLU finale manquante ?")
        same = M.BasicBlock(32, 32)
        self.assertEqual(same(torch.randn(1, 32, 8, 8)).shape, (1, 32, 8, 8))
        self.assertLess(n_params(same), n_params(M.BasicBlock(32, 32, (2, 2))),
                        "le raccourci doit être une conv 1x1 quand le stride change, l'identité sinon")

    def test_cnn_encoder(self):
        enc = build(lambda: M.CNNEncoder(d_model=96))
        self.assertEqual(enc(torch.randn(2, 1, 96, 200)).shape, (2, 96, 6, 25))
        self.assertEqual(enc(torch.randn(1, 1, 64, 512)).shape, (1, 96, 4, 64))
        n = n_params(M.CNNEncoder(256))
        self.assertTrue(3e6 < n < 8e6, f"{n / 1e6:.1f} M paramètres : comparer au tableau du §3")


class Etape3(unittest.TestCase):
    def test_pe2d(self):
        pe = build(lambda: M.PositionalEncoding2D(16))
        self.assertEqual(n_params(pe), 0)
        out = pe(torch.zeros(1, 16, 3, 5))
        self.assertEqual(out.shape, (1, 16, 3, 5))
        i = torch.arange(3.0)[:, None].expand(3, 5)
        j = torch.arange(5.0)[None, :].expand(3, 5)
        torch.testing.assert_close(out[0, 0], torch.sin(i))   # 1re moitié : la ligne
        torch.testing.assert_close(out[0, 1], torch.cos(i))
        torch.testing.assert_close(out[0, 2], torch.sin(i / 10000 ** (2 / 8)))
        torch.testing.assert_close(out[0, 8], torch.sin(j))   # 2de moitié : la colonne
        torch.testing.assert_close(out[0, 9], torch.cos(j))
        x = torch.randn(2, 16, 3, 5)
        torch.testing.assert_close(pe(x), x + out)

    def test_pe1d(self):
        pe = build(lambda: M.PositionalEncoding1D(16, max_len=50))
        self.assertEqual(n_params(pe), 0)
        out = pe(torch.zeros(2, 7, 16))
        t = torch.arange(7.0)
        torch.testing.assert_close(out[1, :, 0], torch.sin(t))
        torch.testing.assert_close(out[1, :, 1], torch.cos(t))
        torch.testing.assert_close(out[1, :, 2], torch.sin(t / 10000 ** (2 / 16)))
        x = torch.randn(2, 7, 16)
        torch.testing.assert_close(pe(x), x + out)


class Etape4(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.mha = build(lambda: M.MultiHeadAttention(32, 4)).eval()

    def reference(self):
        ref = nn.MultiheadAttention(32, 4, batch_first=True).eval()
        with torch.no_grad():
            ref.in_proj_weight.copy_(torch.cat([self.mha.q_proj.weight, self.mha.k_proj.weight,
                                                self.mha.v_proj.weight]))
            ref.in_proj_bias.copy_(torch.cat([self.mha.q_proj.bias, self.mha.k_proj.bias,
                                              self.mha.v_proj.bias]))
            ref.out_proj.weight.copy_(self.mha.out_proj.weight)
            ref.out_proj.bias.copy_(self.mha.out_proj.bias)
        return ref

    def test_attention_croisee_avec_padding(self):
        q, kv = torch.randn(2, 5, 32), torch.randn(2, 9, 32)
        pad = torch.zeros(2, 9, dtype=torch.bool)
        pad[1, 6:] = True
        out = self.mha(q, kv, kv, key_padding_mask=pad)
        self.assertEqual(out.shape, (2, 5, 32))
        want, _ = self.reference()(q, kv, kv, key_padding_mask=pad)
        torch.testing.assert_close(out, want, atol=1e-5, rtol=1e-4)

    def test_auto_attention_causale(self):
        x = torch.randn(2, 6, 32)
        causal = torch.triu(torch.ones(6, 6, dtype=torch.bool), diagonal=1)
        want, _ = self.reference()(x, x, x, attn_mask=causal)
        torch.testing.assert_close(self.mha(x, x, x, causal=True), want, atol=1e-5, rtol=1e-4)

    def test_le_futur_ne_fuit_pas(self):
        x = torch.randn(1, 6, 32)
        y = x.clone()
        y[:, 3:] = torch.randn(1, 3, 32)
        torch.testing.assert_close(self.mha(x, x, x, causal=True)[:, :3],
                                   self.mha(y, y, y, causal=True)[:, :3])


class Etape5(unittest.TestCase):
    def test_feedforward(self):
        ff = build(lambda: M.FeedForward(32, 64))
        self.assertEqual(ff(torch.randn(2, 5, 32)).shape, (2, 5, 32))

    def test_encoder_layer(self):
        layer = build(lambda: M.EncoderLayer(32, 4, 64)).eval()
        self.assertEqual(sum(isinstance(m, nn.LayerNorm) for m in layer.modules()), 2)
        x = torch.randn(2, 7, 32)
        pad = torch.zeros(2, 7, dtype=torch.bool)
        pad[0, 5:] = True
        y = x.clone()
        y[0, 5:] = torch.randn(2, 32)  # on change seulement le padding
        a, b = layer(x, pad), layer(y, pad)
        self.assertEqual(a.shape, (2, 7, 32))
        torch.testing.assert_close(a[0, :5], b[0, :5])

    def test_decoder_layer(self):
        layer = build(lambda: M.DecoderLayer(32, 4, 64)).eval()
        self.assertEqual(sum(isinstance(m, nn.LayerNorm) for m in layer.modules()), 3)
        x, mem = torch.randn(1, 6, 32), torch.randn(1, 10, 32)
        pad = torch.zeros(1, 10, dtype=torch.bool)
        pad[0, 8:] = True
        mem2 = mem.clone()
        mem2[0, 8:] = 100.0
        x2 = x.clone()
        x2[0, 4:] = 0.0
        torch.testing.assert_close(layer(x, mem, pad), layer(x, mem2, pad))  # padding mémoire ignoré
        torch.testing.assert_close(layer(x, mem, pad)[:, :4], layer(x2, mem, pad)[:, :4])  # causal


SMALL = dict(vocab_size=30, d_model=64, n_heads=4, n_enc_layers=1, n_dec_layers=2, d_ff=128, dropout=0.0)


class Etape6(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.m = build(lambda: M.Im2Latex(**SMALL)).eval()

    def test_encode(self):
        mem, pad = self.m.encode(torch.randn(2, 1, 32, 80), torch.tensor([80, 40]))
        self.assertEqual(mem.shape, (2, 2 * 10, 64))  # h = 32/16, w = 80/8, aplati ligne par ligne
        self.assertFalse(pad[0].any())
        want = torch.tensor([c >= 5 for r in range(2) for c in range(10)])  # 40 px -> 5 colonnes utiles
        self.assertTrue(torch.equal(pad[1], want))

    def test_forward_et_causalite(self):
        imgs, widths = torch.randn(2, 1, 32, 80), torch.tensor([80, 56])
        tgt = torch.randint(4, 30, (2, 8))
        logits = self.m(imgs, widths, tgt)
        self.assertEqual(logits.shape, (2, 8, 30))
        tgt2 = tgt.clone()
        tgt2[:, 5:] = 4
        torch.testing.assert_close(logits[:, :5], self.m(imgs, widths, tgt2)[:, :5])

    def test_taille(self):
        n = n_params(M.Im2Latex(vocab_size=600))
        self.assertTrue(8e6 < n < 16e6, f"{n / 1e6:.1f} M paramètres : comparer au §8")


class Etape7(unittest.TestCase):
    def test_surapprentissage(self):
        """Le test qui compte : 4 exemples appris par cœur puis relus à l'identique.
        S'il échoue, il y a un bug (décalage BOS/EOS, masque, PE...), pas un manque d'entraînement."""
        torch.manual_seed(0)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        m = build(lambda: M.Im2Latex(**SMALL)).to(dev)
        build(lambda: m.greedy_decode(torch.zeros(1, 1, 32, 64, device=dev), torch.tensor([64], device=dev),
                                      Vocab.BOS, Vocab.EOS, max_len=3))
        imgs = torch.rand(4, 1, 32, 64, device=dev)
        widths = torch.tensor([64, 64, 48, 32], device=dev)
        for k, w in enumerate(widths.tolist()):
            imgs[k, :, :, w:] = 0
        targets = [[5, 6, 7], [8, 9, 10, 11, 12], [5, 5, 13, 14], [20, 21, 22, 23, 24, 25]]
        T = max(map(len, targets)) + 1
        tgt_in = torch.full((4, T), Vocab.PAD, device=dev)
        tgt_out = torch.full((4, T), Vocab.PAD, device=dev)
        for k, t in enumerate(targets):
            tgt_in[k, :len(t) + 1] = torch.tensor([Vocab.BOS] + t)
            tgt_out[k, :len(t) + 1] = torch.tensor(t + [Vocab.EOS])
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        m.train()
        for _ in range(400):
            logits = m(imgs, widths, tgt_in)
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1),
                                               ignore_index=Vocab.PAD)
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.assertLess(loss.item(), 0.05)
        m.eval()
        self.assertEqual(m.greedy_decode(imgs, widths, Vocab.BOS, Vocab.EOS), targets)


if __name__ == "__main__":
    unittest.main()
