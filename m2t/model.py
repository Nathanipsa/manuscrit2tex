"""Encodeur CNN + décodeur Transformer : image de ligne -> tokens LaTeX.

Squelette à remplir étape par étape, en suivant docs/architecture.md.
Les signatures et les formes (shapes) sont fixées : tests/test_model.py les vérifie.
Notations : B batch, H x W image, h x w grille de l'encodeur (h = H/16, w = W/8),
S = h*w longueur de la mémoire, T longueur de la séquence de tokens, d = d_model.
Convention des masques : True = position de padding, à ignorer.
"""
import torch
from torch import nn


# --- Étape 2 : encodeur convolutif ------------------------------------------

class BasicBlock(nn.Module):
    """Bloc résiduel de ResNet : conv3x3-BN-ReLU-conv3x3-BN + raccourci, puis ReLU.

    stride : tuple (sh, sw). Si stride != (1, 1) ou in_ch != out_ch, le raccourci est
    une conv 1x1 de même stride suivie d'une BN ; sinon c'est l'identité.
    """

    def __init__(self, in_ch, out_ch, stride=(1, 1)):
        super().__init__()
        raise NotImplementedError("Étape 2")

    def forward(self, x):
        """(B, in_ch, H, W) -> (B, out_ch, H/sh, W/sw)"""
        raise NotImplementedError("Étape 2")


class CNNEncoder(nn.Module):
    """Tige + 4 étages de 2 BasicBlock (voir le tableau du §3), puis conv 1x1 vers d_model."""

    def __init__(self, d_model=256):
        super().__init__()
        raise NotImplementedError("Étape 2")

    def forward(self, x):
        """(B, 1, H, W) -> (B, d_model, H/16, W/8). H multiple de 16, W multiple de 8."""
        raise NotImplementedError("Étape 2")


# --- Étape 3 : encodages positionnels ---------------------------------------

class PositionalEncoding2D(nn.Module):
    """Sinusoïdes : la 1re moitié des canaux code la ligne, la 2de moitié la colonne. Aucun paramètre."""

    def __init__(self, d_model):
        super().__init__()
        raise NotImplementedError("Étape 3")

    def forward(self, feat):
        """(B, d, h, w) -> (B, d, h, w), feat + PE"""
        raise NotImplementedError("Étape 3")


class PositionalEncoding1D(nn.Module):
    """Sinusoïdes classiques (Vaswani et al. 2017), précalculées jusqu'à max_len. Aucun paramètre."""

    def __init__(self, d_model, max_len=512):
        super().__init__()
        raise NotImplementedError("Étape 3")

    def forward(self, x):
        """(B, T, d) -> (B, T, d), x + PE[:T]"""
        raise NotImplementedError("Étape 3")


# --- Étape 4 : attention multi-têtes, à la main -----------------------------

class MultiHeadAttention(nn.Module):
    """Attention multi-têtes écrite à la main (pas de nn.MultiheadAttention ni de
    F.scaled_dot_product_attention ici).

    Attributs imposés (le test les recopie dans nn.MultiheadAttention pour comparer) :
    q_proj, k_proj, v_proj, out_proj : nn.Linear(d_model, d_model) avec biais.
    """

    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        raise NotImplementedError("Étape 4")

    def forward(self, query, key, value, key_padding_mask=None, causal=False):
        """query (B, Tq, d), key et value (B, Tk, d)
        key_padding_mask : (B, Tk) booléen, True = clé ignorée ; ou None
        causal : si True, la position i ne voit que les clés j <= i (exige Tq == Tk)
        -> (B, Tq, d)"""
        raise NotImplementedError("Étape 4")


# --- Étape 5 : couches Transformer (pre-LN) ---------------------------------

class FeedForward(nn.Module):
    """Linear(d, d_ff) -> GELU -> Dropout -> Linear(d_ff, d)"""

    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        raise NotImplementedError("Étape 5")

    def forward(self, x):
        raise NotImplementedError("Étape 5")


class EncoderLayer(nn.Module):
    """x = x + Drop(SelfAttn(LN(x))) ; x = x + Drop(FF(LN(x)))"""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        super().__init__()
        raise NotImplementedError("Étape 5")

    def forward(self, x, pad_mask=None):
        """(B, S, d), pad_mask (B, S) -> (B, S, d)"""
        raise NotImplementedError("Étape 5")


class DecoderLayer(nn.Module):
    """x = x + Drop(SelfAttnCausale(LN(x)))
    x = x + Drop(CrossAttn(LN(x), mémoire))
    x = x + Drop(FF(LN(x)))"""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        super().__init__()
        raise NotImplementedError("Étape 5")

    def forward(self, x, memory, memory_pad_mask=None):
        """x (B, T, d), memory (B, S, d), memory_pad_mask (B, S) -> (B, T, d)"""
        raise NotImplementedError("Étape 5")


# --- Étapes 6 et 7 : le modèle complet --------------------------------------

class Im2Latex(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_enc_layers=2,
                 n_dec_layers=4, d_ff=1024, dropout=0.1, max_len=512):
        super().__init__()
        raise NotImplementedError("Étape 6")

    def encode(self, images, widths):
        """images (B, 1, H, W) : encre ≈ 1, papier et padding = 0
        widths (B,) : largeur utile de chaque image en px (le reste est du padding à droite)
        -> memory (B, S, d), memory_pad_mask (B, S)"""
        raise NotImplementedError("Étape 6")

    def decode(self, tgt_in, memory, memory_pad_mask):
        """tgt_in (B, T) indices commençant par BOS -> logits (B, T, vocab_size)"""
        raise NotImplementedError("Étape 6")

    def forward(self, images, widths, tgt_in):
        """-> logits (B, T, vocab_size)"""
        raise NotImplementedError("Étape 6")

    @torch.no_grad()
    def greedy_decode(self, images, widths, bos, eos, max_len=200):
        """Décodage glouton -> liste de B listes d'indices, sans BOS ni EOS."""
        raise NotImplementedError("Étape 7")
