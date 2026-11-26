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
        self.stride = stride

        self.basicBlock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=self.stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

        if stride != (1, 1) or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        """(B, in_ch, H, W) -> (B, out_ch, H/sh, W/sw)"""
        out = self.basicBlock(x)
        out += self.shortcut(x)
        out = torch.nn.functional.relu(out)
        return out


class CNNEncoder(nn.Module):
    """Tige + 4 étages de 2 BasicBlock (voir le tableau du §3), puis conv 1x1 vers d_model."""

    def __init__(self, d_model=256):
        super().__init__()
        self.tige = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.etage1 = nn.Sequential(
            BasicBlock(32, 64, stride=(2, 2)),
            BasicBlock(64, 64)
        )
        self.etage2 = nn.Sequential(
            BasicBlock(64, 128, stride=(2, 2)),
            BasicBlock(128, 128)
        )
        self.etage3 = nn.Sequential(
            BasicBlock(128, 256, stride=(2, 2)),
            BasicBlock(256, 256)
        )
        self.etage4 = nn.Sequential(
            BasicBlock(256, 256, stride=(2, 1)),
            BasicBlock(256, 256)
        )
        self.projection = nn.Conv2d(256, d_model, 1, stride=1, bias=True)

    def forward(self, x):
        """(B, 1, H, W) -> (B, d_model, H/16, W/8). H multiple de 16, W multiple de 8."""
        out = self.tige(x)
        out = self.etage1(out)
        out = self.etage2(out)
        out = self.etage3(out)
        out = self.etage4(out)
        out = self.projection(out)
        return out



# --- Étape 3 : encodages positionnels ---------------------------------------

def sincos(positions, dim):
    table = torch.zeros(len(positions), dim)       # tenseur de zéros, forme (lignes, colonnes)
    for ligne, t in enumerate(positions):          # ligne = indice dans la table, t = valeur de la position
        for c in range(dim):
            i = c // 2
            if c % 2 == 0:
                table[ligne, c] = torch.sin(t / 10_000 ** (2*i/dim))
            else:
                table[ligne, c] = torch.cos(t / 10_000 ** (2*i/dim))
    return table

class PositionalEncoding2D(nn.Module):
    """Sinusoïdes : la 1re moitié des canaux code la ligne, la 2de moitié la colonne. Aucun paramètre."""

    def __init__(self, d_model, max_width=256, max_height=12):
        super().__init__()
        table_lign = sincos(torch.arange(max_height), d_model // 2)
        table_column = sincos(torch.arange(max_width), d_model // 2)
        self.register_buffer("pe_lign", table_lign, persistent=False)
        self.register_buffer("pe_column", table_column, persistent=False)

    def forward(self, feat):
        """(B, d, h, w) -> (B, d, h, w), feat + PE"""
        _, _, h, w = feat.shape
        lign = torch.permute(self.pe_lign[:h, :], (1, 0))
        lign = torch.unsqueeze(lign, dim=2)
        lign = lign.expand(-1, -1, w)
        column = torch.permute(self.pe_column[:w], (1, 0))
        column = torch.unsqueeze(column, dim=1)
        column = column.expand(-1, h, -1)
        concat = torch.cat([lign, column], dim=0)
        return feat + concat

class PositionalEncoding1D(nn.Module):
    """Sinusoïdes classiques (Vaswani et al. 2017), précalculées jusqu'à max_len. Aucun paramètre."""

    def __init__(self, d_model, max_len=512):
        super().__init__()
        table = sincos(torch.arange(max_len), d_model)
        self.register_buffer("pe", table, persistent=False)

    def forward(self, x):
        """(B, T, d) -> (B, T, d), x + PE[:T]"""
        _, T, _ = x.shape
        return x + self.pe[:T]


# --- Étape 4 : attention multi-têtes, à la main -----------------------------

class MultiHeadAttention(nn.Module):
    """Attention multi-têtes écrite à la main (pas de nn.MultiheadAttention ni de
    F.scaled_dot_product_attention ici).

    Attributs imposés (le test les recopie dans nn.MultiheadAttention pour comparer) :
    q_proj, k_proj, v_proj, out_proj : nn.Linear(d_model, d_model) avec biais.
    """

    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.h = n_heads
        self.d_k = d_model // n_heads
        assert d_model % n_heads == 0
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask=None, causal=False):
        """query (B, Tq, d), key et value (B, Tk, d)
        key_padding_mask : (B, Tk) booléen, True = clé ignorée ; ou None
        causal : si True, la position i ne voit que les clés j <= i (exige Tq == Tk)
        -> (B, Tq, d)"""
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        B, Tq, _ = q.shape
        q = q.reshape(B, Tq, self.h, self.d_k)
        q = q.transpose(1, 2)

        _, Tk, _ = k.shape
        k = k.reshape(B, Tk, self.h, self.d_k)
        k = k.transpose(1, 2)

        v = v.reshape(B, Tk, self.h, self.d_k)
        v = v.transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / (self.d_k ** 0.5)

        if key_padding_mask is not None:
            key_padding_mask = torch.unsqueeze(key_padding_mask, 1)
            key_padding_mask = torch.unsqueeze(key_padding_mask, 1)
            scores = scores.masked_fill(key_padding_mask, -float("inf"))

        if causal:
            key_padding_mask = torch.triu(torch.ones(Tq, Tk, dtype=torch.bool, device=scores.device), diagonal=1)
            scores = scores.masked_fill(key_padding_mask, -float("inf"))


        scores = scores.softmax(dim=-1)
        poids = self.dropout(scores)
        poids = poids @ v
        attention = poids.transpose(1, 2)
        attention = attention.reshape(B, Tq, self.d_k * self.h)
        out = self.out_proj(attention)
        return out


# --- Étape 5 : couches Transformer (pre-LN) ---------------------------------

class FeedForward(nn.Module):
    """Linear(d, d_ff) -> GELU -> Dropout -> Linear(d_ff, d)"""

    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        x = x + self.feedforward(x)
        return x

class EncoderLayer(nn.Module):
    """x = x + Drop(SelfAttn(LN(x))) ; x = x + Drop(FF(LN(x)))"""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout=dropout)

    def forward(self, x, pad_mask=None):
        """(B, S, d), pad_mask (B, S) -> (B, S, d)"""
        h1 = self.ln1(x)
        x = x + self.dropout(self.self_attn(h1, h1, h1, key_padding_mask=pad_mask))
        h2 = self.ln2(x)
        x = x + self.dropout(self.feed_forward(h2))
        return x


class DecoderLayer(nn.Module):
    """x = x + Drop(SelfAttnCausale(LN(x)))
    x = x + Drop(CrossAttn(LN(x), mémoire))
    x = x + Drop(FF(LN(x)))"""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.self_attn_causal = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.self_attn_cross = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, memory_pad_mask=None):
        """x (B, T, d), memory (B, S, d), memory_pad_mask (B, S) -> (B, T, d)"""
        h1 = self.ln1(x)
        x = x + self.dropout(self.self_attn_causal(h1, h1, h1, causal=True))
        h2 = self.ln2(x)
        x = x + self.dropout(self.self_attn_cross(h2, memory, memory, key_padding_mask=memory_pad_mask))
        h3 = self.ln3(x)
        x = x + self.dropout(self.feed_forward(h3))
        return x


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
