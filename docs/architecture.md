# Architecture du modèle : image de ligne → LaTeX

Ce document décrit le modèle que tu vas coder dans `m2t/model.py` et `m2t/vocab.py`.
Les signatures sont déjà en place (squelette avec `NotImplementedError`), et
`tests/test_model.py` vérifie chaque étape :

```
venv/bin/python -m unittest tests.test_model -v            # tout (étapes non codées = sautées)
venv/bin/python -m unittest tests.test_model.Etape4 -v     # une seule étape
```

Règle du jeu : pas de `nn.MultiheadAttention`, `nn.Transformer*` ni
`F.scaled_dot_product_attention` dans ton code. Le test de l'étape 4 se sert de
`nn.MultiheadAttention` uniquement comme référence pour comparer.

---

## 1. Le problème et la famille de modèle

On veut $p(y_1, \dots, y_T \mid x)$, où $x$ est une image et $y$ une suite de
tokens LaTeX de longueur variable. On la factorise de façon **autorégressive** :

$$p(y \mid x) = \prod_{t=1}^{T} p(y_t \mid y_{<t}, x)$$

C'est la famille **encodeur-décodeur** (« image captioning », traduction
automatique, Whisper pour l'audio, pix2tex pour les formules) :

- l'**encodeur** transforme l'image en une grille de vecteurs, la **mémoire** ;
- le **décodeur** produit les tokens un par un. Pour choisir le token $t$, il
  regarde les tokens déjà écrits (auto-attention **causale**) et va chercher
  dans la mémoire les zones de l'image utiles à ce moment-là (attention
  **croisée**). C'est ce mécanisme qui lui permet de « lire » une fraction
  numérateur d'abord, puis dénominateur, sans qu'on lui dise où ils sont.

Pourquoi pas CTC (le classique de l'OCR de texte) : CTC suppose que la sortie se
lit de gauche à droite dans le même ordre que l'image. C'est faux pour
`\frac{a}{b}` (a au-dessus de b) ou `x_{i}^{2}` (i et 2 superposés).
L'attention, elle, peut sauter dans l'image.

## 2. Vue d'ensemble

```
image (B,1,H,W)  encre≈1, papier=0, padding à droite = 0
   │
   ▼  CNNEncoder (ResNet maison, §3)               stride total : /16 en hauteur, /8 en largeur
(B,d,h,w)        h = H/16, w = W/8
   │
   ▼  + PositionalEncoding2D (§4)
   ▼  aplatir ligne par ligne → (B, S=h·w, d)      + masque de padding (B,S)
   ▼  2 × EncoderLayer (§6) : chaque case voit toute la ligne
   ▼  LayerNorm
mémoire (B,S,d) ─────────────────────────────────────────┐
                                                         │ attention croisée
tokens précédents (B,T) : [BOS, y1, ..., y_{T-1}]        │
   ▼  Embedding × √d  + PositionalEncoding1D (§4)        │
   ▼  4 × DecoderLayer (§6) ◄────────────────────────────┘
   ▼  LayerNorm, puis Linear(d → V) (poids partagés avec l'Embedding)
logits (B,T,V)  →  softmax  →  p(y_t | y_<t, x)
```

Valeurs par défaut : `d_model = 256`, `n_heads = 8` (32 dimensions par tête),
`d_ff = 1024`, `dropout = 0.1`, hauteur d'image `H = 96`, largeur ≤ 1024.

## 3. Étape 1 : le vocabulaire (`m2t/vocab.py`)

Les lignes sont déjà découpées en tokens par `scripts/normalize_latex.tokenize`.
Le vocabulaire associe un indice à chaque token.

- 4 tokens spéciaux, toujours aux indices 0 à 3 : `<pad>`, `<bos>` (début),
  `<eos>` (fin), `<unk>` (inconnu).
- `itos` : liste indice → token (spéciaux compris). `stoi` : dict token → indice.
- `build` trie par fréquence décroissante, puis par ordre alphabétique pour les
  ex-aequo (un vocabulaire reproductible d'un run à l'autre, sinon un checkpoint
  ne se recharge plus).
- `encode` ajoute BOS au début et EOS à la fin. `decode` fait l'inverse et
  s'arrête au premier EOS.

Indice : `collections.Counter`, et `sorted(..., key=lambda t: (-freq[t], t))`.

## 4. Étape 2 : l'encodeur convolutif (`BasicBlock`, `CNNEncoder`)

### BasicBlock (le bloc de ResNet-18)

```
x ─► conv3x3(stride) ─► BN ─► ReLU ─► conv3x3 ─► BN ─► (+) ─► ReLU ─► y
│                                                       ▲
└──────────── raccourci : identité, ou conv1x1(stride) + BN si la forme change
```

- Les convolutions suivies d'une BatchNorm se passent de biais (`bias=False`) :
  la BN a déjà son propre décalage.
- `padding=1` sur les conv 3x3 pour garder la taille quand stride = 1.
- `stride` est un **tuple** `(sh, sw)` : on réduit la hauteur et la largeur à
  des rythmes différents.

### CNNEncoder

| Partie | Contenu | Canaux | Taille de sortie (H=96, W=512) |
|---|---|---|---|
| tige | conv3x3 stride 1, BN, ReLU | 1 → 32 | 96 × 512 |
| étage 1 | BasicBlock stride (2,2) + BasicBlock | 32 → 64 | 48 × 256 |
| étage 2 | BasicBlock stride (2,2) + BasicBlock | 64 → 128 | 24 × 128 |
| étage 3 | BasicBlock stride (2,2) + BasicBlock | 128 → 256 | 12 × 64 |
| étage 4 | BasicBlock stride **(2,1)** + BasicBlock | 256 → 256 | 6 × 64 |
| projection | conv1x1 (avec biais) | 256 → d_model | 6 × 64 |

Environ 5,3 M de paramètres pour `d_model = 256` (le test accepte 3 à 8 M).

Pourquoi /16 en hauteur mais seulement /8 en largeur : une ligne est très large
et peu haute, et le décodeur a besoin de nombreuses positions horizontales pour
s'y retrouver (environ une colonne de mémoire par caractère). En hauteur, 6
rangées suffisent pour distinguer exposant, ligne de base, indice et les étages
d'une fraction.

## 5. Étape 3 : les encodages positionnels

L'attention est **invariante par permutation** : sans information de position,
la mémoire serait un sac de vecteurs, et le décodeur un sac de tokens. On ajoute
donc à chaque vecteur une signature de sa position.

### 1D (décodeur), pour la position $t$ et le canal $c$ parmi $d$

$$PE(t, 2i) = \sin\!\left(\frac{t}{10000^{2i/d}}\right) \qquad PE(t, 2i+1) = \cos\!\left(\frac{t}{10000^{2i/d}}\right)$$

Canaux pairs en sinus, impairs en cosinus, entrelacés. On précalcule une table
`(max_len, d)` dans `__init__` et on l'enregistre avec
`self.register_buffer("pe", table, persistent=False)` : ce n'est pas un paramètre
(le test vérifie qu'il n'y en a aucun), mais elle suit le modèle sur le GPU avec `.to(device)`.

### 2D (encodeur)

On coupe les $d$ canaux en deux moitiés de $d/2$ : la première reçoit le
codage 1D (en dimension $d/2$) de l'**indice de ligne**, la seconde celui de
l'**indice de colonne**. Le canal 0 au point $(i, j)$ vaut donc $\sin(i)$, et le
canal $d/2$ vaut $\sin(j)$.

Écris une seule fonction `sincos(positions, dim)` qui renvoie la table
`(len(positions), dim)` et sers-t'en pour les deux classes.

## 6. Étape 4 : l'attention multi-têtes, à la main

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}} + M\right) V$$

avec $d_k = d / n_{heads}$ et $M$ un masque qui vaut $-\infty$ là où l'attention est
interdite, 0 ailleurs.

Déroulé, avec les formes (B batch, h têtes, $d_k$ = d/h) :

1. `q_proj(query)`, `k_proj(key)`, `v_proj(value)` : trois `nn.Linear(d, d)`.
2. Séparer les têtes : `(B, T, d)` → `(B, T, h, d_k)` → `transpose(1, 2)` →
   `(B, h, T, d_k)`.
3. Scores : `Q @ K.transpose(-2, -1) / sqrt(d_k)` → `(B, h, Tq, Tk)`.
4. Masques, par `masked_fill(..., float("-inf"))` :
   - `key_padding_mask (B, Tk)` : l'étendre en `(B, 1, 1, Tk)` pour qu'il
     s'applique à toutes les têtes et à toutes les requêtes ;
   - `causal` : `torch.triu(torch.ones(Tq, Tk, dtype=torch.bool), diagonal=1)`,
     vrai **au-dessus** de la diagonale (le futur). Pense au `device=`.
5. `softmax` sur la **dernière** dimension (les clés), puis dropout.
6. `@ V` → `(B, h, Tq, d_k)`, rassembler les têtes : `transpose(1, 2)` puis
   `reshape(B, Tq, d)`. Après un `transpose`, `view` échoue : utiliser
   `reshape` (ou `.contiguous().view`).
7. `out_proj`.

Pourquoi diviser par $\sqrt{d_k}$ : le produit scalaire de deux vecteurs
aléatoires de dimension $d_k$ a une variance proportionnelle à $d_k$. Sans
correction, les scores sont grands, le softmax sature sur un seul élément et
les gradients s'annulent.

Pourquoi plusieurs têtes : chaque tête apprend sa propre façon de chercher
(une suit la position horizontale, une autre repère les traits de fraction,
etc.) pour le même coût de calcul qu'une seule grande tête.

## 7. Étape 5 : les couches Transformer (variante pre-LN)

```
EncoderLayer :  x = x + Drop(SelfAttn(LN1(x), masque de padding))
                x = x + Drop(FF(LN2(x)))

DecoderLayer :  x = x + Drop(SelfAttn(LN1(x), causal=True))
                x = x + Drop(CrossAttn(requêtes = LN2(x), clés = valeurs = mémoire, masque de padding mémoire))
                x = x + Drop(FF(LN3(x)))

FeedForward :   Linear(d, d_ff) → GELU → Dropout → Linear(d_ff, d)
```

**Pre-LN** (normaliser avant chaque sous-bloc, et non après comme dans
l'article original) : le chemin résiduel `x + ...` reste une identité pure d'un
bout à l'autre du réseau. L'entraînement est beaucoup plus stable, et on n'a
pas besoin d'un warmup très long. La contrepartie : il faut une `LayerNorm`
finale après la dernière couche (c'est le rôle de `enc_norm` et `dec_norm` au §8).

Dans la cross-attention, les **requêtes** viennent du décodeur (« qu'est-ce que
je cherche ? ») et les **clés/valeurs** de la mémoire (« voici ce que contient
l'image »). C'est la seule communication entre les deux moitiés du modèle.

## 8. Étape 6 : le modèle complet (`Im2Latex`)

### `encode(images, widths)`

1. `CNNEncoder`, puis `PositionalEncoding2D` → `(B, d, h, w)`.
2. Aplatir **ligne par ligne** : `flatten(2).transpose(1, 2)` → `(B, h·w, d)`.
   La case $(r, c)$ devient l'élément $r \cdot w + c$.
3. Masque de padding : la colonne $c$ est du padding si $c \geq \lceil
   \text{width} / 8 \rceil$, pour toutes les rangées. Le construire en
   `(B, h, w)` puis le mettre à plat dans le même ordre que les vecteurs.
4. `n_enc_layers` × `EncoderLayer`, puis `LayerNorm`.

### `decode(tgt_in, memory, memory_pad_mask)`

1. `Embedding(vocab_size, d)` multiplié par $\sqrt{d}$ : les embeddings sont
   initialisés petits, et sans ce facteur l'encodage positionnel (d'amplitude 1)
   noierait le sens du token.
2. `PositionalEncoding1D`, puis dropout.
3. `n_dec_layers` × `DecoderLayer`, puis `LayerNorm`.
4. `Linear(d, vocab_size, bias=False)` dont le poids **est** celui de
   l'embedding (`self.out.weight = self.emb.weight`). C'est le *weight tying* :
   deux fois moins de paramètres pour cette partie, et les tokens qui se
   ressemblent finissent proches dans les deux sens.

Taille attendue avec les valeurs par défaut et 600 tokens : environ 11 M de
paramètres (le test accepte 8 à 16 M). C'est petit : pix2tex en a 25 M, et
TrOCR-base 334 M.

### Entraînement par *teacher forcing* (rappel, la boucle viendra après)

Pour la cible `y = [y1, ..., yT]` :

```
tgt_in  = [BOS, y1, y2, ..., yT]      ce que le décodeur reçoit
tgt_out = [y1,  y2, ..., yT, EOS]     ce qu'il doit prédire, position par position
```

Une seule passe `forward` calcule les T prédictions en parallèle. C'est le
masque causal qui empêche la position $t$ de tricher en regardant $y_t$. La perte
est une entropie croisée avec `ignore_index=PAD`, pour ignorer le remplissage
des séquences courtes.

## 9. Étape 7 : le décodage glouton (`greedy_decode`)

À l'inférence, on n'a plus `y` : on part de `[BOS]`, on prend l'`argmax` du
dernier logit, on l'ajoute à la séquence, et on recommence jusqu'à EOS ou
`max_len`. On encode l'image **une seule fois**, hors de la boucle.

Pour un batch, on continue tant qu'au moins une séquence n'a pas fini, puis on
coupe chaque séquence à son premier EOS. Le *beam search* (garder les k
meilleures séquences au lieu d'une) viendra plus tard. Il gagne en général 1 à
2 points.

**Le test de l'étape 7** fait apprendre par cœur 4 images aléatoires à un petit
modèle, puis vérifie qu'il les relit exactement. Un modèle juste y arrive en
quelques secondes. S'il échoue, ce n'est jamais un manque d'entraînement, c'est
un bug : décalage BOS/EOS, masque inversé, oubli de `model.eval()`, etc.

## 10. Pièges classiques

- **Le sens des masques.** Ici, `True` = ignorer, comme dans PyTorch. Un masque
  inversé ne plante pas : le modèle s'entraîne sur n'importe quoi.
- **Le softmax sur la mauvaise dimension.** C'est sur les clés (`dim=-1`), pas
  sur les requêtes.
- **`view` après `transpose`** : erreur « not contiguous ». Utilise `reshape`.
- **Le masque créé sur le CPU** alors que les scores sont sur le GPU :
  `device=scores.device`.
- **Une ligne de scores entièrement à $-\infty$** donne des NaN. Ça n'arrive pas
  ici, parce qu'une image a toujours au moins une colonne utile et que le masque
  causal laisse toujours la diagonale.
- **Le padding de l'image doit valoir 0**, comme le papier : on inverse les
  niveaux de gris (`1 - img / 255`) pour que l'encre vaille 1.
- **`model.eval()` avant de décoder** : sinon le dropout reste actif et la
  BatchNorm utilise les statistiques du batch.

## 11. La suite (après les 7 étapes)

1. **Données** : rastériser MathWriting, recadrer chaque image sur l'encre,
   redimensionner à la hauteur 96, inverser, regrouper par largeur
   (*bucketing*) pour limiter le padding.
2. **Boucle d'entraînement** : AdamW (lr 3e-4, weight decay 0,01), warmup
   linéaire puis décroissance en cosinus, `torch.autocast` en bf16, gradient
   clipping à 1,0, label smoothing 0,1. Suivi du CER sur la validation.
3. **Benchmarks du §7.2 de la fiche** : batch maximal et débit selon la hauteur
   64/96/128.
4. **Affinage sur ton écriture** en janvier, avec les lots 01 et 02.
