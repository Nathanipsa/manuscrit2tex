"""Vocabulaire : tokens LaTeX canoniques <-> indices entiers.

Étape 1 de docs/architecture.md. Les tokens viennent de
scripts/normalize_latex.tokenize (commandes entières, sinon caractères).
"""
import json


class Vocab:
    PAD, BOS, EOS, UNK = 0, 1, 2, 3
    SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]

    def __init__(self, tokens):
        """tokens : liste des tokens ordinaires (sans les spéciaux), dans l'ordre des indices 4, 5, ..."""
        self.itos = []
        self.itos.extend(self.SPECIALS)
        for token in tokens:
            self.itos.append(token)

        self.stoi = {token: i for i, token in enumerate(self.itos)}

    @classmethod
    def build(cls, sequences, min_freq=1):
        """sequences : itérable de listes de tokens. Garde les tokens vus au moins min_freq fois,
        triés par fréquence décroissante puis alphabétiquement (ordre reproductible)."""

        token_list = {}

        for lign in sequences:
            for token in lign:
                if token not in token_list:
                    token_list[token] = 1
                else:
                    token_list[token] += 1
        kept_token_list = {}
        for token in token_list:
            if token_list[token] >= min_freq:
                kept_token_list[token] = token_list[token]

        token_list = sorted(kept_token_list.items(), key=lambda item: (1/item[1], item[0]))
        return cls(token for token, freq in token_list)

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens):
        """['\\frac', '{', ...] -> [BOS, ..., EOS] ; token inconnu -> UNK."""
        encoded = []
        encoded.append(self.BOS)
        for token in tokens:
            encoded.append(self.stoi.get(token, self.UNK))

        encoded.append(self.EOS)
        return encoded

    def decode(self, ids):
        """Inverse d'encode : ignore BOS et PAD, s'arrête au premier EOS."""
        decoded = []
        for i in ids:
            if i == self.EOS:
                break
            if i == self.BOS or i == self.PAD:
                continue
            decoded.append(self.itos[i])

        return decoded

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.itos[len(self.SPECIALS):], f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))
