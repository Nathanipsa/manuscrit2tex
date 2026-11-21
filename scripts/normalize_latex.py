"""Forme canonique et tokenisation d'une ligne LaTeX (texte français + maths).

La forme canonique est la vérité terrain : deux sources qui donnent la même
page manuscrite doivent donner les mêmes tokens, sinon le modèle est puni pour
une différence qu'il ne peut pas voir. On retire donc ce que l'écriture à la
main ne montre pas (espaces fins, \\left/\\right, \\displaystyle, \\limits,
\\dfrac/\\tfrac) et on fixe une écriture unique pour le reste (accolades
systématiques autour des arguments, x_{i}^{2}, \\leq, accents en Unicode...).

Le \\displaystyle voulu dans le .tex final se rajoute à la sortie
(normalize(..., display=True)), jamais dans les cibles d'entraînement : à la
main, une fraction n'a pas de « style », elle a juste une taille.

Usage : python normalize_latex.py [--display] [--tokens] "ligne" ...
        (sans ligne en argument : lit l'entrée standard, une ligne par ligne)
"""
import re
import sys
import unicodedata


class LatexError(ValueError):
    pass


class Group(list):
    """Contenu d'une paire d'accolades."""


class _Free(Group):
    """Groupe qui n'est l'argument de personne : candidat à la suppression."""


# --- Tables -----------------------------------------------------------------

_SYNONYMS = {
    r"\le": r"\leq", r"\ge": r"\geq", r"\ne": r"\neq",
    r"\to": r"\rightarrow", r"\gets": r"\leftarrow",
    r"\implies": r"\Longrightarrow", r"\iff": r"\Longleftrightarrow",
    r"\lbrace": r"\{", r"\rbrace": r"\}",
    r"\vert": "|", r"\lvert": "|", r"\rvert": "|",
    r"\Vert": r"\|", r"\lVert": r"\|", r"\rVert": r"\|",
    r"\land": r"\wedge", r"\lor": r"\vee", r"\lnot": r"\neg",
    r"\dots": r"\ldots", r"\dotsc": r"\ldots",
    r"\dotsb": r"\cdots", r"\dotsm": r"\cdots",
    r"\dfrac": r"\frac", r"\tfrac": r"\frac", r"\cfrac": r"\frac",
    r"\dbinom": r"\binom", r"\tbinom": r"\binom",
    r"\textrm": r"\text", r"\mbox": r"\text", r"\textnormal": r"\text",
    r"\textup": r"\text", r"\hbox": r"\text", r"\textbf": r"\text",
    r"\textit": r"\text", r"\emph": r"\text",
    r"\boldsymbol": r"\mathbf", r"\bm": r"\mathbf", r"\Bbb": r"\mathbb",
    r"\qquad": r"\quad",
    r"\lt": "<", r"\gt": ">", r"\ast": "*", r"\colon": ":",
}

# \quad reste : un grand blanc entre deux formules se voit aussi à la main.
_SPACES = {r"\,", r"\;", r"\:", r"\!", "\\ ", "~",
           r"\enspace", r"\thinspace", r"\medspace", r"\thickspace"}
_STYLES = {r"\displaystyle", r"\textstyle", r"\scriptstyle",
           r"\scriptscriptstyle", r"\limits", r"\nolimits",
           r"\nonumber", r"\notag"}
_DROP = _SPACES | _STYLES
_DROP_WITH_ARG = {r"\label", r"\tag", r"\hspace", r"\vspace"}

# Tailles de délimiteurs : invisibles à la main. « \left. » disparaît en entier.
_SIZERS = {r"\left", r"\right", r"\middle"} | {
    s + v for s in (r"\big", r"\Big", r"\bigg", r"\Bigg") for v in ("", "l", "r", "m")}

_ARITY = {r"\frac": 2, r"\binom": 2, r"\overset": 2, r"\underset": 2,
          r"\stackrel": 2, r"\begin{array}": 1, r"\begin{subarray}": 1}
for _c in ("mathrm mathbf mathbb mathcal mathfrak mathscr mathit mathsf mathtt "
           "operatorname hat widehat tilde widetilde bar overline underline vec "
           "overrightarrow dot ddot check breve acute grave boxed overbrace "
           "underbrace cancel").split():
    _ARITY["\\" + _c] = 1

_TEXT_CMDS = {r"\text", r"\textrm", r"\mbox", r"\textnormal", r"\textup",
              r"\hbox", r"\textbf", r"\textit", r"\emph"}
_OLD_FONTS = {r"\rm": r"\mathrm", r"\bf": r"\mathbf", r"\it": r"\mathit",
              r"\cal": r"\mathcal", r"\sf": r"\mathsf", r"\tt": r"\mathtt"}
_OPERATORS = set("sin cos tan cot sec csc arcsin arccos arctan sinh cosh tanh coth "
                 "exp log ln lg lim sup inf max min det dim ker arg deg gcd "
                 "Pr hom".split())

_UNICODE = {
    "≤": r"\leq", "≥": r"\geq", "≠": r"\neq", "×": r"\times", "·": r"\cdot",
    "÷": r"\div", "±": r"\pm", "∓": r"\mp", "∞": r"\infty",
    "→": r"\rightarrow", "←": r"\leftarrow", "⇒": r"\Rightarrow",
    "⇔": r"\Leftrightarrow", "∈": r"\in", "∉": r"\notin", "⊂": r"\subset",
    "⊆": r"\subseteq", "∪": r"\cup", "∩": r"\cap", "∀": r"\forall",
    "∃": r"\exists", "∑": r"\sum", "∏": r"\prod", "∫": r"\int",
    "∂": r"\partial", "∇": r"\nabla", "√": r"\sqrt", "≈": r"\approx",
    "≡": r"\equiv", "−": "-", "′": "'", "…": r"\ldots",
}
for _ch in [chr(c) for c in range(0x3B1, 0x3CA)] + list("ΓΔΘΛΞΠΣΥΦΨΩ"):
    _name = unicodedata.name(_ch).split()[-1].lower().replace("lamda", "lambda")
    if _ch.isupper():
        _name = _name.capitalize()
    _UNICODE[_ch] = "o" if _name == "omicron" else "\\" + _name

_TEXT_STRIP = {r"\textbf", r"\textit", r"\emph", r"\textsf", r"\texttt",
               r"\textsl", r"\textmd", r"\textup", r"\textrm", r"\textnormal"}
_TEXT_SYMBOLS = {r"\oe": "œ", r"\OE": "Œ", r"\ae": "æ", r"\AE": "Æ",
                 r"\og": "«", r"\fg": "»", r"\ldots": "...", r"\dots": "...",
                 "…": "...", "’": "'", "~": " ", "\\ ": " "}
_TEXT_DROP = {r"\,", r"\;", r"\:", r"\!", r"\/", r"\xspace"}

_DISPLAY_ENVS = {"equation", "equation*", "displaymath", "align", "align*",
                 "gather", "gather*", "multline", "multline*"}

_CMD = re.compile(r"\\([A-Za-z]+)")
_LETTER_CMD = re.compile(r"\\[A-Za-z]+$")
_ACCENT = re.compile(r"""\\([`'^"~])\s*(?:\{\s*\\?([A-Za-z])\s*\}|\\?([A-Za-z]))""")
_CEDILLA = re.compile(r"\\c\s*(?:\{\s*([A-Za-z])\s*\}|\s([A-Za-z]))")
_COMBINING = {"'": "\u0301", "`": "\u0300", "^": "\u0302", '"': "\u0308",
              "~": "\u0303"}


# --- Lecture ----------------------------------------------------------------

def _text_prep(s):
    """Accents LaTeX (\\'e, \\c{c}...) → caractères Unicode composés."""
    s = _ACCENT.sub(lambda m: (m.group(2) or m.group(3)) + _COMBINING[m.group(1)], s)
    s = _CEDILLA.sub(lambda m: (m.group(1) or m.group(2)) + "\u0327", s)
    return unicodedata.normalize("NFC", s)


def _match_brace(s, i):
    depth = 0
    k = i
    while k < len(s):
        if s[k] == "\\":
            k += 2
            continue
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return k
        k += 1
    raise LatexError("accolade non fermée")


def _lex(s, text=False):
    """Découpe en tokens. En mode texte, les espaces comptent (un seul token " ")."""
    toks = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c == "%":
            j = s.find("\n", i)
            i = n if j < 0 else j
            continue
        if c.isspace():
            while i < n and s[i].isspace():
                i += 1
            if text and (not toks or toks[-1] != " "):
                toks.append(" ")
            continue
        if c != "\\":
            toks.append(c)
            i += 1
            continue
        m = _CMD.match(s, i)
        if not m:
            if i + 1 >= n:
                raise LatexError("\\ isolé en fin de ligne")
            toks.append(s[i:i + 2])
            i += 2
            continue
        cmd = m.group(0)
        i = m.end()
        while i < n and s[i].isspace():  # TeX avale l'espace après \mot
            i += 1
        if cmd in (r"\begin", r"\end"):
            j = s.find("}", i)
            if i >= n or s[i] != "{" or j < 0:
                raise LatexError(f"{cmd} sans nom d'environnement")
            toks.append(cmd + s[i:j + 1])
            i = j + 1
        elif cmd in _TEXT_CMDS and not text and i < n and s[i] == "{":
            j = _match_brace(s, i)
            toks += [cmd, "{", *_lex(_text_prep(s[i + 1:j]), text=True), "}"]
            i = j + 1
        else:
            toks.append(cmd)
    return toks


def _parse(toks):
    stack = [Group()]
    for t in toks:
        if t == "{":
            stack.append(Group())
        elif t == "}":
            if len(stack) == 1:
                raise LatexError("accolade fermante en trop")
            g = stack.pop()
            stack[-1].append(g)
        else:
            stack[-1].append(t)
    if len(stack) != 1:
        raise LatexError("accolade non fermée")
    return stack[0]


# --- Normalisation des maths ------------------------------------------------

def _infix(seq):
    """{a \\over b} → \\frac{a}{b}, {n \\choose k} → \\binom{n}{k}."""
    for k, t in enumerate(seq):
        if t in (r"\over", r"\choose"):
            cmd = r"\frac" if t == r"\over" else r"\binom"
            return [cmd, Group(seq[:k]), Group(seq[k + 1:])]
    return seq


def _take_arg(seq, i, cmd):
    """Argument de cmd commençant en i, normalisé, comme Group.

    Sans accolades, une commande à arguments emporte ses arguments avec elle :
    2^\\frac{a}{b} vaut 2^{\\frac{a}{b}}, \\sqrt\\text{bar} vaut \\sqrt{\\text{bar}}."""
    if i >= len(seq):
        raise LatexError(f"argument manquant après {cmd}")
    a = seq[i]
    if isinstance(a, Group):
        return i + 1, Group(_norm_math(list(a)))
    j = _arg_end(seq, i)
    return j, Group(_norm_math(list(seq[i:j])))


def _arg_end(seq, i):
    """Indice qui suit l'argument sans accolades commençant en i (récursif : \\mathbf\\mathsf{f})."""
    a = seq[i] if i < len(seq) else None
    if not isinstance(a, str):
        return i + 1
    a = _SYNONYMS.get(a, a)
    j = i + 1
    for _ in range(_ARITY.get(a, 1 if a in (r"\text", r"\sqrt") else 0)):
        j = _arg_end(seq, j)
    return j


def _norm_math(seq):
    seq = _infix(seq)
    out = []
    i = 0
    while i < len(seq):
        t = seq[i]
        i += 1
        if isinstance(t, Group):
            if t and isinstance(t[0], str) and t[0] in _OLD_FONTS:
                # {\rm d} → \mathrm{d}, relu comme tel : {\rm sin} doit finir en \sin
                seq[i:i] = [_OLD_FONTS[t[0]], Group(t[1:])]
            else:
                out.append(_Free(_norm_math(list(t))))
            continue
        t = _UNICODE.get(t, t)
        t = _SYNONYMS.get(t, t)
        if t in _SIZERS:
            if i < len(seq) and seq[i] == ".":
                i += 1
            continue
        if t in _DROP:
            continue
        if t in _DROP_WITH_ARG:
            i, _ = _take_arg(seq, i, t)
            continue
        if t == r"\not" and i < len(seq) and seq[i] in ("=", r"\in"):
            out.append(r"\neq" if seq[i] == "=" else r"\notin")
            i += 1
            continue
        if t == r"\text":
            if i >= len(seq):
                raise LatexError(r"argument manquant après \text")
            a = seq[i]
            i += 1
            out += [t, Group(_norm_text(list(a) if isinstance(a, Group) else [a]))]
            continue
        if t == r"\sqrt":
            out.append(t)
            if i < len(seq) and seq[i] == "[":
                j = next((k for k in range(i, len(seq)) if seq[k] == "]"), None)
                if j is None:
                    raise LatexError(r"\sqrt[ sans ]")
                out += ["[", *_norm_math(list(seq[i + 1:j])), "]"]
                i = j + 1
            i, a = _take_arg(seq, i, t)
            out.append(a)
            continue
        if t in _ARITY:
            args = []
            for _ in range(_ARITY[t]):
                i, a = _take_arg(seq, i, t)
                args.append(a)
            word = "".join(x for x in args[0] if isinstance(x, str))
            if (t in (r"\operatorname", r"\mathrm") and word in _OPERATORS
                    and len(word) == len(args[0])):
                out.append("\\" + word)  # \operatorname{sin} → \sin
            else:
                out += [t, *args]
            continue
        if t in ("^", "_"):
            i, a = _take_arg(seq, i, t)
            if t == "^" and a and all(x == r"\prime" for x in a):
                out += ["'"] * len(a)  # f^{\prime} → f'
            else:
                out += [t, a]
            continue
        out.append(t)
    return _order_scripts(_splice(out))


def _splice(items):
    """Retire les accolades libres inutiles ({x}+y → x+y), garde {}^{14}C."""
    res = []
    for k, it in enumerate(items):
        if type(it) is not _Free:
            res.append(it)
            continue
        nxt = items[k + 1] if k + 1 < len(items) else None
        single = len(it) == 1 and isinstance(it[0], str)
        if isinstance(nxt, str) and nxt in ("^", "_") and not single:
            res.append(Group(it))  # base d'exposant : {a+b}^{2} n'est pas a+b^{2}
        else:
            res.extend(it)
    return res


def _order_scripts(items):
    """x^{2}_{i} → x_{i}^{2} : même rendu, une seule écriture."""
    res = list(items)
    k = 0
    while k + 3 < len(res):
        if (isinstance(res[k], str) and res[k] == "^"
                and isinstance(res[k + 2], str) and res[k + 2] == "_"):
            res[k:k + 4] = ["_", res[k + 3], "^", res[k + 1]]
            k += 4
        else:
            k += 1
    return res


# --- Normalisation du texte -------------------------------------------------

def _norm_text(seq):
    out = []
    i = 0
    while i < len(seq):
        t = seq[i]
        i += 1
        if isinstance(t, Group):
            inner = _norm_text(list(t))
            if out and isinstance(out[-1], str) and out[-1].startswith("\\"):
                out.append(Group(inner))  # argument d'une commande : accolades utiles
            else:
                out += inner  # {mot} : accolades invisibles
            continue
        if t in _TEXT_STRIP:  # gras, italique : invisibles à la main
            if i < len(seq) and isinstance(seq[i], Group):
                out += _norm_text(list(seq[i]))
                i += 1
            continue
        if t in _TEXT_DROP:
            continue
        if t in _TEXT_SYMBOLS:
            out += list(_TEXT_SYMBOLS[t])
            if i < len(seq) and isinstance(seq[i], Group) and not seq[i]:
                i += 1  # \fg{} → »
            continue
        out.append(t)
    return out


def _fix_spaces(toks):
    """Un seul espace ; aucun avant : ; ! ? » (babel français les met) ni avant . , ),
    aucun après « et (. À la main, ces espaces-là ne se voient pas."""
    res = []
    for t in toks:
        if t == " ":
            if res and res[-1] in (" ", "«", "("):
                continue
        elif t in (":", ";", "!", "?", "»", ".", ",", ")") and res and res[-1] == " ":
            res.pop()
        res.append(t)
    return res


# --- Découpage d'une ligne texte / maths -----------------------------------

def _find_dollar(s, i):
    while i < len(s):
        if s[i] == "\\":
            i += 2
            continue
        if s[i] == "$":
            return i
        i += 1
    raise LatexError("$ non fermé")


def split_segments(line):
    """Liste de (genre, contenu), genre ∈ {text, inline, display}."""
    segs, buf = [], []
    i, n = 0, len(line)

    def flush():
        if buf:
            segs.append(("text", "".join(buf)))
            buf.clear()

    def close(end, start):
        j = line.find(end, start)
        if j < 0:
            raise LatexError(f"{end} manquant")
        return j

    while i < n:
        if line.startswith("\\\\", i) or line.startswith("\\$", i):
            buf.append(line[i:i + 2])
            i += 2
            continue
        if line.startswith("$$", i):
            j = _find_dollar(line, i + 2)
            if not line.startswith("$$", j):
                raise LatexError("$$ non fermé")
            flush()
            segs.append(("display", line[i + 2:j]))
            i = j + 2
            continue
        if line[i] == "$":
            j = _find_dollar(line, i + 1)
            flush()
            segs.append(("inline", line[i + 1:j]))
            i = j + 1
            continue
        if line.startswith("\\(", i) or line.startswith("\\[", i):
            inline = line[i + 1] == "("
            j = close("\\)" if inline else "\\]", i + 2)
            flush()
            segs.append(("inline" if inline else "display", line[i + 2:j]))
            i = j + 2
            continue
        m = re.match(r"\\begin\{([A-Za-z]+\*?)\}", line[i:])
        if m and m.group(1) in _DISPLAY_ENVS:
            env = m.group(1)
            end = "\\end{%s}" % env
            j = close(end, i)
            body = line[i + m.end():j]
            base = env.rstrip("*")
            if base in ("align", "gather"):  # plusieurs lignes : environnement interne
                body = "\\begin{%sed}%s\\end{%sed}" % (base, body, base)
            flush()
            segs.append(("display", body))
            i = j + len(end)
            continue
        buf.append(line[i])
        i += 1
    flush()
    return segs


def _flatten(items):
    res = []
    for it in items:
        if isinstance(it, Group):
            res += ["{", *_flatten(it), "}"]
        else:
            res.append(it)
    return res


# --- API --------------------------------------------------------------------

def tokenize(line):
    """Tokens canoniques d'une ligne : commandes LaTeX entières, sinon caractères."""
    toks = []
    for kind, content in split_segments(unicodedata.normalize("NFC", line)):
        if kind == "text":
            toks += _flatten(_norm_text(_parse(_lex(_text_prep(content), text=True))))
            continue
        body = _flatten(_norm_math(_parse(_lex(content))))
        # \quad en bord de formule = indentation de mise en page ; plusieurs = un seul.
        body = [t for k, t in enumerate(body) if not (t == r"\quad" and k and body[k - 1] == r"\quad")]
        while body and body[0] == r"\quad":
            body.pop(0)
        while body and body[-1] == r"\quad":
            body.pop()
        if not body:
            continue
        if kind == "inline":
            toks += ["$", *body, "$"]
        else:
            toks += [r"\[", *body, r"\]"]
    toks = _fix_spaces(toks)
    # Pas d'espace en bord de ligne ni autour d'une formule centrée.
    toks = [t for k, t in enumerate(toks)
            if not (t == " " and (k == 0 or k == len(toks) - 1
                                  or toks[k - 1] == r"\]" or toks[k + 1] == r"\["))]
    return toks


def detokenize(toks):
    """Tokens → chaîne LaTeX valide (sert aussi pour les sorties du modèle)."""
    out = []
    for t in toks:
        if out and _LETTER_CMD.search(out[-1]) and t[:1].isalpha():
            out.append(" ")  # \alpha x, pas \alphax
        out.append(t)
    return "".join(out)


def add_displaystyle(toks):
    """Met \\displaystyle au début de chaque formule en ligne ($...$)."""
    res, opened = [], False
    for t in toks:
        res.append(t)
        if t == "$":
            opened = not opened
            if opened:
                res.append(r"\displaystyle")
    return res


def normalize(line, display=False):
    """Forme canonique d'une ligne. display=True : formules en ligne en \\displaystyle."""
    toks = tokenize(line)
    return detokenize(add_displaystyle(toks) if display else toks)


def main(argv):
    display = "--display" in argv
    as_tokens = "--tokens" in argv
    lines = [a for a in argv if not a.startswith("--")] or sys.stdin.read().splitlines()
    for line in lines:
        try:
            if as_tokens:
                print(" ".join(repr(t) if t == " " else t for t in tokenize(line)))
            else:
                print(normalize(line, display=display))
        except LatexError as e:
            print(f"ERREUR ({e}) : {line}", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])
