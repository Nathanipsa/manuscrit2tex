"""Feuilles gabarits pour collecter mon écriture : sources .tex → PDF à remplir + manifest.

Chaque page A4 porte 4 marqueurs ArUco aux coins (redressement), un QR code
avec l'ID de page, et des cadres numérotés, chacun sous le rendu de la ligne à
recopier. Les positions des cadres sont écrites en mm dans manifest.json :
crop_scans.py n'a rien d'autre à détecter que les marqueurs et le QR.

Le manifest EST la vérité terrain : il se versionne, et un lot imprimé ne se
régénère jamais (les lignes tirées au hasard changeraient). Les lignes déjà
présentes dans un autre lot de templates/ ne sont pas reprises.

Usage :
  python scripts/make_templates.py ~/rapport-de-stage/chapitres/*.tex \\
      --defs ~/rapport-de-stage/main.tex ~/rapport-de-stage/glossaire.tex --batch lot01
  (--list : affiche les lignes candidates sans rien produire)
"""
import argparse
import json
import random
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from normalize_latex import LatexError, add_displaystyle, detokenize, split_segments, tokenize  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# --- Mise en page (mm, origine en haut à gauche) ----------------------------
# Les marqueurs restent à 10 mm du bord : le recadrage auto de l'appli scanner
# ne doit pas les couper.
PAGE_W, PAGE_H = 210, 297
MARKER = 14
MARKERS = {0: (10, 10), 1: (186, 10), 2: (186, 273), 3: (10, 273)}  # coin haut-gauche ; HG, HD, BD, BG
QR = (95, 7, 20)  # x, y, côté
X0, WIDTH = 15, 180
Y0, Y_MAX = 34, 268
GAP = 3
REF_H = {"short": 6, "tall": 10, "xtall": 14}   # rendu de la ligne à recopier
BOX_H = {"short": 16, "tall": 26, "xtall": 36}  # cadre d'écriture ; xtall : matrices, cases

MAX_WIDTH = 55   # caractères visibles par ligne manuscrite
MIN_WIDTH = 12   # en dessous, une ligne de texte gaspille un cadre
MAX_FORMULA = 40

PREAMBLE = r"""\documentclass[a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[french]{babel}
\usepackage{amsmath,amssymb,stmaryrd,graphicx,tikz}
\usetikzlibrary{babel}
\frenchsetup{og=«, fg=»}
\pagestyle{empty}
\begin{document}
"""

TALL = {r"\frac", r"\sum", r"\int", r"\iint", r"\oint", r"\prod", r"\binom",
        r"\lim", r"\overset", r"\underset", r"\overbrace", r"\underbrace"}
INVISIBLE = {"{", "}", "^", "_", "$", r"\[", r"\]", r"\frac", r"\sqrt", r"\text",
             r"\mathrm", r"\mathbf", r"\mathbb", r"\mathcal", r"\hat", r"\bar",
             r"\vec", r"\overline", r"\tilde"}

# --- Nettoyage des sources .tex ---------------------------------------------

DROP_ENVS = ("figure", "figure*", "table", "table*", "tabular", "tabular*",
             "longtable", "tikzpicture", "lstlisting", "verbatim", "minted",
             "thebibliography", "wrapfigure", "algorithm", "tcblisting")
SECTIONS = ("chapter", "section", "subsection", "subsubsection", "paragraph")


def _brace_end(s, i):
    """Indice juste après l'accolade fermante qui correspond à s[i] == '{'."""
    depth = 0
    for k in range(i, len(s)):
        if s[k] == "\\":
            continue
        if s[k] == "{" and (k == 0 or s[k - 1] != "\\"):
            depth += 1
        elif s[k] == "}" and s[k - 1] != "\\":
            depth -= 1
            if depth == 0:
                return k + 1
    return len(s)


def _bracket_end(s, i):
    """Indice juste après le ] qui ferme s[i] == '[', en sautant les {...} et [...] imbriqués."""
    depth = 0
    k = i
    while k < len(s):
        if s[k] == "\\":
            k += 2
            continue
        if s[k] == "{":
            k = _brace_end(s, k)
            continue
        if s[k] == "[":
            depth += 1
        elif s[k] == "]":
            depth -= 1
            if depth == 0:
                return k + 1
        k += 1
    return len(s)


def _sub_cmd(text, names, nargs, repl):
    """Remplace \\nom*[opt]{a1}...{an} par repl(liste des arguments)."""
    pat = re.compile(r"\\(%s)\b\*?" % "|".join(re.escape(n) for n in names))
    out, pos = [], 0
    for m in pat.finditer(text):
        if m.start() < pos:
            continue
        i = m.end()
        while text.startswith("[", i):
            i = _bracket_end(text, i)
        args = []
        for _ in range(nargs):
            while i < len(text) and text[i].isspace():
                i += 1
            if not text.startswith("{", i):
                break
            j = _brace_end(text, i)
            args.append(text[i + 1:j - 1])
            i = j
        if len(args) < nargs:
            continue
        out += [text[pos:m.start()], repl(m.group(1), args)]
        pos = i
    return "".join(out) + text[pos:]


def load_defs(paths):
    """Glossaire, acronymes et macros simples (\\newcommand, \\newtcbox) des préambules."""
    glossary, macros = {}, []
    for p in paths:
        text = Path(p).read_text(encoding="utf-8")
        text = "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in text.splitlines())
        glossary.update(_glossary(text))
        for m in re.finditer(r"\\(?:re)?newcommand\*?\s*\{?\\([A-Za-z]+)\}?\s*(?:\[(\d)\])?\s*\{", text):
            body = text[m.end() - 1:_brace_end(text, m.end() - 1)][1:-1]
            if "@" not in body and "newcommand" not in body:
                macros.append((m.group(1), int(m.group(2) or 0), body))
        for m in re.finditer(r"\\newtcbox\s*\{\\([A-Za-z]+)\}", text):
            macros.append((m.group(1), 1, "#1"))  # boîte colorée : on garde le contenu
    return glossary, macros


def expand_macros(text, macros):
    for _ in range(2):  # macros qui en appellent d'autres
        for name, n, body in macros:
            if n == 0:
                text = re.sub(r"\\%s(?![A-Za-z])" % name, lambda _: body, text)
            else:
                def repl(_, args, body=body):
                    out = body
                    for k, a in enumerate(args):
                        out = out.replace("#%d" % (k + 1), a)
                    return out
                text = _sub_cmd(text, [name], n, repl)
    return text


def _glossary(text):
    """\\newglossaryentry{clé}{name={...}, plural={...}} et \\newacronym{clé}{court}{long}."""
    entries = {}

    def entry(_, args):
        key, body = args
        fields = {}
        for f in ("name", "plural"):
            m = re.search(r"\b%s\s*=\s*" % f, body)
            if m:
                i = m.end()
                if body.startswith("{", i):
                    fields[f] = body[i + 1:_brace_end(body, i) - 1]
                else:
                    fields[f] = re.match(r"[^,}]*", body[i:]).group(0).strip()
        if "name" in fields:
            entries[key] = (fields["name"], fields.get("plural", fields["name"] + "s"))
        return ""

    _sub_cmd(text, ["newglossaryentry"], 2, entry)
    _sub_cmd(text, ["newacronym"], 3, lambda _, a: entries.setdefault(a[0], (a[1], a[1] + "s")) and "")
    return entries


def clean_tex(text, glossary, macros):
    """Source LaTeX → paragraphes de texte et de maths que la main peut recopier."""
    text = re.sub(r"(?<!\\)%[^\n]*\n?", "", text)  # comme TeX : le % mange aussi le retour à la ligne
    m = re.search(r"\\begin\{document\}(.*)\\end\{document\}", text, re.S)
    if m:
        text = m.group(1)
    # Les vraies fins de paragraphe sont figées en \x00 dès maintenant : retirer
    # un \label seul sur sa ligne ne doit pas couper l'équation en deux.
    text = re.sub(r"\n[ \t]*\n\s*", "\x00", text).replace("\n", " ")
    for env in DROP_ENVS:
        text = re.sub(r"\\begin\{%s\}.*?\\end\{%s\}" % (re.escape(env), re.escape(env)),
                      "\n\n", text, flags=re.S)
    text = expand_macros(text, macros)
    # WithArrows : calcul multiligne à flèches commentées → aligned, sans les flèches.
    text = re.sub(r"\\begin\{WithArrows\}(\[[^\]]*\])?", lambda _: r"\begin{aligned}", text)
    text = text.replace(r"\end{WithArrows}", r"\end{aligned}")
    text = re.sub(r"\\begin\{DispWithArrows\*?\}(\[[^\]]*\])?", lambda _: r"\[\begin{aligned}", text)
    text = re.sub(r"\\end\{DispWithArrows\*?\}", lambda _: r"\end{aligned}\]", text)
    text = _sub_cmd(text, ["Arrow"], 1, lambda _, a: "")
    text = _sub_cmd(text, ["tcbhighmath"], 1, lambda _, a: a[0])
    text = _sub_cmd(text, ["hypertarget", "hyperlink"], 2, lambda _, a: a[1])
    text = re.sub(r"\\tcblower\b", "\x00", text)
    text = re.sub(r"\\\[\s*\$([^$]*)\$\s*\\\]", lambda m: r"\[" + m.group(1) + r"\]", text)  # \rond{$...$} dans \[ \]
    # Environnements de mise en forme (listes, boîtes perso) : on garde le contenu.
    keep = {"equation", "equation*", "align", "align*", "gather", "gather*",
            "multline", "multline*", "displaymath", "aligned", "gathered", "cases",
            "array", "matrix", "pmatrix", "bmatrix", "vmatrix", "Vmatrix", "smallmatrix"}
    text = re.sub(r"\\(begin|end)\{([A-Za-z*]+)\}(\[[^\]]*\])?",
                  lambda m: m.group(0) if m.group(2) in keep else "\n\n", text)
    text = re.sub(r"\\item\b(\[[^\]]*\])?", "\n\n", text)

    def gls(name, args):
        sing, plur = glossary.get(args[0], (args[0], args[0] + "s"))
        word = plur if name.lower().endswith("pl") else sing
        return word[:1].upper() + word[1:] if name[0] == "G" else word

    text = _sub_cmd(text, ["gls", "glspl", "Gls", "Glspl"], 1, gls)
    text = _sub_cmd(text, ["glslink"], 2, lambda _, a: a[1])
    text = _sub_cmd(text, SECTIONS, 1, lambda _, a: "\n\n" + a[0] + "\n\n")
    text = _sub_cmd(text, ["textsc", "textsuperscript", "mbox"], 1, lambda _, a: a[0])
    text = _sub_cmd(text, ["ref", "eqref", "autoref", "cref"], 1, lambda _, a: "2")
    text = _sub_cmd(text, ["label", "cite", "citep", "citet", "glsadd", "caption",
                           "footnote", "includegraphics", "input", "include",
                           "vspace", "hspace"], 1, lambda _, a: "")
    text = _sub_cmd(text, ["addcontentsline"], 3, lambda _, a: "")
    text = re.sub(r"\\(centering|noindent|newpage|clearpage|medskip|smallskip|bigskip)\b", "", text)
    text = re.sub(r"\\par\b", "\n\n", text)
    paras = re.split(r"\x00|\n\n", text)
    return [" ".join(p.split()) for p in paras if p.strip()]


# --- Découpage en lignes ----------------------------------------------------

def _width(toks):
    return sum(1 for t in toks if t not in INVISIBLE and t != " ") + toks.count(" ")


def _kind(toks):
    has_math = "$" in toks or r"\[" in toks
    if not has_math:
        return "texte"
    outside, depth = [], 0
    for t in toks:
        if t in ("$", r"\[", r"\]"):
            depth = 1 - depth if t == "$" else (1 if t == r"\[" else 0)
        elif depth == 0 and t != " ":
            outside.append(t)
    return "mixte" if any(t.isalpha() for t in outside) else "formule"


def _make_line(src, origin):
    try:
        toks = tokenize(src)
    except LatexError:
        return None
    if not toks:
        return None
    kind = _kind(toks)
    width = _width(toks)
    if kind == "texte" and width < MIN_WIDTH:
        return None
    rows = toks.count(r"\\") + 1  # matrices, cases
    if width > MAX_FORMULA and kind == "formule" or rows > 3:
        return None
    size = "xtall" if rows > 1 else "tall" if any(t in TALL or t.startswith(r"\begin") for t in toks) else "short"
    return {"latex": detokenize(toks), "kind": kind, "size": size, "source": origin}


def split_rows(body):
    """Calcul sur plusieurs lignes (aligned, WithArrows) → une formule par ligne écrite.

    Seuls les \\\\ de premier niveau coupent : ceux d'une matrice ou d'un cases
    sont dans un environnement imbriqué et restent. Le premier & de chaque ligne
    disparaît (alignement), les suivants deviennent \\quad (deuxième colonne).
    """
    b = body.strip()
    m = re.fullmatch(r"\\begin\{(aligned|gathered|split)\}(.*)\\end\{\1\}", b, re.S)
    if m:
        b = m.group(2)
    rows, cur, depth, amps, i = [], [], 0, 0, 0
    while i < len(b):
        c = b[i]
        if b.startswith(r"\begin{", i) or c == "{":
            depth += 1
        elif b.startswith(r"\end{", i) or c == "}":
            depth -= 1
        if depth == 0 and b.startswith("\\\\", i):
            rows.append("".join(cur))
            cur, amps = [], 0
            i += 2
            m = re.match(r"\s*\[[^\]]*\]", b[i:])  # \\[7pt]
            if m:
                i += m.end()
            continue
        if depth == 0 and c == "&":
            cur.append(r" \quad " if amps else " ")
            amps += 1
            i += 1
            continue
        if c == "\\":
            cur.append(b[i:i + 2])
            i += 2
            continue
        cur.append(c)
        i += 1
    rows.append("".join(cur))
    return [r.strip() for r in rows if r.strip()]


def split_lines(paragraph, origin):
    """Remplit des lignes de MAX_WIDTH caractères visibles, sans couper une formule.

    Une formule collée à sa ponctuation (« $k$. ») forme un seul mot : on ne
    coupe et on ne met d'espace que là où la source en avait un.
    """
    lines, words = [], []  # words : [chaîne, largeur]

    def add(unit, w, space):
        if space or not words:
            words.append([unit, w])
        else:
            words[-1][0] += unit
            words[-1][1] += w

    def fill():
        cur, w = [], 0
        for unit, uw in words:
            if cur and w + uw + 1 > MAX_WIDTH:
                lines.append(_make_line(" ".join(cur), origin))
                cur, w = [], 0
            if uw <= MAX_WIDTH * 1.2:  # sinon formule trop longue pour un cadre
                cur.append(unit)
                w += uw + 1
        if cur:
            lines.append(_make_line(" ".join(cur), origin))
        words.clear()

    try:
        segs = split_segments(paragraph)
    except LatexError:
        return []
    space = True
    for kind, content in segs:
        if kind == "display":
            fill()
            for row in split_rows(re.sub(r"(?<!\\)\$", "", content)):  # $ hérités d'une macro
                lines.append(_make_line(r"\[" + row + r"\]", origin))
            space = True
        elif kind == "inline":
            try:
                w = _width(tokenize("$" + content + "$"))
            except LatexError:
                return [x for x in lines if x]
            add("$" + content + "$", w, space)
            space = False
        else:
            content = re.sub(r"\\\\(\[[^\]]*\])?", " ", content)  # retour à la ligne forcé
            for m in re.finditer(r"\S+", content):
                add(m.group(0), len(m.group(0)), space or m.start() > 0)
                space = False
            space = space or content[-1:].isspace() or not content.strip()
    fill()
    return [x for x in lines if x]


# --- Rendu et contrôle de compilation --------------------------------------

def reference(line):
    """LaTeX imprimé au-dessus du cadre : la forme canonique, en displaystyle."""
    toks = tokenize(line["latex"])
    if toks[0] == r"\[" and toks[-1] == r"\]":
        return r"$\displaystyle " + detokenize(toks[1:-1]) + "$"
    return detokenize(add_displaystyle(toks))


def _compile(tex, workdir, runs=1):
    path = Path(workdir) / "doc.tex"
    path.write_text(tex, encoding="utf-8")
    for _ in range(runs):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "doc.tex"],
                           cwd=workdir, capture_output=True, text=True)
        if r.returncode != 0:
            return False, r.stdout
    return True, ""


def failing(lines, workdir):
    """Lignes qui ne compilent pas (macros perso, etc.).

    Une candidate par ligne de source et pas de -halt-on-error : le log donne le
    numéro de ligne de chaque erreur (« l.123 »), on retire et on recommence.
    La dichotomie ne sert qu'en secours, pour les erreurs sans numéro.
    """
    bad, todo = [], list(lines)
    offset = PREAMBLE.count("\n")
    for _ in range(10):
        if not todo:
            break
        tex = PREAMBLE + "".join(reference(x) + "\\par\n" for x in todo) + "\\end{document}\n"
        (Path(workdir) / "doc.tex").write_text(tex, encoding="utf-8")
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "doc.tex"],
                       cwd=workdir, capture_output=True)
        log = (Path(workdir) / "doc.log").read_text(encoding="latin-1")
        if "! " not in log:
            return bad
        hit = {int(n) - offset - 1 for n in re.findall(r"^l\.(\d+)", log, re.M)}
        hit = {k for k in hit if 0 <= k < len(todo)}
        if not hit:
            break
        bad += [todo[k] for k in sorted(hit)]
        todo = [x for k, x in enumerate(todo) if k not in hit]
    return bad + _bisect(todo, workdir)


def _bisect(lines, workdir):
    if not lines:
        return []
    ok, _ = _compile(PREAMBLE + "\n\n".join(reference(x) for x in lines) + "\n\\end{document}\n", workdir)
    if ok:
        return []
    if len(lines) == 1:
        return lines
    mid = len(lines) // 2
    return _bisect(lines[:mid], workdir) + _bisect(lines[mid:], workdir)


# --- Pages ------------------------------------------------------------------

def layout(lines, batch, n_pages):
    pages, zones, y = [], [], Y0
    for line in lines:
        size = line["size"]
        h = REF_H[size] + BOX_H[size]
        if y + h > Y_MAX:
            pages.append(zones)
            if len(pages) == n_pages:
                break
            zones, y = [], Y0
        zones.append({
            "n": len(zones) + 1, "kind": line["kind"], "latex": line["latex"],
            "source": line["source"], "ref_mm": [X0, y],
            "box_mm": [X0, y + REF_H[size], WIDTH, BOX_H[size]],
        })
        y += h + GAP
    else:
        if zones:
            pages.append(zones)
    return [{"id": f"{batch}-p{k + 1:03d}", "zones": z} for k, z in enumerate(pages)]


def mix(lines, weights, rng):
    """Ordre de tirage : à chaque cadre, un type au hasard selon les poids.

    Sans ça, le texte (de loin le plus abondant) remplirait presque tout.
    """
    pools = {}
    for x in lines:
        pools.setdefault(x["kind"], []).append(x)
    for pool in pools.values():
        rng.shuffle(pool)
    out = []
    while True:
        kinds = [k for k, p in pools.items() if p and weights.get(k, 0) > 0]
        if not kinds:
            return out
        k = rng.choices(kinds, [weights[k] for k in kinds])[0]
        out.append(pools[k].pop())


def page_tex(page, n_pages, k):
    pid = page["id"]
    out = [r"\null", r"\begin{tikzpicture}[remember picture, overlay]",
           r"\begin{scope}[shift={(current page.north west)}, x=1mm, y=-1mm]"]
    for mid, (x, y) in MARKERS.items():
        out.append(r"\node[inner sep=0, anchor=north west] at (%g,%g) "
                   r"{\includegraphics[width=%dmm]{aruco%d.png}};" % (x, y, MARKER, mid))
    qx, qy, qs = QR
    out.append(r"\node[inner sep=0, anchor=north west] at (%g,%g) "
               r"{\includegraphics[width=%dmm]{qr_%s.png}};" % (qx, qy, qs, pid))
    out.append(r"\node[anchor=west, font=\small\ttfamily] at (28,14) {%s};" % pid)
    out.append(r"\node[anchor=west, font=\scriptsize] at (28,20) {page %d/%d};" % (k, n_pages))
    out.append(r"\node[anchor=west, font=\scriptsize, text width=60mm] at (120,17) "
               r"{Recopier chaque ligne dans son cadre, sans en toucher le bord.};")
    for z in page["zones"]:
        x, y = z["ref_mm"]
        bx, by, bw, bh = z["box_mm"]
        out.append(r"\node[anchor=north west, inner sep=0, text width=%dmm, font=\footnotesize] "
                   r"at (%g,%g) {\textbf{%d.}~%s};" % (bw, x, y, z["n"], reference(z)))
        out.append(r"\draw[black!35, line width=0.4pt] (%g,%g) rectangle ++(%g,%g);" % (bx, by, bw, bh))
    out += [r"\end{scope}", r"\end{tikzpicture}", r"\newpage", ""]
    return "\n".join(out)


def write_images(pages, build):
    aruco = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for mid in MARKERS:
        cv2.imwrite(str(build / f"aruco{mid}.png"), cv2.aruco.generateImageMarker(aruco, mid, 400))
    enc = cv2.QRCodeEncoder.create()
    for p in pages:
        qr = cv2.copyMakeBorder(enc.encode("m2t:" + p["id"]), 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=255)
        qr = cv2.resize(qr, None, fx=16, fy=16, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(build / f"qr_{p['id']}.png"), qr)


def used_latex():
    """Lignes déjà distribuées dans un lot existant."""
    seen = set()
    for f in (ROOT / "templates").glob("*/manifest.json"):
        for p in json.loads(f.read_text(encoding="utf-8"))["pages"]:
            seen.update(z["latex"] for z in p["zones"])
    return seen


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sources", nargs="+", type=Path)
    ap.add_argument("--defs", nargs="*", type=Path, default=[],
                    help="fichiers où lire glossaire, acronymes et \\newcommand (main.tex, glossaire.tex)")
    ap.add_argument("--mix", default="texte=1,mixte=1,formule=1",
                    help="poids de tirage par type de ligne, tant qu'il en reste")
    ap.add_argument("--batch", default="lot01")
    ap.add_argument("--pages", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--list", action="store_true", help="afficher les lignes candidates et s'arrêter")
    args = ap.parse_args()

    out_dir = ROOT / "templates" / args.batch
    if (out_dir / "manifest.json").exists() and not args.list:
        sys.exit(f"{out_dir}/manifest.json existe déjà : un lot imprimé ne se régénère pas.")

    glossary, macros = load_defs(args.defs)
    seen, lines = used_latex(), []
    for src in args.sources:
        g, m = load_defs([src])  # macros du préambule de la source elle-même
        text = src.read_text(encoding="utf-8")
        for k, para in enumerate(clean_tex(text, {**glossary, **g}, macros + m)):
            for line in split_lines(para, f"{src.name}:§{k + 1}"):
                if line["latex"] not in seen:
                    seen.add(line["latex"])
                    lines.append(line)

    with tempfile.TemporaryDirectory() as tmp:
        bad = failing(lines, tmp)
    bad_ids = {id(x) for x in bad}
    lines = [x for x in lines if id(x) not in bad_ids]

    counts = {k: sum(x["kind"] == k for x in lines) for k in ("texte", "mixte", "formule")}
    print(f"{len(lines)} lignes utilisables {counts}, {len(bad)} rejetées à la compilation")
    if args.list:
        for x in lines:
            print(f"[{x['kind']:7}] {x['latex']}")
        for x in bad:
            print(f"[REJET  ] {x['latex']}")
        return

    weights = {k: float(v) for k, v in (kv.split("=") for kv in args.mix.split(","))}
    pages = layout(mix(lines, weights, random.Random(args.seed)), args.batch, args.pages)
    if len(pages) < args.pages:
        print(f"Attention : seulement {len(pages)} pages, pas assez de lignes pour {args.pages}.")

    build = out_dir / "build"
    build.mkdir(parents=True, exist_ok=True)
    write_images(pages, build)
    tex = PREAMBLE + "".join(page_tex(p, len(pages), k + 1) for k, p in enumerate(pages)) + "\\end{document}\n"
    ok, log = _compile(tex, build, runs=2)  # 2 passes : TikZ « remember picture »
    if not ok:
        sys.exit("Échec de compilation des gabarits :\n" + log[-3000:])
    shutil.copy(build / "doc.pdf", out_dir / "gabarits.pdf")

    manifest = {
        "batch": args.batch, "created": date.today().isoformat(), "seed": args.seed,
        "sources": [str(s) for s in args.sources], "page_mm": [PAGE_W, PAGE_H],
        "aruco": {"dict": "DICT_4X4_50", "size_mm": MARKER,
                  "top_left_mm": {str(k): v for k, v in MARKERS.items()}},
        "qr_mm": list(QR), "pages": pages,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    n = sum(len(p["zones"]) for p in pages)
    per_kind = {k: sum(z["kind"] == k for p in pages for z in p["zones"]) for k in counts}
    print(f"{len(pages)} pages, {n} cadres {per_kind} → {out_dir}/gabarits.pdf et manifest.json")


if __name__ == "__main__":
    main()
