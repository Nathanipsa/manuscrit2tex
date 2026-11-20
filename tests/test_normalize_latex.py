import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from normalize_latex import LatexError, normalize, tokenize  # noqa: E402

# (entrée, forme canonique attendue)
CASES = [
    # fractions
    (r"${a \over b}$", r"$\frac{a}{b}$"),
    (r"$a+1 \over b$", r"$\frac{a+1}{b}$"),
    (r"$\dfrac12$", r"$\frac{1}{2}$"),
    (r"$\tfrac{x}{2}$", r"$\frac{x}{2}$"),
    (r"${n \choose k}$", r"$\binom{n}{k}$"),
    # indices, exposants, accolades
    (r"$x^2_i$", r"$x_{i}^{2}$"),
    (r"$x_i^2$", r"$x_{i}^{2}$"),
    (r"$e^{{i\pi}}$", r"$e^{i\pi}$"),
    (r"${x}+{{y}}$", r"$x+y$"),
    (r"${}^{14}C$", r"${}^{14}C$"),
    (r"${a+b}^2$", r"${a+b}^{2}$"),
    (r"${x}^2$", r"$x^{2}$"),
    (r"$f^{\prime\prime}(x)$", r"$f''(x)$"),
    (r"$\sqrt2$", r"$\sqrt{2}$"),
    (r"$\sqrt[3]{x}$", r"$\sqrt[3]{x}$"),
    # délimiteurs et espaces invisibles à la main
    (r"$\left( \frac{a}{b} \right)$", r"$(\frac{a}{b})$"),
    (r"$\left. \frac{df}{dx} \right|_{x=0}$", r"$\frac{df}{dx}|_{x=0}$"),
    (r"$\bigl[ x \bigr]$", r"$[x]$"),
    (r"$\int f(x) \, dx$", r"$\int f(x)dx$"),
    (r"$a \qquad b\;c\!d$", r"$a\quad bcd$"),
    (r"$\displaystyle\sum\limits_{i=1}^n i$", r"$\sum_{i=1}^{n}i$"),
    # synonymes et Unicode
    (r"$a \le b \ne c$", r"$a\leq b\neq c$"),
    (r"$a \not= b$", r"$a\neq b$"),
    ("$α≤β$", r"$\alpha\leq\beta$"),
    ("$Δx → 0$", r"$\Delta x\rightarrow0$"),
    (r"$\operatorname{sin} x$", r"$\sin x$"),
    (r"$\mathrm{d}x$", r"$\mathrm{d}x$"),
    (r"${\rm d}x$", r"$\mathrm{d}x$"),
    (r"$\mathbb R$", r"$\mathbb{R}$"),
    (r"$\lbrace x \rbrace$", r"$\{x\}$"),
    # texte dans les maths
    (r"$\text{pour tout } x$", r"$\text{pour tout }x$"),
    (r"$\mbox{d\'ej\`a}$", r"$\text{déjà}$"),
    # texte français
    (r"L'\'el\`eve a re\c{c}u", "L'élève a reçu"),
    ("L’élève", "L'élève"),
    (r"\og bonjour \fg{}", "«bonjour»"),
    ("Donc : vrai", "Donc: vrai"),
    ("Donc   :   vrai !", "Donc: vrai!"),
    (r"C'est \textbf{important}", "C'est important"),
    (r"10~\% de plus\ldots", r"10 \% de plus..."),
    # lignes mixtes et formules centrées
    (r"Soit $f\colon \mathbb R \to \mathbb R$ une fonction.",
     r"Soit $f:\mathbb{R}\rightarrow\mathbb{R}$ une fonction."),
    (r"Alors \( x>0 \) donc", r"Alors $x>0$ donc"),
    (r"$$ \sum_{k=0}^n k $$", r"\[\sum_{k=0}^{n}k\]"),
    (r"\begin{equation} E = mc^2 \label{eq:e} \end{equation}", r"\[E=mc^{2}\]"),
    (r"\begin{align*} a &= b \\ c &= d \end{align*}",
     r"\[\begin{aligned}a&=b\\c&=d\end{aligned}\]"),
    (r"$\begin{pmatrix} a & b \\ c & d \end{pmatrix}$",
     r"$\begin{pmatrix}a&b\\c&d\end{pmatrix}$"),
    (r"$\begin{array}{cc} 1 & 2 \end{array}$", r"$\begin{array}{cc}1&2\end{array}$"),
    (r"Prix : 5 \$", r"Prix: 5 \$"),
    ("  du texte seul  ", "du texte seul"),
    (r"\[\quad\quad a \quad\quad b \qquad\]", r"\[a\quad b\]"),
    ("trois contraintes . Puis ( voir $x$ ) , fin", "trois contraintes. Puis (voir $x$), fin"),
    ("{Problème de synthèse}", "Problème de synthèse"),
    (r"$\text{a {b} c}$", r"$\text{a b c}$"),
]


class TestNormalize(unittest.TestCase):
    def test_cases(self):
        for src, want in CASES:
            with self.subTest(src=src):
                self.assertEqual(normalize(src), want)

    def test_idempotent(self):
        for src, _ in CASES:
            with self.subTest(src=src):
                once = normalize(src)
                self.assertEqual(normalize(once), once)
                # La sortie displaystyle doit revenir à la même cible.
                self.assertEqual(normalize(normalize(src, display=True)), once)

    def test_display(self):
        self.assertEqual(normalize(r"$\frac12$ et $x$", display=True),
                         r"$\displaystyle\frac{1}{2}$ et $\displaystyle x$")
        # Les formules centrées sont déjà en display.
        self.assertEqual(normalize(r"\[\frac12\]", display=True), r"\[\frac{1}{2}\]")

    def test_tokens(self):
        self.assertEqual(tokenize(r"$\frac{a}{b}$"),
                         ["$", r"\frac", "{", "a", "}", "{", "b", "}", "$"])
        self.assertEqual(tokenize(r"si $x$"), ["s", "i", " ", "$", "x", "$"])
        self.assertEqual(tokenize(r"$\begin{pmatrix}a\end{pmatrix}$"),
                         ["$", r"\begin{pmatrix}", "a", r"\end{pmatrix}", "$"])

    def test_same_render_same_tokens(self):
        variants = [r"$\left(x^2_i\right)$", r"$( x_{i}^{2} )$", r"$\bigl({x}_i^{2}\bigr)$"]
        self.assertEqual(len({tuple(tokenize(v)) for v in variants}), 1)

    def test_errors(self):
        for bad in (r"$x", r"${x$", r"$x}$", r"$\frac{a}$", r"\[x"):
            with self.subTest(bad=bad), self.assertRaises(LatexError):
                tokenize(bad)


PREAMBLE = r"""\documentclass{article}
\usepackage[T1]{fontenc}
\usepackage[french]{babel}
\usepackage{amsmath,amssymb}
\frenchsetup{og=«, fg=»}
\begin{document}
"""


@unittest.skipUnless(shutil.which("pdflatex"), "pdflatex absent")
class TestCompile(unittest.TestCase):
    def test_canonical_forms_compile(self):
        body = "\n\n".join(normalize(s, display=d) for s, _ in CASES for d in (False, True))
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "t.tex").write_text(PREAMBLE + body + "\n\\end{document}\n")
            r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "t.tex"],
                               cwd=tmp, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout[-2000:])


if __name__ == "__main__":
    unittest.main()
