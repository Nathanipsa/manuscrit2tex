import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from make_templates import _glossary, clean_tex, expand_macros, split_lines, split_rows  # noqa: E402


class TestExtraction(unittest.TestCase):
    def test_label_ne_coupe_pas_l_equation(self):
        src = "Texte\n\\begin{equation}\nx=1\n\\label{eq:a}\n\\end{equation}\nsuite.\n"
        self.assertEqual(len(clean_tex(src, {}, [])), 1)

    def test_commentaire_ne_coupe_pas_le_paragraphe(self):
        self.assertEqual(clean_tex("début\n% note\nfin\n\nautre", {}, []), ["début fin", "autre"])

    def test_ponctuation_collee_a_la_formule(self):
        lines = split_lines("le vecteur $k$. Puis ($x$), fin", "t")
        self.assertEqual([x["latex"] for x in lines], ["le vecteur $k$. Puis ($x$), fin"])

    def test_formule_centree_isolee(self):
        kinds = [x["kind"] for x in split_lines(r"Soit la relation \[a=\frac{b}{c}\] valable partout ici.", "t")]
        self.assertEqual(kinds, ["texte", "formule", "texte"])

    def test_glossaire_et_macros(self):
        g = _glossary(r"\newglossaryentry{raie}{name={raies spectrales}} \newacronym{irap}{IRAP}{Institut}")
        self.assertEqual(clean_tex(r"Les \glspl{irap} et les \gls{raie}.", g, []), ["Les IRAPs et les raies spectrales."])
        self.assertEqual(expand_macros(r"$\norme{x}$", [("norme", 1, "||#1||")]), "$||x||$")

    def test_calcul_multiligne_une_formule_par_ligne(self):
        body = r"\begin{aligned}f(x)&=a\\[7pt]&=\begin{pmatrix}1\\2\end{pmatrix} & \forall x\end{aligned}"
        self.assertEqual(split_rows(body), [r"f(x) =a", r"=\begin{pmatrix}1\\2\end{pmatrix}  \quad  \forall x"])

    def test_witharrows_et_tcbhighmath(self):
        src = (r"\[\begin{WithArrows}\tcbhighmath[remember as=fx, overlay={\draw[blue] (a) to (b);}]{u}"
               r" &= v \Arrow{on passe \\ à la limite} \\ &= w\end{WithArrows}\]")
        self.assertEqual([x["latex"] for x in split_lines(clean_tex(src, {}, [])[0], "t")],
                         [r"\[u=v\]", r"\[=w\]"])

    def test_rond_dans_formule_et_saut_de_ligne(self):
        src = r"Les vecteurs $v$. \\ Puis \[\rond{$a=b$}\]"
        lines = split_lines(clean_tex(src, {}, [("rond", 1, "#1")])[0], "t")
        self.assertEqual([x["latex"] for x in lines], [r"Les vecteurs $v$. Puis", r"\[a=b\]"])


if __name__ == "__main__":
    unittest.main()
