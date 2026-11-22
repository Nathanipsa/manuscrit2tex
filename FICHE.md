# Projet : reconnaissance de mon écriture manuscrite → fichiers `.tex`

> Fiche de passation, issue d'une conversation avec Claude (20/11/2026), à reprendre sur le PC fixe avec Claude Code.
> **Pour Claude Code** : lis toute cette fiche avant de commencer. La section « À faire » est la feuille de route. Les benchmarks (section 7) sont à exécuter **sur cette machine**. Coche les cases au fur et à mesure et note les résultats dans la section 9 (« Journal des résultats »).

---

## 1. Objectif

- **Entrée** : photos ou scans de feuilles A4 manuscrites (au départ, uniquement mon écriture). Elles contiennent du texte en français et des formules mathématiques.
- **Sortie** : un fichier `.tex` compilable qui reproduit le contenu.
- **Matériel** : RTX 3080 12 Go, Linux.

---

## 2. Décision d'architecture (et pourquoi)

### 2.1 Pourquoi ni ResNet seul ni U-Net seul

On cherche à modéliser $p(y_1,\dots,y_T \mid x)$ : $x$ est une image et $y$ une suite de tokens LaTeX **de longueur variable**.

- Un **ResNet** de classification fait image → vecteur de taille fixe. Il ne peut pas produire une séquence.
- Un **U-Net** fait image → carte de pixels de même taille. C'est de la segmentation, pas de la transcription.

Il faut donc un **encodeur-décodeur autorégressif** :

$$p(y\mid x)=\prod_{t=1}^{T} p(y_t \mid y_{<t}, x)$$

- **Encodeur visuel** : CNN (ResNet) et/ou ViT. Il produit une grille de vecteurs de caractéristiques, avec un encodage positionnel 2D.
- **Décodeur Transformer** : il génère les tokens un par un, avec une attention croisée sur la sortie de l'encodeur.
- **Entraînement** : entropie croisée en *teacher forcing*. **Inférence** : décodage glouton ou *beam search*.

**→ Le ResNet garde sa place comme backbone de l'encodeur. Le U-Net garde la sienne pour l'étape de mise en page (segmentation des lignes).**

### 2.2 Pipeline retenu (on ne vise pas « page entière → .tex » d'un seul coup)

1. **Prétraitement** de la page : redressement (deskew), correction de perspective si photo, normalisation du contraste ou binarisation.
2. **Mise en page** : segmentation des lignes et classification de chaque ligne en `texte` / `formule centrée` / `mixte`. Outil : U-Net (masques de lignes) ou détecteur de boîtes. **Au départ, on peut s'en passer grâce aux feuilles gabarits (section 3.1).**
3. **Reconnaissance ligne par ligne** : image de ligne → chaîne LaTeX (encodeur-décodeur).
4. **Assemblage** : insertion des lignes dans un gabarit `.tex` (préambule, environnements `equation`, paragraphes), puis **test de compilation**.

### 2.3 Modèles candidats pour l'étape 3

| Option | Description | Intérêt | Lien |
|---|---|---|---|
| **A. pix2tex (LaTeX-OCR)** | Encodeur ViT avec backbone ResNet + décodeur Transformer, ~25M paramètres, entraîné sur ~100k paires image-formule surtout synthétiques (imprimées) | Le plus proche de l'idée de départ, code d'entraînement fourni, tient largement en 12 Go | https://github.com/lukas-blecher/LaTeX-OCR |
| **B. TrOCR** | Encodeur Transformer d'image pré-entraîné + décodeur Transformer de texte. Conçu pour des images d'**une seule ligne**. | Décodeur déjà « modèle de langue ». Il faut adapter le tokenizer aux commandes LaTeX. | https://huggingface.co/microsoft/trocr-base-handwritten — article : https://arxiv.org/abs/2109.10282 |
| C. Modèles spécialisés HMER | CoMER (ECCV 2022), TAMER (AAAI), ICAL | Références de l'état de l'art sur CROHME / HME100K | https://github.com/Green-Wood/CoMER — https://github.com/qingzhenduyu/TAMER — https://arxiv.org/abs/2405.09032 |
| D. VLM affiné (v2 éventuelle) | Ex. Uni-MuMER (affinage multi-tâche d'un modèle vision-langage) | Meilleurs résultats publiés récemment. **Non vérifié** que ça tienne dans 12 Go. | https://arxiv.org/abs/2505.23566 |

**Ordre de grandeur de la difficulté** : CoMER atteint 59,33 / 59,81 / 62,97 % d'ExpRate (formule exacte à 100 %) sur CROHME 2014/2016/2019. Avec un seul scripteur, la tâche est plus facile, mais il faut rester réaliste.

**Choix de départ** : option A (pix2tex) en référence principale, option B (TrOCR) en comparaison. On décide sur benchmark (section 7).

---

## 3. Base de données

### 3.1 Mes propres données (affinage sur mon écriture)

**Principe : partir du LaTeX et écrire à la main, pas l'inverse.** La vérité terrain est alors connue d'avance et je n'ai rien à transcrire.

- **Source du LaTeX** : mes `.tex` existants (rapports, notes de cours sur Overleaf), découpés en lignes.
- **Feuilles gabarits** à imprimer :
  - des **marqueurs ArUco** aux 4 coins, pour corriger la perspective automatiquement ;
  - des **zones de ligne numérotées**, chacune avec le LaTeX à recopier imprimé en petit au-dessus (ou une « dictée » sur une feuille séparée, référencée par un ID) ;
  - un **ID de page** imprimé, idéalement en QR code, pour relier le scan au fichier de vérité terrain.
- Après numérisation, le script détecte les ArUco, redresse la page et découpe chaque zone. On obtient des paires `(image_ligne.png, ligne.tex)` **sans annotation manuelle**.
- **Couvrir les trois types de lignes** : texte français seul, formule seule, **lignes mixtes** (texte + `$...$`). Les lignes mixtes sont le cas le plus important à fournir moi-même, car les jeux publics séparent généralement texte et formules.
- **Acquisition** : scanner à 300 dpi **et** photos au téléphone, si je compte utiliser le téléphone en usage réel. Varier les stylos, l'éclairage et les jours.
- **Découpage train / val / test par page ou par séance d'écriture, jamais par ligne**, pour éviter une fuite de données qui donnerait des scores trop optimistes.
- **Quantité** : pas de chiffre vérifié pour un seul scripteur. On mesure avec une **courbe d'apprentissage** : affiner avec 25 → 50 → 100 → … pages et tracer l'erreur en fonction de la quantité de données. On arrête quand la courbe sature.

### 3.2 Normalisation et tokenisation du LaTeX

- **Forme canonique obligatoire** : `\frac{a}{b}` plutôt que `{a\over b}`, espaces, `\left( \right)`, accolades superflues, `\mathrm` / `\text`, etc. Sinon, le modèle est pénalisé pour des différences invisibles au rendu.
- S'inspirer des règles de normalisation de **MathWriting**, qui fournit une forme normalisée de chaque expression. pix2tex utilise aussi KaTeX (Node.js) pour le prétraitement de son jeu de données.
- **Tokens = commandes LaTeX** (`\int`, `\alpha`, `{`, `}`, `^`, `_`, …) plutôt que des caractères.

### 3.3 Jeux publics pour le pré-entraînement

| Jeu | Contenu | Remarques | Lien |
|---|---|---|---|
| **MathWriting** (Google, 2024) | ~230k formules manuscrites + ~400k synthétiques, avec LaTeX normalisé | Données « online » (tracés au stylet) : il faut les **rastériser** avec le code fourni. Licence **CC BY-NC-SA** (non commerciale). Archive de 2,9 Go. | Article : https://arxiv.org/abs/2404.10690 — données : https://storage.googleapis.com/mathwriting_data/mathwriting-2024.tgz — code : https://github.com/google-research/google-research/tree/master/mathwriting |
| **HME100K** | 100k images de formules manuscrites, environ 10 000 scripteurs, surtout prises à l'appareil photo | Proche de photos réelles. Vérifier les conditions d'accès. | https://arxiv.org/abs/2203.01601 — https://ai.100tal.com/dataset |
| **CROHME** 2014/2016/2019 | Benchmark de référence en HMER | Petit, utile surtout pour comparer | (via les dépôts CoMER / TAMER) |
| **RIMES** (lignes, 2011) | ~12k lignes de lettres manuscrites **en français** | Pour la partie texte. Licence MIT sur la version Hugging Face. | https://huggingface.co/datasets/Teklia/RIMES-2011-line — complet : https://zenodo.org/records/10812725 |
| IAM | Lignes manuscrites en anglais, 300 dpi | Optionnel (anglais) | — |

### 3.4 Augmentations

Petites rotations et cisaillement, déformations élastiques, flou, variation de l'épaisseur du trait (érosion/dilatation), texture de papier, bruit JPEG, ombres et gradients d'éclairage (pour les photos).

---

## 4. Stratégie d'entraînement

1. **Pré-entraînement** (ou reprise d'un checkpoint) sur MathWriting rastérisé + HME100K (+ RIMES pour le texte).
2. **Affinage** sur mes lignes (section 3.1), avec un taux d'apprentissage réduit. Éventuellement, geler d'abord l'encodeur.
3. Comparer l'affinage complet et un affinage partiel (LoRA / dégel progressif) si on passe sur TrOCR.

---

## 5. Contraintes matérielles (RTX 3080 12 Go)

- Lignes redimensionnées à une **hauteur fixe** (tester 64, 96 et 128 px), largeur variable.
- **Batchs regroupés par largeur** (bucketing) pour limiter le padding.
- **Précision mixte bf16** : l'architecture Ampere la supporte, à confirmer par benchmark (section 7).
- **Accumulation de gradient** si le batch tenant en mémoire est trop petit.
- pix2tex (~25M paramètres) tient largement. Pour TrOCR-base, la taille de batch maximale est **à mesurer**, non garantie.

---

## 6. Métriques

- **CER / TER** : distance d'édition au niveau des caractères et des tokens LaTeX.
- **ExpRate** : pourcentage de lignes exactement justes (après normalisation).
- **Taux de compilation** du `.tex` produit (`latexmk` / `pdflatex` sans erreur).
- (Bonus) **Comparaison des rendus** : rendre la vérité terrain et la prédiction en image et les comparer, car deux LaTeX différents peuvent donner le même rendu.

---

## 7. Benchmarks à exécuter sur le PC fixe (pour Claude Code)

### 7.1 Environnement
- [x] `nvidia-smi` : RTX 3080, driver 615.78, 12 288 Mio (≈1 Go pris au repos).
- [x] Environnement : `venv` → `cours ipsa/ciri/env` (torch 2.12+cu130). Env pix2tex séparé : à créer.
- [x] `cuda.is_available()` True, `is_bf16_supported()` True, RTX 3080.
- [x] `pdflatex`, `latexmk` et `node` présents.

### 7.2 Débit et mémoire
- [ ] Pour pix2tex et TrOCR-base : **taille de batch maximale** avant OOM, en fp32, fp16 et bf16, pour des hauteurs de ligne de 64, 96 et 128 px.
- [ ] **Débit d'entraînement** (échantillons/s) et **latence d'inférence** (ms/ligne, glouton et beam=5).
- [ ] Noter les résultats dans un tableau (section 9).

### 7.3 Références sans entraînement (zero-shot)
- [ ] Préparer un **petit jeu de test personnel** : ~5 pages écrites à la main, découpées en lignes avec leur vérité terrain.
- [ ] pix2tex pré-entraîné (sans affinage) sur mes lignes de formules : CER, ExpRate.
- [ ] TrOCR-base-handwritten (sans affinage) sur mes lignes de texte français : CER (attendu médiocre, car entraîné sur de l'anglais).
- [ ] Ces chiffres servent de **référence plancher** pour juger l'apport de l'affinage.

### 7.4 Courbe d'apprentissage (après collecte)
- [ ] Affiner le meilleur candidat avec 25 / 50 / 100 pages (même jeu de test figé).
- [ ] Tracer CER et ExpRate en fonction du nombre de pages, et décider s'il faut collecter plus.

---

## 8. À faire (feuille de route)

### Phase 0 : mise en place
- [x] Créer le dépôt du projet (`~/manuscrit2tex`, 20/11) (arborescence proposée : `data/`, `templates/`, `scripts/`, `models/`, `configs/`, `notebooks/`, `results/`).
- [ ] Exécuter les benchmarks d'environnement (section 7.1).

### Phase 1 : outillage des données
- [x] **Script `make_templates.py`** (20/11, lot01 généré : 25 pages, 218 cadres) : prend des `.tex`, extrait et découpe les lignes, puis génère des PDF de feuilles gabarits (ArUco aux coins, ID de page / QR code, zones de ligne numérotées, LaTeX de référence) et un fichier `manifest.json` (ID de page → liste des lignes LaTeX).
- [x] **Script `crop_scans.py`** (20/11, validé sur photos simulées : erreur ≤ 0,2 mm ; reste à essayer sur un vrai scan téléphone) : détection des ArUco (OpenCV), correction de perspective, découpe des zones, lecture de l'ID de page, écriture des paires `(png, tex)`.
- [x] **Module `normalize_latex.py`** (20/11, `scripts/`, 7 tests dont un de compilation pdflatex ; option `display=True` pour la sortie) : forme canonique + tokenizer LaTeX, avec des tests unitaires sur des cas pièges (`\frac` / `\over`, espaces, `\left`/`\right`, `\text{}` français avec accents).
- [ ] Script d'évaluation `evaluate.py` : CER, TER, ExpRate et taux de compilation.

### Phase 2 : collecte
- [ ] Imprimer et remplir un premier lot (~25 pages). **Prêts (20/11)** : `lot01` (rapport de stage : 110 texte, 97 mixtes, 11 formules) et `lot02` (4 rapports de maths dans `data/sources/` : 65 texte, 63 mixtes, 69 formules). Il reste ~320 formules inutilisées dans ces sources pour les lots suivants. Ajouté le 22/11 : `cours_ecg2.tex` (cours de maths ECG2 : 590 texte, 491 mixtes, 139 formules candidates).
- [ ] Mettre de côté ~5 pages comme **jeu de test figé**, qu'on ne touche plus jamais.
- [ ] Varier stylos, scanner et téléphone.

### Phase 3 : pré-entraînement
- [x] Télécharger MathWriting et le rastériser (21/11, `scripts/rasterize_mathwriting.py`, notre propre rendu plutôt que le code officiel : lit le .tgz en flux, cibles = `label` brut passé dans notre normaliseur, 649 196 lignes de 96 px, 0 rejet). Licence CC BY-NC-SA : usage personnel OK. Vocabulaire train + synthetic : 321 tokens ; 7,2 % des images dépassent 1024 px de large.
- [ ] Télécharger RIMES-2011-line (Hugging Face).
- [ ] Voir les conditions d'accès à HME100K.
- [ ] Unifier tous les jeux dans le même format et la même normalisation LaTeX.

### Phase 4 : modèles
- [ ] Coder le modèle maison étape par étape : `docs/architecture.md`, squelette `m2t/`, tests `tests/test_model.py` (Nathan code, Claude relit). Étapes 1 (vocabulaire) et 2 (encodeur CNN) faites le 22/11.
- [ ] Benchmarks zero-shot (section 7.3).
- [ ] Affiner pix2tex sur mes données (avec ou sans pré-entraînement intermédiaire sur MathWriting).
- [ ] Affiner TrOCR avec un tokenizer étendu aux commandes LaTeX.
- [ ] Comparer sur le jeu de test figé, puis choisir.

### Phase 5 : pipeline complet
- [ ] Segmentation des lignes sur une page libre, **sans gabarit** (U-Net ou détecteur). Il faudra annoter quelques pages libres.
- [ ] Classification des lignes (texte / formule centrée / mixte).
- [ ] Assemblage en `.tex` et test de compilation automatique.
- [ ] Script final `page2tex.py image.jpg -o sortie.tex`.

---

## 9. Journal des résultats (à remplir par Claude Code)

| Date | Expérience | Modèle | Données | CER | ExpRate | Compil. % | Notes |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

**Benchmarks matériels**

| Modèle | Hauteur ligne | Précision | Batch max | Échantillons/s (train) | ms/ligne (inférence) |
|---|---|---|---|---|---|
| | | | | | |

---

## 10. Questions ouvertes / à trancher

**Décidé le 20/11 :** modèle écrit par moi (encodeur + décodeur maison), mais partir d'une base générale est permis : jeux publics **et** poids pré-entraînés en point de départ ou en comparaison. pix2tex tourne dans un env séparé `venv-pix2tex/` (Python 3.14 seul sur la machine, ses dépendances sont anciennes). Notre code utilise `venv` → `cours ipsa/ciri/env` (torch 2.12+cu130, OpenCV 5.0).

- ~~Usage final : scanner ou photo au téléphone ?~~ **Tranché le 20/11 : appli scanner du téléphone.** Ces applis recadrent, redressent et filtrent déjà la page : garder un réglage de filtre fixe (couleur ou N&B, toujours le même) pour la collecte et l'usage réel, et laisser les ArUco à l'intérieur des marges pour qu'un recadrage automatique ne les coupe pas.
- Gabarit de sortie `.tex` : quel préambule et quels packages (`amsmath`, `amssymb`, `babel` en français…) ?
- Faut-il gérer les schémas et figures sur les feuilles ? Probablement hors périmètre en v1 : on les détecte et on les remplace par un `% TODO figure`.
- Licence non commerciale de MathWriting : OK pour un usage personnel, à revoir si le projet est diffusé.

---

## Sources

- MathWriting : https://arxiv.org/abs/2404.10690
- HME100K / SAN : https://arxiv.org/abs/2203.01601
- CoMER : https://arxiv.org/abs/2207.04410
- TAMER : https://ojs.aaai.org/index.php/AAAI/article/view/33190
- ICAL : https://arxiv.org/abs/2405.09032
- Uni-MuMER : https://arxiv.org/abs/2505.23566
- pix2tex : https://github.com/lukas-blecher/LaTeX-OCR
- Description de pix2tex (ResNet + Transformer, ~25M paramètres, ~100k paires) : https://arxiv.org/abs/2408.04015
- TrOCR : https://arxiv.org/abs/2109.10282 — https://huggingface.co/docs/transformers/model_doc/trocr
- RIMES : https://huggingface.co/datasets/Teklia/RIMES-2011-line — https://zenodo.org/records/10812725
