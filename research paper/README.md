# Transfer Learning for Cross-Climate Building Energy Prediction

**IEEE Conference Paper Draft**

## Authors
1. **Shree Santosh Yadav** (USN: `1RUA24CSE0436`, `shreesantoshyadavbtech24@rvu.edu.in`)
2. **Samyak Jain** (USN: `1RUA24CSE0406`, `samyakjainbtech24@rvu.edu.in`)
3. **Meda Vijaya Snehan** (USN: `1RUA24CSE0232`, `mvijayasnehanbtech24@rvu.edu.in`)

*School of Computer Science and Engineering, RV University, Bengaluru, India*

---

## Directory Contents
- `paper.tex` — Full LaTeX source code conforming to the IEEE Conference format.
- `paper.pdf` — Pre-compiled camera-ready PDF with clickable reference hyperlinks.
- `IEEEtran.cls` — Official IEEE LaTeX document class.
- `IEEEtran.bst` — IEEE bibliography style definition.
- `cite.sty` — IEEE citation ordering and grouping package.
- `MERMAID_DIAGRAMS.md` — Reproducible Mermaid diagram definitions for system pipeline and ANN architecture.
- `figures/` — High-resolution figures and source Mermaid scripts:
  - `System_Pipeline.png` & `.mmd` (Horizontal 4-stage transfer learning methodology)
  - `ANN_Architecture.png` & `.mmd` (Vertical neural network architecture and frozen layers)
  - `Primary_Data_Efficiency.png` (Data efficiency curves and crossover point)
  - `Convergence_Plots.png` (Loss convergence across data fractions)
  - `Transferability_Heatmap.png` (Cross-climate transferability performance matrix)

---

## Compilation Instructions
To recompile the document locally using `pdflatex`:

```bash
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex
```

Running twice ensures all citations, equation numbers, and hyperref jump links resolve completely.
