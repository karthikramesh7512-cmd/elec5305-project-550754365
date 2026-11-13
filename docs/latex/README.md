# ELEC5305 LaTeX report

Open `ELEC5305_Final_Report.tex` as the main file. All eight report figures are included under `figures/`, and the eight references are embedded with `thebibliography` so no BibTeX step is required.

## Compile

With a standard TeX Live or MiKTeX installation, run twice from this folder:

```
pdflatex -interaction=nonstopmode -halt-on-error ELEC5305_Final_Report.tex
pdflatex -interaction=nonstopmode -halt-on-error ELEC5305_Final_Report.tex
```

Or use `latexmk -pdf ELEC5305_Final_Report.tex` or `tectonic ELEC5305_Final_Report.tex`.

For Overleaf, upload the ZIP contents, select this main file, and use pdfLaTeX. The source uses standard packages (including newtx fonts, amsmath, graphicx, tabularx, booktabs, caption, microtype and hyperref). It contains no absolute figure paths or raw ultrasonic data.

The source contains the report text, mathematical equations, numbered tables, figure references and bibliography. LaTeX generates the contents, page numbering and cross-references during compilation.
