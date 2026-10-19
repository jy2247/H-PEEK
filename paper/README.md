# LaTeX report

`main.tex` is the continuously updated paper-style experiment report. It uses
tables only for the current phase; figures will be added after the experiment
matrix and parameter choices are stable.

Compile in an environment with a standard TeX Live installation:

```bash
pdflatex main.tex
pdflatex main.tex
```

This server does not currently provide `pdflatex` or `latexmk`, so the source is
syntax-reviewed here but PDF compilation must be performed in a TeX-equipped
environment.
