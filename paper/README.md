# LaTeX report

`main.tex` is the continuously updated paper-style experiment report. The
figures in `figures/` are generated from the checked-in JSON results with
`plot_results.py`; each point/bar reports a three-seed mean and its whisker is
the sample standard deviation.

Compile in an environment with a standard TeX Live installation:

```bash
pdflatex main.tex
pdflatex main.tex
```

Regenerate the figures before compiling:

```bash
python paper/plot_results.py
```

This server does not currently provide `pdflatex` or `latexmk`, so the source is
syntax-reviewed here but PDF compilation must be performed in a TeX-equipped
environment.
