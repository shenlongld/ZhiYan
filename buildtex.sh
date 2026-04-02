#!/usr/bin/env bash
set -e

latexmk -xelatex -interaction=nonstopmode -halt-on-error "dl_framework_beamer.tex"
latexmk -c "dl_framework_beamer.tex"
rm -f "dl_framework_beamer.nav" "dl_framework_beamer.snm" "dl_framework_beamer.vrb"