%============================================================================
% plos.cls  --  MINIMAL FUNCTIONAL PLACEHOLDER (NOT the official PLOS class)
%============================================================================
% This file exists ONLY so the PLOS-structured submission source
% (main_plos.tex) compiles and can be reviewed locally. It is based on the
% standard `article` class and provides the PLOS template command set
% (\affil, \email, \journal, \subjects). It does NOT reproduce PLOS's
% production typography.
%
% >> BEFORE PRODUCTION SUBMISSION replace this file with the official
%    plos.cls downloaded from:
%    https://plos.org/resource/plos-latex-template/
%    (PLOS accepts a PDF generated from the author's own format at INITIAL
%     submission; the official class is required only at production stage.)
%============================================================================
\ProvidesClass{plos}[2026/10/26 placeholder PLOS class for local compilation]
\LoadClass[10pt,oneside]{article}
\RequirePackage{geometry}
\geometry{margin=1in}
\RequirePackage{etoolbox}

% --- PLOS frontmatter command set -----------------------------------------
\newcommand{\affil}[1]{\def\@affil{#1}}
\newcommand{\email}[1]{\def\@email{#1}}
\newcommand{\journal}[1]{}        % accepted journal (ignored in placeholder)
\newcommand{\subjects}[1]{}       % PLOS subject areas (ignored in placeholder)

\date{}                           % suppress the default date line

% --- Title block ----------------------------------------------------------
\renewcommand{\maketitle}{%
  \begin{center}
    {\Large\bfseries \@title\par}\vskip 1.1em
    {\large \@author\par}
    \ifdef{\@affil}{\vskip 0.35em \@affil\par}{}
    \ifdef{\@email}{\vskip 0.2em \texttt{\@email}\par}{}
  \end{center}\vskip 1.4em}
\endinput
