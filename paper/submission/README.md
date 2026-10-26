# PLOS Computational Biology 投稿包（submission/）

本目录是 **2026 刚果金（DRC）本迪布焦埃博拉疫情** 论文的投稿前准备包，对应已通过终检的
工作稿 `paper/main.tex`（elsarticle 版，未改动）。

## 目录内容

| 文件 | 说明 |
|------|------|
| `main_plos.tex` / `main_plos.pdf` | PLOS 结构的投稿版稿件（由 `main.tex` 经 `_build_plos.py` 转换）。frontmatter 已重构为 PLOS 命令集：`\affil`/`\email`、6 项声明（Author Summary / Ethics / Funding / Competing interests / Author contributions / Data availability）、Keywords；去除了 Elsevier 专属的 `frontmatter`/`keyword`/`highlights` 环境。 |
| `plos.cls` | **占位类文件（非官方）**。本沙箱无法联网下载官方 `plos.cls`，故提供一份基于 `article`、提供 PLOS 命令集、可本地编译的占位 cls。**生产投稿（录用后）前，请用 PLOS 官方 `plos.cls` 替换本文件**：https://plos.org/resource/plos-latex-template/ |
| `supplementary.tex` / `supplementary.pdf` | 补充材料：核实后的周度序列（wk21–wk41，23 个数据点）、数据来源 URL 清单（UKHSA/WHO/ECDC/Africa CDC/DRC MoH 及 4 条文献 URL）、模型与代码清单、可复现说明。 |
| `cover_letter.tex` / `cover_letter.pdf` | 投稿附信（PLOS CB 格式）。 |
| `fig_v6_*.png` | 4 张正文 figure（已从项目根目录复制，随包自包含）。 |
| `plos_compile*.log` 等 | 编译日志。 |

## 投稿前必做两件事

1. **填入真实 Zenodo DOI**：将下列三处 `10.5281/zenodo.XXXXXXX` 替换为你的真实存档 DOI：
   - `main_plos.tex`（第 36 行宏定义 + 第 114、1266 行引用）
   - `supplementary.tex`（S3 节）
   - `cover_letter.tex`
   改一处宏定义即可联动 `main_plos.tex` 内的全部引用。

2. **替换官方 `plos.cls`**（仅在生产投稿阶段需要）：用 PLOS 官方 `plos.cls` 覆盖本目录的占位 `plos.cls`。初投（initial submission）PLOS 接受作者自有格式的 PDF，故 `main.pdf`（工作稿）或 `main_plos.pdf` 均可直接作为初投 PDF 上传；官方 cls 主要用于录用后的生产排版。

## 重新编译

```bash
# 在 submission/ 目录下（xelatex 两次）
xelatex main_plos.tex
xelatex main_plos.tex
xelatex supplementary.tex
xelatex cover_letter.tex
```

## 转换脚本

`paper/_build_plos.py` 负责把工作稿 `main.tex` 转写为本目录的 `main_plos.tex`
（frontmatter 重构、去除 Elsevier 环境、DOI 占位宏）。若 `main.tex` 后续有更新，
可重跑该脚本再生成本包稿件（不会改动 `main.tex`）。

## 状态（2026-10-26）

- [x] PLOS 模板结构（单独投稿版，工作稿 `main.tex` 不动）
- [~] Zenodo DOI 占位已就位，待填入真实 DOI
- [x] 补充材料（周度序列 + 来源 URL + 代码清单）
- [x] 投稿附信
- [x] 6 项 PLOS 声明齐全
- [x] 工作稿 `main.tex` 终检 VERDICT: PASS（presence 331/0、stale 91/0、引用/标签零缺失）
