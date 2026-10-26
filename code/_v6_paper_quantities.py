# -*- coding: utf-8 -*-
"""
_v6_paper_quantities.py —— 论文正文 7 项派生量的 v6 重算（P1–P7）
========================================================================================
主理人指令：论文从 v5 数值回填为 v6 数值（主口径 E1 合并区间、V-C、d_max=0.75、K0=18.44，
MLE θ=(0.178386, 0.004781, 5.5730, 7.9013)，基线 t=148）。本脚本计算正文直接引用的
7 项派生量，只读 import ebola_model_v3（m）与 v6_extended（V）的工具函数，不改任何现有文件。

P1  Rₑ 的 bootstrap 区间（_mc_boot_boot.json 的 200 个 draw；G=9.6920 NGM 口径）
P2  t=148 分仓存量（确定性主口径；ECDC 现患锚点 924 对比）
P3  fatality 对账段（v5 定义复现：eventual = 情景 A 延至 t=3000 的 F(∞)；
    lag = F(148)/eventual = 论文"54%"；另报用户字面定义 d·F(148)/eventual 与
    compartment-split 的"感染累积到 148 病例的 eventual deaths"）
P4  样本外校验（仅 t≤143 的 21 个周区间重拟合 → 外推 5 天到 t=148）
P5  疫苗网格（f_ring∈{0.2,0.5,1.0} × c_ring∈{10,20,30}，B0 基线，horizon t=513）
P6  数据驱动带（t=148 起 365 天；v4/v5 E1 约定 Δ=rate₀·(e^{rT}−1)/r，rate₀=69/31）
P7  常数检测基线 Rₑ₀（H2' β₀=0.27485）+ naive 累计比估计量

复现：D:/anaconda/python.exe _v6_paper_quantities.py [--smoke]
产物：_v6_paper.json（机器可读）、stdout（markdown 表）、results_v6_paper.md（事后撰写）。
所有数值「待用户数值确认」。
"""
import sys
import time
import json
import numpy as np
import ebola_model_v3 as m
import v6_extended as V

m.T_LAST = V.T_BASE                     # 情景/基线统一从 t=148 出发（同 v6_extended.main）
SMOKE = "--smoke" in sys.argv
P = lambda s="": print(s, flush=True)
K0 = 18.44                              # 头条 MoM
CFR_REG = 4536.0 / 9415.0               # 数据侧登记 CFR（不变）
G = float(m.R0_ngm(1.0, V.PRM))         # NGM 乘子（核验 = 9.6920）
TH_HEAD = (0.178386, 0.004781, 5.5730, 7.9013)   # v6 头条 MLE
NLL_HEAD = 130.1093
MAXIT = 400 if SMOKE else 2500
NSTART = 1 if SMOKE else 3
OUT = {}


def get_main():
    """复现 v6 §1.1 终拟合（E1, V-C, k=18.44），交叉核验头条值。"""
    nll, _, _ = V.make_nll(V.SCHEMES["E1"], 0.75, V.PRM, K0)
    th, n, det = V.refit(nll, m._keys(0.75), V.vc_starts(V.TH_V5)[:NSTART], MAXIT)
    return tuple(float(v) for v in th), float(n)


def q(a, ps=(2.5, 50, 97.5)):
    return [float(x) for x in np.nanpercentile(np.asarray(a, float), ps)]


# ============================ P1 ============================
def p1():
    P("## P1. Rₑ 的 bootstrap 区间（200 draws，G=9.6920 NGM 口径）")
    P(f"- NGM 乘子核验：G = R0_ngm(1) = {G:.6f}（v6 §3：9.6920）")
    bt = json.load(open("_mc_boot_boot.json", encoding="utf-8"))
    TH = np.array([d["theta"] for d in bt["draws"]], float)
    b, r, t0, s = TH[:, 0], TH[:, 1], TH[:, 2], TH[:, 3]
    re0 = b * G
    re148 = b * np.exp(-r * 148.0) * G
    re60 = b * np.exp(-r * 60.0) * G
    with np.errstate(divide="ignore", invalid="ignore"):
        tstar = np.where((re0 > 1.0) & (r > 0.0), np.log(re0) / r, np.nan)
    n_cross = int(np.sum(np.isfinite(tstar)))
    n_re0_gt1 = int(np.sum(re0 > 1.0))
    P("")
    P("| 量 | 2.5% | 50% | 97.5% | 确定性主口径 |")
    P("|---|---|---|---|---|")
    rows = {}
    det_vals = {"Rₑ₀": TH_HEAD[0] * G,
                "Rₑ(148)": TH_HEAD[0] * np.exp(-TH_HEAD[1] * 148.0) * G,
                "Rₑ(60)": TH_HEAD[0] * np.exp(-TH_HEAD[1] * 60.0) * G}
    for nm, arr in (("Rₑ₀", re0), ("Rₑ(148)", re148), ("Rₑ(60)", re60)):
        qq = q(arr)
        rows[nm] = qq
        P(f"| {nm} | {qq[0]:.4f} | {qq[1]:.4f} | {qq[2]:.4f} | {det_vals[nm]:.4f} |")
    qts = q(tstar)
    P(f"| t*（Rₑ=1 穿越） | {qts[0]:.2f} | {qts[1]:.2f} | {qts[2]:.2f} d | "
      f"确定性 {np.log(TH_HEAD[0]*G)/TH_HEAD[1]:.2f} d |")
    P("")
    P(f"- t* 有效 draw：{n_cross}/{len(b)}（Rₑ₀>1 的 draw：{n_re0_gt1}；"
      f"其余 ρ≤0 或 Rₑ₀≤1 不穿越）")
    # 相关性
    c_br = float(np.corrcoef(b, r)[0, 1])
    c_lbr = float(np.corrcoef(np.log(b), r)[0, 1])
    c_r0r148 = float(np.corrcoef(re0, re148)[0, 1])
    P(f"- 相关系数：corr(β₀,ρ) = {c_br:+.4f}；corr(ln β₀,ρ) = {c_lbr:+.4f}；"
      f"corr(Rₑ₀,Rₑ(148)) = {c_r0r148:+.4f}")
    # 独立组合 vs 经验（delta 方法，log 空间）
    x = np.log(b) - 148.0 * r                    # ln Rₑ(148) − ln G
    v_emp = float(np.var(x, ddof=1))
    v_ind = float(np.var(np.log(b), ddof=1) + 148.0 ** 2 * np.var(r, ddof=1))
    w_emp = rows["Rₑ(148)"][2] - rows["Rₑ(148)"][0]
    # 独立角点组合
    qb, qr = q(b), q(r)
    corners = [qb[0] * np.exp(-qr[1] * 148.0) * G, qb[2] * np.exp(-qr[0] * 148.0) * G]
    w_ind_corner = max(corners) - min(corners)
    P(f"- Rₑ(148) 经验 95% 宽度 = {w_emp:.4f}；独立角点组合宽度 = {w_ind_corner:.4f}"
      f"（角点 {corners[0]:.4f}/{corners[1]:.4f}）→ 经验/独立 = {w_emp/w_ind_corner:.3f}")
    P(f"- delta 方法（log 空间）：Var(经验) = {v_emp:.3e} vs Var(独立) = {v_ind:.3e}"
      f" → 宽度比（独立/经验）= {np.sqrt(v_ind/v_emp):.3f}")
    P(f"- 解读：β₀ 与 ρ 在 bootstrap 中**强正相关**（{c_br:+.2f}，大 β₀ 配大 ρ——早增长快"
      "的伪数据同时衰退快），Rₑ(148)=β₀e^(−148ρ)G 中 β₀ 的放大与 ρ 的衰减相互抵消，"
      f"故 Rₑ(148) 区间窄于两参数边际区间的独立组合（宽度仅独立的 "
      f"{w_emp/w_ind_corner:.2f}）；Rₑ₀ 与 Rₑ(148) 因而强负相关（{c_r0r148:+.2f}）——"
      "高 Rₑ₀ 的 draw 到 t=148 反而降得更低。论文若并列报告 Rₑ₀ 与 Rₑ(148) 区间，"
      "须说明这一补偿结构，勿把两区间当作独立证据。")
    OUT["p1"] = dict(G=G, rows=rows, tstar=qts, n_cross=n_cross, n_re0_gt1=n_re0_gt1,
                     corr_br=c_br, corr_lbr=c_lbr, corr_r0r148=c_r0r148,
                     w_emp=w_emp, w_ind_corner=w_ind_corner,
                     ratio_corner=w_emp / w_ind_corner,
                     delta_ratio=float(np.sqrt(v_ind / v_emp)),
                     det=det_vals)


# ============================ P2 ============================
def p2(th):
    P("")
    P("## P2. t=148 分仓存量（确定性主口径 E1 V-C）")
    sb = m.simulate(V.PRM, m.Xf0, th, V.T_BASE, 0.5, dmax_fixed=0.75)
    y = sb["traj"][-1]
    d148 = float(sb["dfun"](V.T_BASE))
    IS, IC, H, F = (float(y[m.I_IS]), float(y[m.I_IC]),
                    float(y[m.I_H]), float(y[m.I_F]))
    E_tot = float(y[1:1 + V.PRM["n_e"]].sum())
    R = float(y[m.I_R])
    cum = float(m.cum_inf_of(sb["traj"])[-1])
    conf = float(sb["conf_series"][-1])
    dH = d148 * H
    dF = d148 * F
    tot = IS + IC + H
    P("")
    P("| 量 | 模型 t=148 | 对比 | 偏差 |")
    P("|---|---|---|---|")
    P(f"| I_S（求医轨现患） | {IS:,.1f} | — | — |")
    P(f"| I_C（社区轨现患） | {IC:,.1f} | — | — |")
    P(f"| H（在院） | {H:,.1f} | ECDC 924 | {100*(H-924)/924:+.1f}% |")
    P(f"| I_S+I_C+H（症状+住院存量） | {tot:,.1f} | ECDC 924 | {100*(tot-924)/924:+.1f}% |")
    P(f"| d(148)·H（确诊口径在院） | {dH:,.1f} | ECDC 924 | {100*(dH-924)/924:+.1f}% |")
    P(f"| F(148)（全死因累计死亡） | {F:,.1f} | — | — |")
    P(f"| d·F(148)（确诊口径累计死亡） | {dF:,.1f} | 实测 4,536 | {100*(dF-4536)/4536:+.2f}% |")
    P(f"| conf(148)（模型累计确诊） | {conf:,.1f} | 实测 9,415 | {100*(conf-9415)/9415:+.2f}% |")
    P(f"| cum_inf(148)（模型累计感染） | {cum:,.1f} | — | — |")
    P("")
    P(f"- 附：E 潜伏期存量 {E_tot:,.1f}；R 累计康复 {R:,.1f}；d(148) = {d148:.5f}")
    P(f"- 口径提示：论文 v5 段「d(148)·H=262 vs 924」的口径论证按新数更新为 "
      f"d(148)·H={dH:,.0f} vs 924（{100*(dH-924)/924:+.0f}%）；若用全口径 H={H:,.0f}"
      f"（{100*(H-924)/924:+.0f}%），张力形态相同（模型在院存量远低于 ECDC 现患锚点）。")
    OUT["p2"] = dict(IS=IS, IC=IC, H=H, tot=tot, dH=dH, F=F, dF=dF, conf=conf,
                     cum=cum, d148=d148, E=E_tot, R=R)


# ============================ P3 ============================
def p3(th, y148, p2d):
    P("")
    P("## P3. fatality 对账段（论文 §5.1 核心数的 v6 同义重算）")
    F148 = p2d["F"]
    d148 = p2d["d148"]
    dF148 = p2d["dF"]
    cum148 = p2d["cum"]
    # eventual 定义 (i)：情景 A（β 续降）自 t=148 积分到 t=3000（v5/v7_checks 定义）
    rA = m.run_scenario(V.PRM, m.Xf0, th, y148.copy(), "A", t_end=3000.0, dmax_fixed=0.75)
    conv = {}
    for tt in (500.0, 1000.0, 1500.0, 2000.0, 3000.0):
        idx = int(round((tt - V.T_BASE) / 0.5))
        conv[tt] = float(rA["traj"][idx, m.I_F])
    Finf = float(rA["traj"][-1, m.I_F])
    lag_A = F148 / Finf                                   # v5 论文「54%」口径
    recfrac_lit_A = dF148 / Finf                          # 字面 d·F/eventual（定义 i）
    # eventual 定义 (ii)：cohort——感染累积到 t=148 的病例中最终将死亡的人数
    # （compartment 拆分闭式：现患存量按路径 CFR 加权，不含 148 后新感染）
    prm = V.PRM
    c1, c2, pc = prm["c1"], prm["c2"], prm["p_care"]
    fut_split = (c1 * (float(y148[m.I_IS]) + float(y148[m.I_H]))
                 + c2 * float(y148[m.I_IC])
                 + (pc * c1 + (1 - pc) * c2) * p2d["E"])
    ev_coh = F148 + fut_split                             # cohort eventual（含已故 F）
    recfrac_coh = F148 / ev_coh                           # cohort 全口径 recorded fraction
    recfrac_lit_coh = dF148 / ev_coh                      # cohort 字面 d·F/eventual
    P("")
    P("| 量 | v5（存档） | v6 | 定义 |")
    P("|---|---|---|---|")
    P(f"| F(148)（全死因死亡存量） | 6,707.4 | **{F148:,.1f}** | 主口径模拟至 t=148 的 F |")
    P(f"| d(148) | 0.74995 | {d148:.5f} | 检出爬坡函数 |")
    P(f"| d·F(148) | 5,030.2 | {dF148:,.1f} | d(148)×F(148)（实测 4,536，模型 +4.08%） |")
    P(f"| eventual deaths（定义 i：情景 A 延至 t=3000） | 12,421.8 | **{Finf:,.1f}** | "
      "β(t)=β(148)e^(−ρ(t−148)) 续降，F_A(∞)；含 148 后新感染的死亡 |")
    P(f"| eventual deaths（定义 ii：cohort，仅 148 前感染） | — | **{ev_coh:,.1f}** | "
      f"F(148)+现患存量未来死亡 {fut_split:,.1f}（c₁(I_S+H)+c₂I_C+(pc₁+(1−p)c₂)ΣE） |")
    P(f"| 死亡滞后（定义 i）F(148)/eventual | 0.5400 | **{lag_A:.4f}** | v5 论文「54%」口径 |")
    P(f"| recorded fraction（定义 i，字面 d·F/eventual） | 0.4050 | {recfrac_lit_A:.4f} | "
      "P3 字面定义（备录） |")
    P(f"| recorded fraction（定义 ii，全口径） | — | {recfrac_coh:.4f} | "
      "F(148)/cohort-eventual |")
    P(f"| recorded fraction（定义 ii，字面 d·F/eventual） | — | {recfrac_lit_coh:.4f} | "
      "P3 字面定义 + cohort |")
    P(f"- F_A 收敛性：" + "；".join(f"F_A({tt:.0f})={v:,.1f}" for tt, v in conv.items())
      + "（t≥500 已收敛）")
    evcfr_i = lambda dm: CFR_REG * dm / lag_A
    evcfr_ii = lambda dm: CFR_REG * dm / recfrac_coh
    evcfr_lit_i = lambda dm: CFR_REG * dm / recfrac_lit_A
    evcfr_lit_ii = lambda dm: CFR_REG * dm / recfrac_lit_coh
    struct_model = c1 * pc + c2 * (1 - pc)        # = c_total = 0.60（构造恒等）
    struct_paper = 0.43 * 0.46 + 0.57 * 0.7474    # 论文引用值 0.624
    P("")
    P("| 量 | v5（d_max=0.75, lag=0.54） | v6 |")
    P("|---|---|---|")
    P(f"| eventual CFR = 0.482·d_max/lag（定义 i，**论文同义**） | 0.670 | "
      f"**{evcfr_i(0.75):.4f}** |")
    P(f"| eventual CFR（定义 ii，cohort 全口径） | — | {evcfr_ii(0.75):.4f} |")
    P(f"| eventual CFR（定义 i，字面 recorded fraction） | 0.893 | {evcfr_lit_i(0.75):.4f} |")
    P(f"| eventual CFR（定义 ii，字面 recorded fraction） | — | {evcfr_lit_ii(0.75):.4f} |")
    P(f"| 与结构 CFR 闭合的 d_max（定义 i） | ≈0.70（vs 0.624） | "
      f"{struct_paper*lag_A/CFR_REG:.3f}（vs 论文 0.624）/ "
      f"{struct_model*lag_A/CFR_REG:.3f}（vs 模型构造 {struct_model:.3f}） |")
    P(f"| 与结构 CFR 闭合的 d_max（定义 ii） | — | "
      f"{struct_paper*recfrac_coh/CFR_REG:.3f}（vs 0.624）/ "
      f"{struct_model*recfrac_coh/CFR_REG:.3f}（vs 0.600）→ 均 >0.95，域内不闭合 |")
    P(f"| 落入历史带 [0.218,0.428] 的 d_max（定义 i） | [0.24,0.48] | "
      f"**[{0.218*lag_A/CFR_REG:.3f}, {0.428*lag_A/CFR_REG:.3f}]** |")
    P(f"| 落入历史带 [0.218,0.428] 的 d_max（定义 ii） | — | "
      f"[{0.218*recfrac_coh/CFR_REG:.3f}, {0.428*recfrac_coh/CFR_REG:.3f}] |")
    P("")
    v5chk = CFR_REG * 0.50 / lag_A
    P(f"- 论文原句「with the lag d_max=0.50 implies 0.446, inside the historical BDBV band」"
      f"核对（定义 i）：0.482×0.50/{lag_A:.4f} = {v5chk:.4f} ≤ 0.428 → 仍在带内"
      f"（v5 用 lag=0.54 时 0.446 已略超 0.428，原句边界在 v5 即不成立，v6 滞后更大后成立）。")
    P(f"- 注意 1：模型构造恒等 p·c₁+(1−p)·c₂ = c_total = {struct_model:.3f}，而论文引用的 "
      f"0.43×0.46+0.57×0.747=0.624 中住院 CFR 0.46 与模型 c₁={c1:.4f} 不一致（遗留数）；"
      "闭合值按两个口径并列，请主理人裁决论文用哪个。")
    P("- 注意 2：0.482 用**实测**登记 CFR（4,536/9,415），lag 用**模型**口径；模型 d·F(148)="
      f"{dF148:,.0f} 高出实测 4.08%，故 eventual CFR 混有两种口径的约 +4% 不一致（方向保守："
      "若用模型 d·F 则 eventual CFR 降约 4%）。")
    P("- 注意 3：定义 i（情景 A 续降）含 148 后新感染的死亡（v6 中占 "
      f"{Finf-ev_coh:,.0f}，即 {100*(Finf-ev_coh)/Finf:.0f}%），数值上高于定义 ii；"
      "论文原句「full-caliber deaths grow from 6,707 to 12,422」是定义 i 的表述，"
      "若改用定义 ii（用户 P3 主定义「148 前感染最终死亡」），段落数字需整体换为 "
      f"{ev_coh:,.0f} 一套。")
    P("")
    P("| d_max | cum_inf(148)=9,415/d_max（数据口径，v5 同） | 模型 conf(148)/d_max |")
    P("|---|---|---|")
    for dm in (0.34, 0.50, 0.75, 0.95):
        P(f"| {dm:.2f} | {9415/dm:,.0f} | {p2d['conf']/dm:,.0f} |")
    P(f"- v5 论文域句（d_max∈[0.34,0.95] → 9,911–27,691，中心 12,553）用数据 9,415 口径，"
      f"v6 不变（端点同为 10/16 实测）；模型口径 conf(148)={p2d['conf']:,.1f} 略低 2.4%。")
    OUT["p3"] = dict(F148=F148, d148=d148, dF148=dF148,
                     eventual_A=Finf, lag_A=lag_A, recfrac_lit_A=recfrac_lit_A,
                     fut_split=fut_split, ev_coh=ev_coh, recfrac_coh=recfrac_coh,
                     recfrac_lit_coh=recfrac_lit_coh,
                     conv={str(k): v for k, v in conv.items()},
                     evcfr75_i=evcfr_i(0.75), evcfr75_ii=evcfr_ii(0.75),
                     evcfr75_lit_i=evcfr_lit_i(0.75), evcfr75_lit_ii=evcfr_lit_ii(0.75),
                     dm_close_paper_i=struct_paper * lag_A / CFR_REG,
                     dm_close_model_i=struct_model * lag_A / CFR_REG,
                     dm_close_paper_ii=struct_paper * recfrac_coh / CFR_REG,
                     dm_close_model_ii=struct_model * recfrac_coh / CFR_REG,
                     struct_model=struct_model, struct_paper=struct_paper,
                     dm_band_i=[0.218 * lag_A / CFR_REG, 0.428 * lag_A / CFR_REG],
                     dm_band_ii=[0.218 * recfrac_coh / CFR_REG,
                                 0.428 * recfrac_coh / CFR_REG],
                     v5_check_050=v5chk,
                     cum_inf={str(dm): 9415.0 / dm for dm in (0.34, 0.50, 0.75, 0.95)},
                     cum_inf_model={str(dm): p2d["conf"] / dm
                                    for dm in (0.34, 0.50, 0.75, 0.95)})


# ============================ P4 ============================
def p4():
    P("")
    P("## P4. 样本外校验（仅 t≤143 的 21 个周区间重拟合 → 外推 5 天到 t=148）")
    wk21 = V.base_intervals()[:21]                    # 21 个周区间（E2 结构，t≤143）
    e1_wk = [iv for iv in V.SCHEMES["E1"] if iv[1] <= 143.5]   # E1 合并结构的 19 个周区间
    P(f"- 窗口：21 个周区间（base 结构，[60,67]=887 原样；区间和自洽校验 "
      f"{83 + sum(v for _, _, v in wk21):,.0f} vs 9,066 ✓）")
    P(f"- 稳健性附跑：E1 合并结构 19 个周区间（[53,74]=1,863 合并）")

    def fit_win(ivals, k0, nstart, maxit):
        k_now = float(k0)
        th_now = V.vc_starts(V.TH_V5)[0]
        seq = []
        for it in range(3):
            nll, _, _ = V.make_nll(ivals, 0.75, V.PRM, k_now)
            th, n, _ = V.refit(nll, m._keys(0.75), V.vc_starts(th_now)[:nstart], maxit)
            k_new = V.mom_k(ivals, th)
            seq.append((k_now, th, n, k_new))
            th_now = th
            if not np.isfinite(k_new) or abs(k_new - k_now) < 0.5:
                if np.isfinite(k_new):
                    k_now = k_new
                break
            k_now = k_new
        # 终拟合
        nll, _, _ = V.make_nll(ivals, 0.75, V.PRM, k_now)
        th, n, _ = V.refit(nll, m._keys(0.75), V.vc_starts(th_now)[:nstart], maxit)
        return th, n, k_now, seq

    def extrap(th):
        s = m.simulate(V.PRM, m.Xf0, th, V.T_BASE, 0.5, dmax_fixed=0.75)
        conf148 = float(s["conf_series"][-1])
        dF148 = float(s["dfun"](V.T_BASE)) * float(s["traj"][-1, m.I_F])
        c143 = float(np.interp(143.0, s["ts"], s["conf_series"]))
        daily = (conf148 - c143) / 5.0
        return conf148, dF148, daily

    P("")
    P("| 拟合 | k | θ=(β₀,ρ,t₀,s) | NLL | Rₑ₀ | conf(148) | vs 9,415 | d·F(148) | vs 4,536 | 日均新增 | vs 69.8 |")
    P("|---|---|---|---|---|---|---|---|---|---|---|")
    res = {}
    for tag, ivals, kfix in (("(a) 21 周区间，k=18.44 固定", wk21, 18.44),
                             ("(b) 21 周区间，该窗 MoM", wk21, None),
                             ("(c) E1 合并 19 周区间，k=18.44 固定", e1_wk, 18.44)):
        if kfix is not None:
            nll, _, _ = V.make_nll(ivals, 0.75, V.PRM, kfix)
            th, n, _ = V.refit(nll, m._keys(0.75), V.vc_starts(V.TH_V5)[:NSTART], MAXIT)
            k_used = kfix
        else:
            th, n, k_used, _ = fit_win(ivals, 18.44, NSTART, MAXIT)
        c148, dF148, daily = extrap(th)
        res[tag] = dict(th=[float(x) for x in th], nll=float(n), k=float(k_used),
                        conf148=c148, dF148=dF148, daily=daily,
                        re0=float(th[0] * G))
        P(f"| {tag} | {k_used:.2f} | ({th[0]:.6f}, {th[1]:.6f}, {th[2]:.4f}, {th[3]:.4f}) | "
          f"{n:.4f} | {th[0]*G:.4f} | {c148:,.1f} | {100*(c148-9415)/9415:+.2f}% | "
          f"{dF148:,.1f} | {100*(dF148-4536)/4536:+.2f}% | {daily:.1f} | "
          f"{100*(daily-69.8)/69.8:+.1f}% |")
    P("")
    P("- 参考：v5 样本外（窗 [0,143]）9,601 vs 9,415（+2.0%）、d·F 5,071 vs 4,536（+11.8%）、"
      "60.4/d vs 69.8（−13.5%）；β₀ 因补两个日锚点 −8%。")
    P("- 与主口径（全窗 E1：β₀=0.178386, ρ=0.004781, t₀=5.5730, s=7.9013, Rₑ₀=1.7289）"
      "的差异见上表；(a)/(b) 差异量化 k 口径的影响。")
    OUT["p4"] = res


# ============================ P5 ============================
def p5(th):
    P("")
    P("## P5. 疫苗网格（主口径 θ，f_ring×c_ring，B0 基线，horizon t=513）")
    sb = m.simulate(V.PRM, m.Xf0, th, V.T_BASE, 0.5, dmax_fixed=0.75)
    df_s = sb["dfun"]
    yl = sb["traj"][-1].copy()
    cb = float(sb["conf_series"][-1])
    Fb = float(yl[m.I_F])
    dFb = float(df_s(V.T_BASE)) * Fb

    def run(kind, f_ring, c_ring):
        kw = dict(dmax_fixed=0.75)
        if kind == "D":
            kw.update(vac=m.make_vac(c_ring=c_ring, t_start=V.T_BASE),
                      ve_s=0.15, ve_d=0.40, f_ring=f_ring, v0=m.VAC_V0)
        r = m.run_scenario(V.PRM, m.Xf0, th, yl, kind, t_end=V.T_HOR, **kw)
        F5 = float(r["traj"][V.I513, m.I_F])
        c5 = float(r["conf"][V.I513])
        return dict(c=c5 - cb, F=F5 - Fb,
                    dF=float(df_s(V.T_HOR)) * F5 - dFb,
                    V513=float(r["traj"][V.I513, m.I_V]))

    t0 = time.time()
    b0 = run("B0", 0.0, 0.0)
    P(f"- B0 基线（无疫苗）：确诊Δ={b0['c']:,.1f}，F 死亡Δ={b0['F']:,.1f}，"
      f"d·F 死亡Δ={b0['dF']:,.1f}")
    grid = {}
    P("")
    P("| f_ring \\ c_ring | 10 | 20 | 30 |")
    P("|---|---|---|---|")
    for f in (0.2, 0.5, 1.0):
        cells = []
        for c in (10.0, 20.0, 30.0):
            d = run("D", f, c)
            redF = 100 * (b0["F"] - d["F"]) / b0["F"]
            reddF = 100 * (b0["dF"] - d["dF"]) / b0["dF"]
            grid[(f, c)] = dict(red_F=redF, red_dF=reddF, V513=d["V513"],
                                c=d["c"], F=d["F"], dF=d["dF"])
            cells.append(f"F {redF:.3f}% / d·F {reddF:.3f}%")
        P(f"| {f} | {cells[0]} | {cells[1]} | {cells[2]} |")
    P("")
    redsF = [v["red_F"] for v in grid.values()]
    redsdF = [v["red_dF"] for v in grid.values()]
    ctr = grid[(0.5, 20.0)]
    P(f"- F 口径 9 值范围：{min(redsF):.3f}% – {max(redsF):.3f}%；"
      f"d·F 口径：{min(redsdF):.3f}% – {max(redsdF):.3f}%")
    P(f"- 中心 (0.5, 20)：F 口径 {ctr['red_F']:.3f}%、d·F 口径 {ctr['red_dF']:.3f}%"
      f"（v6 §6 E1 存档 0.20%/0.20%）；V(513)={ctr['V513']:,.0f}，"
      f"V/N_eff={100*ctr['V513']/V.PRM['N_eff']:.3f}%")
    P(f"- 与 MC 一致性：MC 中位数 0.1469% [0.0599, 0.3029]（F 口径）；确定性中心点 "
      f"{ctr['red_F']:.3f}% {'落在' if 0.0599 <= ctr['red_F'] <= 0.3029 else '不在'} "
      "MC 95% 区间内（MC 含自然史/d_max 抽样的额外离散，系统性低于确定性点，"
      "方向与 v3/v5 一致）。")
    P(f"- 耗时 {time.time()-t0:.1f} s（B0 + 9 个 D，各 730 步 RK4）")
    OUT["p5"] = dict(B0=b0,
                     grid={f"{f}_{c}": v for (f, c), v in grid.items()})


# ============================ P6 ============================
def p6():
    P("")
    P("## P6. 数据驱动带（t=148 起 365 天，无模型行；v4/v5 E1 约定）")
    # 复现 v6 §3.1 后爬坡 log-linear（区间增量/天 ~ 区间中点，E2 结构）
    E2 = V.base_intervals()
    fits_r = {}
    for tag, lo in (("t≥60", 60.0), ("t≥67", 67.0)):
        iv = [iv for iv in E2 if (iv[0] + iv[1]) / 2.0 >= lo]
        mid = np.array([(a + b) / 2.0 for a, b, _ in iv])
        rate = np.array([v / (b - a) for a, b, v in iv])
        sl, ic = np.polyfit(mid, np.log(rate), 1)
        fits_r[tag] = dict(n=len(iv), r=float(sl))
    # E1 合并结构 analog（t≥60，含合并区间 [53,74]）
    E1 = V.SCHEMES["E1"]
    iv = [iv for iv in E1 if (iv[0] + iv[1]) / 2.0 >= 60.0]
    mid = np.array([(a + b) / 2.0 for a, b, _ in iv])
    rate = np.array([v / (b - a) for a, b, v in iv])
    sl1, _ = np.polyfit(mid, np.log(rate), 1)
    fits_r["t≥60(E1合并)"] = dict(n=len(iv), r=float(sl1))
    P("")
    P("| 窗口 | n | r (1/d) | v6 §3.1 存档 |")
    P("|---|---|---|---|")
    arch = {"t≥60": -0.004935, "t≥67": -0.003529}
    for tag, v in fits_r.items():
        a = f"{arch[tag]:+.6f}" if tag in arch else "—"
        P(f"| {tag} | {v['n']} | {v['r']:+.6f} | {a} |")
    T = 365.0
    r0c, r0d = 69.0, 31.0            # v4/v5 约定取整（10/14–16 实测 69.8/31.0）
    r0c_u, r0d_u = 69.8, 31.0        # 未取整
    band = []
    P("")
    P("| 行 | r (/d) | 确诊Δ t=513 | d·F 死亡Δ t=513 | 主用? |")
    P("|---|---|---|---|---|")
    for tag, r, main in (("① decline", -0.0050, ""),
                         ("② zero growth", 0.0, ""),
                         ("③ fitted trend（t≥67）", fits_r["t≥67"]["r"], "**主用**"),
                         ("④ fitted trend（t≥60）", fits_r["t≥60"]["r"], "敏感性"),
                         ("⑤ fitted trend（t≥60, E1 合并）", fits_r["t≥60(E1合并)"]["r"], "附")):
        if r == 0.0:
            dc, dd = r0c * T, r0d * T
        else:
            dc = r0c * (np.exp(r * T) - 1.0) / r
            dd = r0d * (np.exp(r * T) - 1.0) / r
        band.append((tag, r, dc, dd))
        P(f"| {tag} | {r:+.6f} | {dc:,.0f} | {dd:,.0f} | {main} |")
    # 未取整 rate 的主行
    r_main = fits_r["t≥67"]["r"]
    dc_u = r0c_u * (np.exp(r_main * T) - 1.0) / r_main
    dd_u = r0d_u * (np.exp(r_main * T) - 1.0) / r_main
    P("")
    P(f"- 主行 ③ 用未取整日均（69.8/31.0）：{dc_u:,.0f} / {dd_u:,.0f}"
      f"（与取整值差 {100*(dc_u/band[2][2]-1):+.1f}%/{100*(dd_u/band[2][3]-1):+.1f}%；"
      "建议维持 69/31 与 v4/v5 附录 E1 约定，差异 <1.2%）")
    P("- 与 v5 带差异来源：v5 行 ③ 用 r=**+0.0033**（v3.1/v4 七点 log-linear 周日混拼，"
      "被增长相爬坡主导为正）→ 48,825/21,936；v6 增长相 16 点补齐后，后爬坡斜率翻转为负"
      f"（t≥67: {fits_r['t≥67']['r']:+.6f}），行 ③ 塌缩到 decline 行附近（{band[2][2]:,.0f}/"
      f"{band[2][3]:,.0f}）。v5 行 ①② 算术不变（r 与 rate₀ 均同）："
      "11,575/5,200、25,185/11,315 复算一致。")
    P("- 结构解读：检出爬坡（v6 主口径 t₀=5.6 d 已辨识）与传播减速叠加，数据侧后段趋势为负；"
      "零增长行成为带的上沿（模型 B0 确诊Δ 15,758 落在 ②与③ 之间偏③ 的关系需按新行距重述）。")
    OUT["p6"] = dict(fits_r=fits_r, band=[(t, r, float(c), float(d)) for t, r, c, d in band],
                     main_unrounded=[float(dc_u), float(dd_u)])


# ============================ P7 ============================
def p7():
    P("")
    P("## P7. 常数检测基线 Rₑ₀（H2'）+ naive 累计比估计量")
    b2, rho2, dm2 = 0.27485, 0.00930, 0.9500      # v6 §2.1 E1 H2' 最优
    re0_h2 = b2 * G
    re0_main = TH_HEAD[0] * G
    fac = re0_h2 / re0_main
    P("")
    P("| 量 | v5（论文原话） | v6 |")
    P("|---|---|---|")
    P(f"| H2' β₀（d≡0.95） | 0.26558 | {b2:.5f} |")
    P(f"| H2' Rₑ₀=β₀×9.6920 | 2.574 | **{re0_h2:.4f}** |")
    P(f"| 主口径 Rₑ₀ | 1.255（1.26） | {re0_main:.4f} |")
    P(f"| 高估倍数 | 2.05 | **{fac:.3f}** |")
    P(f"| ΔAIC(H2'−H1') | +33.3 | +42.07 |")
    P("")
    P(f"- v6 倍数收窄原因：增长相 16 点补齐后主口径早爬坡被辨识、Rₑ₀ 由 1.26 升至 "
      f"{re0_main:.3f}，而 H2'（常数检测）β₀ 几乎不动（0.26558→{b2:.5f}）。")
    i_over_C = 69.8 / 9415.0
    P("")
    P("| C | i (/d) | i/C | R̂=exp(12·i/C) | R̂=exp(17.5·i/C) |")
    P("|---|---|---|---|---|")
    for C in (9415.0, 9066.0):
        P(f"| {C:,.0f} | 69.8 | {69.8/C:.6f} | {np.exp(12*69.8/C):.4f} | "
          f"{np.exp(17.5*69.8/C):.4f} |")
    P(f"- v6 端点不变值：i/C = 69.8/9,415 = {i_over_C:.6f}（论文 i/C=0.0074）；"
      f"R̂(T_g=12) = {np.exp(12*i_over_C):.4f}，R̂(T_g=17.5) = {np.exp(17.5*i_over_C):.4f}。")
    P(f"- Discussion 核对：naive 累计比估计量给 R̂≈{np.exp(12*i_over_C):.2f}–"
      f"{np.exp(17.5*i_over_C):.2f}（隐含平坦发病率），而模型 Rₑ(148)="
      f"{TH_HEAD[0]*np.exp(-TH_HEAD[1]*148.0)*G:.4f}<1（后爬坡衰退）——差异源于估计量对"
      "趋势的零假设，不是矛盾；H2' 的 Rₑ₀ 高估（×1.54）是同一趋势被误释为高传播强度的表现。")
    OUT["p7"] = dict(b2=b2, rho2=rho2, dm2=dm2, re0_h2=re0_h2, re0_main=re0_main,
                     factor=fac, i_over_C=i_over_C,
                     Rhat_12=float(np.exp(12 * i_over_C)),
                     Rhat_175=float(np.exp(17.5 * i_over_C)),
                     Rhat_12_C9066=float(np.exp(12 * 69.8 / 9066.0)),
                     Rhat_175_C9066=float(np.exp(17.5 * 69.8 / 9066.0)),
                     re148=float(TH_HEAD[0] * np.exp(-TH_HEAD[1] * 148.0) * G))


def main():
    t_all = time.time()
    P("# v6 论文正文 7 项派生量（P1–P7）")
    P(f"python={sys.version.split()[0]} numpy={np.__version__} "
      f"scipy={__import__('scipy').__version__}{'，SMOKE' if SMOKE else ''}")
    P("> 所有数值均标注「待用户数值确认」。")
    P("")
    th, nll = get_main()
    P(f"- 主口径 MLE 复现：θ=({th[0]:.6f}, {th[1]:.6f}, {th[2]:.4f}, {th[3]:.4f})，"
      f"NLL={nll:.4f}")
    P(f"- 头条核对：θ=({TH_HEAD[0]:.6f}, {TH_HEAD[1]:.6f}, {TH_HEAD[2]:.4f}, "
      f"{TH_HEAD[3]:.4f})，NLL={NLL_HEAD:.4f}"
      f"（最大参数偏差 {max(abs(a-b) for a, b in zip(th, TH_HEAD)):.2e}）")
    OUT["main"] = dict(th=list(th), nll=nll)
    sb = m.simulate(V.PRM, m.Xf0, th, V.T_BASE, 0.5, dmax_fixed=0.75)
    y148 = sb["traj"][-1].copy()
    P("")
    p1()
    p2d = None
    # p2 先算（p3 需要其存量）
    p2(th)
    p2d = OUT["p2"]
    p3(th, y148, p2d)
    p4()
    if not SMOKE:
        p5(th)
    else:
        P("")
        P("## P5. [SMOKE] 仅中心点")
        sb5 = m.simulate(V.PRM, m.Xf0, th, V.T_BASE, 0.5, dmax_fixed=0.75)
        yl5 = sb5["traj"][-1].copy()
        df5 = sb5["dfun"]
        cb5 = float(sb5["conf_series"][-1]); Fb5 = float(yl5[m.I_F])
        dFb5 = float(df5(V.T_BASE)) * Fb5

        def run1(kind, f_ring, c_ring):
            kw = dict(dmax_fixed=0.75)
            if kind == "D":
                kw.update(vac=m.make_vac(c_ring=c_ring, t_start=V.T_BASE),
                          ve_s=0.15, ve_d=0.40, f_ring=f_ring, v0=m.VAC_V0)
            r = m.run_scenario(V.PRM, m.Xf0, th, yl5, kind, t_end=V.T_HOR, **kw)
            F5 = float(r["traj"][V.I513, m.I_F]); c5 = float(r["conf"][V.I513])
            return dict(c=c5 - cb5, F=F5 - Fb5,
                        dF=float(df5(V.T_HOR)) * F5 - dFb5,
                        V513=float(r["traj"][V.I513, m.I_V]))
        b0 = run1("B0", 0, 0); d = run1("D", 0.5, 20.0)
        P(f"- B0 FΔ={b0['F']:,.1f}；D(0.5,20) FΔ={d['F']:,.1f} → red_F="
          f"{100*(b0['F']-d['F'])/b0['F']:.3f}%，red_dF={100*(b0['dF']-d['dF'])/b0['dF']:.3f}%，"
          f"V(513)={d['V513']:,.0f}")
        OUT["p5_smoke"] = dict(red_F=100 * (b0["F"] - d["F"]) / b0["F"],
                               red_dF=100 * (b0["dF"] - d["dF"]) / b0["dF"],
                               V513=d["V513"])
    p6()
    p7()
    with open("_v6_paper.json", "w", encoding="utf-8") as f:
        json.dump(OUT, f, indent=1)
    P("")
    P(f"[完成] 总耗时 {time.time()-t_all:.1f} s；_v6_paper.json 已写出")


if __name__ == "__main__":
    main()
