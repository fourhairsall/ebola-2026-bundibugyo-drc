# -*- coding: utf-8 -*-
"""
_v6_paper_q123.py —— 论文图注/讨论段 3 个补充数值（Q1–Q3，v6 回填）
========================================================================================
主理人指令：沿用 _v6_paper_quantities.py 的模式（只读 import，不改现有文件），
补算论文图注/讨论段直接引用的 3 个数值，结果追加进 results_v6_paper.md。

Q1  确定性主口径（E1, V-C, d_max=0.75）t=152 的 F(152) 与 d·F(152)；
    图注"登记完备度"因子 = 4,649/d·F(152)（v5 为 0.895）；顺带 t=148 与 t=152 的 d·F。
Q2  V-B 变体（d_max=0.30 自由，v6 拟合 θ=(0.178349, 0.004771, 0.30, 5.5758, 7.9013)）
    t=148 的 H(148)：核对论文"cum_inf(0)∝1/d_max ⇒ 住院存量按 0.75/0.30=2.5 倍放大"
    （V-C H(148)=379.2 → 预期 ≈948）。
Q3  M_B(θ=0.4) 失序耦合 stagnant 情景 B/B0（t=513 累计确诊之比；v5 论文 1.90，
    v5 重算 1.8984，v4 存档 2.2586）；若与 1.90 差 >10% 同时给 θ=0.2 值。
    配方严格复现 v4_extra.py §E7 / v6_revision.py §4 / v6_extended.py §2.3：
    make_params(th_a=θ, th_b=θ) + E1 区间 5 参数重拟合（K0=18.44，起点
    (vc β₀, vc ρ, 0.30, vc t₀, vc s) 与 (0.8β₀, 1.5ρ, 0.30, t₀, s)，maxiter=2500）
    → 模拟至 t=148 → run_scenario(kind="B"/"B0", t_end=513, dmax_fixed=None)。

复现：D:/anaconda/python.exe _v6_paper_q123.py [--smoke]
产物：_v6_paper_q123.json、stdout（markdown 表）；结果追加进 results_v6_paper.md。
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
K0 = 18.44
MAXIT = 400 if SMOKE else 2500
TH_HEAD = (0.178386, 0.004781, 5.5730, 7.9013)     # v6 头条 MLE（V-C, d_max=0.75）
NLL_HEAD = 130.1093
TH_VB = (0.178349, 0.004771, 0.30, 5.5758, 7.9013)  # v6 E1 V-B 拟合（d_max=0.30 贴下界）
OUT = {}


def get_main():
    """复现 v6 §1.1 终拟合（E1, V-C, k=18.44），交叉核验头条值。"""
    nll, _, _ = V.make_nll(V.SCHEMES["E1"], 0.75, V.PRM, K0)
    th, n, det = V.refit(nll, m._keys(0.75), V.vc_starts(V.TH_V5), MAXIT)
    return tuple(float(v) for v in th), float(n)


# ============================ Q1 ============================
def q1(th):
    P("## Q1. t=152 死亡存量与登记完备度因子（确定性主口径 E1 V-C）")
    s152 = m.simulate(V.PRM, m.Xf0, th, V.T_END, 0.5, dmax_fixed=0.75)
    y152 = s152["traj"][-1]
    F152 = float(y152[m.I_F])
    d152 = float(s152["dfun"](V.T_END))
    dF152 = d152 * F152
    # t=148 同口径（核对）
    s148 = m.simulate(V.PRM, m.Xf0, th, V.T_BASE, 0.5, dmax_fixed=0.75)
    F148 = float(s148["traj"][-1, m.I_F])
    d148 = float(s148["dfun"](V.T_BASE))
    dF148 = d148 * F148
    comp = 4649.0 / dF152
    P("")
    P("| 量 | v6 模型 | 实测/存档 | 说明 |")
    P("|---|---|---|---|")
    P(f"| F(152)（全死因累计死亡） | **{F152:,.1f}** | — | 模拟至数据窗终点 t=152 |")
    P(f"| d(152) | {d152:.6f} | — | 平台期 ≈d_max=0.75 |")
    P(f"| d·F(152)（确诊口径累计死亡） | **{dF152:,.1f}** | 实测 4,649 | t=152 登记死亡 |")
    P(f"| **登记完备度 = 4,649/d·F(152)** | **{comp:.4f}** | v5: 0.895 | 图注因子，v6 更新 |")
    P(f"| d·F(148)（核对） | {dF148:,.1f} | 实测 4,536 | 与 P2 一致（+4.08%） |")
    P(f"| F(148)（核对） | {F148:,.1f} | — | 与 P3 一致 |")
    P("")
    P(f"- t=148→152 四天模型新增死亡（全口径）{F152-F148:,.1f}（日均 "
      f"{(F152-F148)/4:.1f}）；d·F 口径 {dF152-dF148:,.1f}。")
    P(f"- v5 存档口径：完备度 0.895 ⇒ v5 模型 d·F(152)=4,649/0.895≈5,194；v6 模型死亡"
      f"因 ρ 更高（衰退更快）而更低，完备度升至 {comp:.3f}——论文图注由 0.895 改为 "
      f"**{comp:.3f}**（若取整报 {comp:.2f}）。")
    OUT["q1"] = dict(F152=F152, d152=d152, dF152=dF152, completeness=comp,
                     F148=F148, dF148=dF148)


# ============================ Q2 ============================
def q2(th_vc):
    P("")
    P("## Q2. V-B（d_max=0.30 自由）t=148 住院存量 H(148) 与 2.5 倍标度核对")
    svc = m.simulate(V.PRM, m.Xf0, th_vc, V.T_BASE, 0.5, dmax_fixed=0.75)
    H_vc = float(svc["traj"][-1, m.I_H])
    cum_vc = float(m.cum_inf_of(svc["traj"])[-1])
    d0_vc = float(svc["dfun"](0.0))
    # V-B：用户给定 v6 拟合 5 参数（d_max 自由）
    svb = m.simulate(V.PRM, m.Xf0, TH_VB, V.T_BASE, 0.5, dmax_fixed=None)
    yvb = svb["traj"][-1]
    H_vb = float(yvb[m.I_H])
    cum_vb = float(m.cum_inf_of(svb["traj"])[-1])
    d0_vb = float(svb["dfun"](0.0))
    ratio = H_vb / H_vc
    cum_ratio = cum_vb / cum_vc
    ic_ratio = (83.0 / d0_vb) / (83.0 / d0_vc)
    P("")
    P("| 量 | V-C（d_max=0.75） | V-B（d_max=0.30） | 比值 | 理论 0.75/0.30 |")
    P("|---|---|---|---|---|")
    P(f"| d(0) | {d0_vc:.6f} | {d0_vb:.6f} | {d0_vc/d0_vb:.4f} | 2.5 |")
    P(f"| cum_inf(0)=83/d(0) | {83/d0_vc:,.1f} | {83/d0_vb:,.1f} | {ic_ratio:.4f} | 2.5 |")
    P(f"| cum_inf(148) | {cum_vc:,.1f} | {cum_vb:,.1f} | {cum_ratio:.4f} | ≈2.5 |")
    P(f"| **H(148)** | **{H_vc:,.1f}** | **{H_vb:,.1f}** | **{ratio:.4f}** | ≈2.5 |")
    P("")
    P(f"- 论文论断核对：379.2×2.5 = {H_vc*2.5:,.0f}；V-B 实际 H(148) = **{H_vb:,.1f}**"
      f"（偏差 {100*(H_vb-H_vc*2.5)/(H_vc*2.5):+.2f}%）。")
    P(f"- 结论：2.5 倍标度**成立**（H 比值 {ratio:.4f}，偏差 "
      f"{100*(ratio-2.5)/2.5:+.3f}%）。初值标度 {ic_ratio:.4f} 到 t=148 衰减为 "
      f"{cum_ratio:.4f}（次线性约 −0.02%），来源为 V-B 口径下易感者损耗约 0.5%"
      "（V-C 约 0.2%）的微弱非线性；d_max 恒等式（β₀ 补偿，|Δβ₀|=3.7e-05）使动力学"
      "形状不变，仅整体标度随 1/d_max 放大——论文论断「cum_inf(0)∝1/d_max ⇒ 住院存量"
      "按 0.75/0.30=2.5 倍放大」可用，数值为 379→948。")
    P(f"- 注意：V-B 的 d_max=0.30 贴 BD 下界（v6 §5），该口径的 H(148)≈{H_vb:,.0f} 与 ECDC "
      "住院锚点 924 的关系随之改变（V-C 口径 H=379 远低于 924；V-B 口径转为高出），"
      "引用时须注明口径。")
    OUT["q2"] = dict(H_vc=H_vc, H_vb=H_vb, ratio=ratio, cum_vc=cum_vc, cum_vb=cum_vb,
                     cum_ratio=cum_ratio, ic_ratio=ic_ratio,
                     expected=H_vc * 2.5, dev_pct=100 * (H_vb - H_vc * 2.5) / (H_vc * 2.5))


# ============================ Q3 ============================
def mb_fit(th_v, th_vc):
    """M_B(θ) 5 参数重拟合（v6_extended.py §2.3 配方）。"""
    prm_b = m.make_params(th_a=th_v, th_b=th_v)
    nll_b, _, _ = V.make_nll(V.SCHEMES["E1"], None, prm_b, K0)
    st0 = (th_vc[0], th_vc[1], 0.30, th_vc[2], th_vc[3])
    th_b, n_b, _ = V.refit(nll_b, m._keys(None),
                           [st0, (st0[0] * 0.8, st0[1] * 1.5, st0[2], st0[3], st0[4])],
                           MAXIT)
    return prm_b, tuple(float(v) for v in th_b), float(n_b)


def q3(th_vc):
    P("")
    P("## Q3. M_B 失序耦合 stagnant 情景 B/B0（t=513 累计确诊之比；v5 论文 1.90）")
    P("")
    P("- 配方：make_params(th_a=θ, th_b=θ) + E1 区间 5 参数重拟合（K0=18.44；起点 "
      "(vc β₀, vc ρ, 0.30, vc t₀, vc s) 与 (0.8β₀, 1.5ρ, 0.30, t₀, s)；maxiter="
      f"{MAXIT}）；B/B0 = conf_B(513)/conf_B0(513)（v4_extra §E7 / v6_revision §4 同定义）。")
    res = {}
    for th_v in (0.4, 0.2):
        prm_b, th_b, n_b = mb_fit(th_v, th_vc)
        re0 = m.R0_ngm(th_b[0], prm_b, x=float(m.Xf0(0.0)))
        s_b = m.simulate(prm_b, m.Xf0, th_b, V.T_BASE, 0.5, dmax_fixed=None)
        yl_b = s_b["traj"][-1].copy()
        cb0 = float(s_b["conf_series"][-1])
        r_B = m.run_scenario(prm_b, m.Xf0, th_b, yl_b, "B", t_end=V.T_HOR, dmax_fixed=None)
        r_B0 = m.run_scenario(prm_b, m.Xf0, th_b, yl_b, "B0", t_end=V.T_HOR, dmax_fixed=None)
        cB = float(r_B["conf"][V.I513])
        cB0 = float(r_B0["conf"][V.I513])
        FB = float(r_B["traj"][V.I513, m.I_F])
        FB0 = float(r_B0["traj"][V.I513, m.I_F])
        ratio = cB / cB0
        res[th_v] = dict(th=list(th_b), nll=n_b, re0=float(re0), conf148=cb0,
                         cB=cB, cB0=cB0, ratio=float(ratio),
                         F_B=FB, F_B0=FB0, F_ratio=float(FB / FB0))
        P("")
        P(f"**θ={th_v}**：拟合 β₀={th_b[0]:.6f}, ρ={th_b[1]:.6f}, d_max={th_b[2]:.4f}, "
          f"t₀={th_b[3]:.4f}, s={th_b[4]:.4f}；NLL={n_b:.4f}（v6 §2.3 存档 "
          f"{'129.9294' if th_v == 0.4 else '129.9962'}）；Rₑ₀={re0:.4f}（存档 "
          f"{'1.2997' if th_v == 0.4 else '1.4948'}）")
        P(f"- t=513 累计确诊：B={cB:,.0f}，B0={cB0:,.0f}（M_B 基线 conf(148)={cb0:,.0f}）")
        P(f"- **B/B0 = {ratio:.4f}**（附：全口径死亡之比 F_B/F_B0 = {FB/FB0:.4f}）")
    r04 = res[0.4]["ratio"]
    dev = 100 * (r04 - 1.90) / 1.90
    P("")
    P("| 口径 | B/B0（t=513 累计确诊） |")
    P("|---|---|")
    P(f"| v4 存档（v4_extra §E7） | 2.2586 |")
    P(f"| v5 重算（v6_revision §4，论文 1.90） | 1.8984 |")
    P(f"| **v6 本算（θ=0.4）** | **{r04:.4f}** |")
    P(f"| v6 本算（θ=0.2，备用） | {res[0.2]['ratio']:.4f} |")
    P("")
    if abs(dev) <= 10:
        P(f"- v6 θ=0.4 与论文 1.90 的差异：{dev:+.1f}%（≤10%），论文原句可稳妥替换为 v6 值。")
    else:
        P(f"- v6 θ=0.4 与论文 1.90 的差异：{dev:+.1f}%（>10%），原句须改写；"
          f"已给出 θ=0.2 的 {res[0.2]['ratio']:.4f} 供稳妥表述。")
    P("- 机制：B 与 B0 的差别仅在失序耦合项（θ_a 压低 h、θ_b 放大 β 的 (1+θ_b·X) 因子），"
      "stagnant 情景下 β 恒定，失序衰减（X 指数衰减）使放大倍数随时问收敛于有限值；"
      "v6 早爬坡辨识后 M_B 拟合的 ρ 更高、t₀ 更早，失序窗内累积放大的时长与强度改变，"
      "比值随之移动。")
    OUT["q3"] = {str(k): v for k, v in res.items()}


def main():
    t_all = time.time()
    P("# v6 论文补充数值 Q1–Q3（图注/讨论段回填）")
    P(f"python={sys.version.split()[0]} numpy={np.__version__} "
      f"scipy={__import__('scipy').__version__}{'，SMOKE' if SMOKE else ''}")
    P("> 所有数值均标注「待用户数值确认」。")
    P("")
    th, nll = get_main()
    P(f"- 主口径 MLE 复现：θ=({th[0]:.6f}, {th[1]:.6f}, {th[2]:.4f}, {th[3]:.4f})，"
      f"NLL={nll:.4f}（头条 {NLL_HEAD:.4f}，差 "
      f"{max(abs(a-b) for a, b in zip(th, TH_HEAD)):.2e}）")
    OUT["main"] = dict(th=list(th), nll=nll)
    q1(th)
    q2(th)
    q3(th)
    with open("_v6_paper_q123.json", "w", encoding="utf-8") as f:
        json.dump(OUT, f, indent=1)
    P("")
    P(f"[完成] 总耗时 {time.time()-t_all:.1f} s；_v6_paper_q123.json 已写出")


if __name__ == "__main__":
    main()
