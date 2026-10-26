# -*- coding: utf-8 -*-
"""
_v6_basins.py —— v6 E1 V-C 似然的全局盆地搜索 + 方案 E4 + （条件性）双分量检测函数
================================================================================
背景（主理人复核提出的全局最优性问题）：
  v6 主口径 E1 V-C 拟合落在**早爬坡盆地**（t₀=5.57 d, s=7.90, β₀=0.1784,
  NLL=130.1093, Rₑ₀=1.73）。该盆地可能只是局部最优：晚爬坡解读（t₀≈50–60 d，
  即 7–8 月检出份额才开始上升）与数据形态同样自洽。若晚爬坡盆地 NLL 与早爬坡
  相差 <2，则 Rₑ₀ 结论完全反转。本脚本系统搜索全局盆地并给出判定。

任务（按序）：
  T1  E1 V-C（k=18.44, d_max=0.75 固定, 23 区间）全局多起点搜索
      - 起点盒（主理人指定）：β₀∈[0.03,0.60], ρ∈[0,0.020], t₀∈[−20,140], s∈[3,60]
      - 分层抽样 LHS 256 起点 + ~94 个定向起点（v5/v6 解、晚爬坡猜测、盒角点、
        v6 邻域扰动对照）
      - maxiter=600 粗筛（xatol=1e-6, fatol=1e-8）→ 聚类（β₀ 相对 10%、
        |Δt₀|<5 d、s 相对 20%）→ 每盆地 3 扰动起点 × maxiter=2500 精炼
        （xatol=1e-10, fatol=1e-12）
      - 附加扩展盒稳健性搜索：β₀∈[0.03,3.0], ρ∈[0,0.05], t₀∈[−20,220], s∈[3,120]
        （LHS 128 + 定向），排除"晚爬坡盆地需要更大 β₀ 而躲在指定盒外"的可能
      - 每盆地诊断：(β₀,ρ,t₀,s,NLL,AIC,Rₑ₀)、逐区间偏差 pattern、
        d(t)@t=0/12/32/60/92/148、早段真实感染增速 r_early=ln(ci(12)/ci(5))/7、
        cum_inf(152)、Rₑ₀>3 判不可信标记
  T2  方案 E4：疑似积压重分类污染 [0,5] 与 [5,12] → 似然从 t=12 开始
      （21 区间；C(0)=83 仍作 build_ic 锚点；k=18.44 保持可比）；拟 V-C/V-B 全套
  T3  （条件性）双分量检测函数 d(t)=d_max·[w·σ((t−t₁)/s₁)+(1−w)·σ((t−t₂)/s₂)]
      7 参数（d_max=0.75 固定）：报告 NLL 改善与 ΔAIC；仅当 T1/T2 表明单 logistic
      结构性不足时运行
  T4  结果汇总写入 results_v6_basins.md（由本脚本之外按 JSON 落盘数据撰写，
      本脚本只产出 _basins_t1/t2/t3.json + 控制台表 + fig_v6_basins.png）

纪律：不改 ebola_model_v3.py / v6_extended.py；所有数值标「待用户数值确认」；
      不为跑通而改模型；负值/非单调/不守恒即报。
复现：D:/anaconda/python.exe _v6_basins.py --phase t1   （t2/t3/fig 同理）
"""
import io
import sys
import time
import json
import numpy as np
import multiprocessing as mp
from scipy.optimize import minimize
from scipy.stats import qmc
import ebola_model_v3 as m
import v6_extended as V

m.T_LAST = V.T_BASE

# ---------------- 全局常量 ----------------
K0 = 18.44                                   # v6 主口径 MoM k_obs（保持可比）
TH_V6 = (0.178386, 0.004781, 5.5730, 7.9013)  # v6 E1 V-C 解（对照）
IV_E1 = V.SCHEMES["E1"]
IV_E4 = [iv for iv in IV_E1 if iv[0] >= 12.0]  # T2：剔除 [0,5]/[5,12]（重分类污染）
N_PROC = 12
P = lambda s="": print(s, flush=True)

LO_SPEC = np.array([0.03, 0.0, -20.0, 3.0])    # 主理人指定搜索盒
HI_SPEC = np.array([0.60, 0.020, 140.0, 60.0])
LO_EXT = np.array([0.03, 0.0, -20.0, 3.0])     # 扩展盒（稳健性）
HI_EXT = np.array([3.0, 0.05, 220.0, 120.0])

# 聚类容差（主理人指定）
TOL_BETA, TOL_T0, TOL_S = 0.10, 5.0, 0.20

DQ = [0, 12, 32, 60, 92, 148]                  # d(t) 诊断点


# ---------------- 起点生成 ----------------
def lhs_starts(n, lo, hi, seed):
    """分层抽样（LHS）n 个 4 维起点。"""
    eng = qmc.LatinHypercube(d=4, seed=seed)
    u = eng.random(n)
    return [tuple(float(lo[j] + u[i, j] * (hi[j] - lo[j])) for j in range(4))
            for i in range(n)]


def lhs_starts5(n, lo, hi, seed):
    eng = qmc.LatinHypercube(d=5, seed=seed)
    u = eng.random(n)
    return [tuple(float(lo[j] + u[i, j] * (hi[j] - lo[j])) for j in range(5))
            for i in range(n)]


def directed_spec():
    """指定盒内的定向起点：v5/v6 解、晚爬坡猜测网格、盒角点、v6 邻域扰动对照。"""
    out = [tuple(float(x) for x in V.TH_V5), TH_V6]
    for t0 in (35., 40., 45., 50., 55., 60., 65., 70., 80., 90., 110., 130.):
        for (b0, rh, sd) in ((0.08, 0.001, 8.), (0.20, 0.003, 12.),
                             (0.35, 0.007, 18.), (0.55, 0.015, 25.)):
            out.append((b0, rh, t0, sd))
    for b0 in (0.03, 0.60):
        for rh in (0.0, 0.020):
            for t0 in (-20., 70., 140.):
                for sd in (3., 60.):
                    out.append((b0, rh, t0, sd))
    rng = np.random.default_rng(7)
    for _ in range(24):                          # v6 邻域扰动：对照盆地必须被重新找到
        out.append((TH_V6[0] * rng.uniform(0.8, 1.25), TH_V6[1] * rng.uniform(0.5, 2.0),
                    TH_V6[2] + rng.uniform(-6, 6), TH_V6[3] * rng.uniform(0.7, 1.4)))
    return out


def directed_ext():
    """扩展盒定向起点：高 β₀ 晚爬坡（指定盒外）+ v5/v6。"""
    out = [tuple(float(x) for x in V.TH_V5), TH_V6]
    for t0 in (40., 60., 80., 100., 130., 160., 200.):
        for (b0, rh, sd) in ((0.6, 0.01, 15.), (1.0, 0.02, 25.),
                             (2.0, 0.03, 40.), (3.0, 0.05, 60.)):
            out.append((b0, rh, t0, sd))
    return out


# ---------------- 并行 worker（spawn 安全：全部模块级） ----------------
def fit_worker(arg):
    """单个起点的 Nelder-Mead；返回 (i, theta, nll, nit, nfev)。"""
    i, th0, ivals, k0, dmf, maxiter, xatol, fatol = arg
    nll, _, _ = V.make_nll(ivals, dmf, V.PRM, k0)
    keys = m._keys(dmf)
    st = tuple(min(max(v, m.BD[k][0]), m.BD[k][1]) for v, k in zip(th0, keys))
    r = minimize(nll, st, method="Nelder-Mead",
                 options=dict(xatol=xatol, fatol=fatol, maxiter=maxiter,
                              maxfev=maxiter, adaptive=True))
    return (i, tuple(float(v) for v in r.x), float(r.fun), int(r.nit), int(r.nfev))


def run_pool(args, tag=""):
    """imap_unordered + 逐条进度。"""
    t0 = time.time()
    res = []
    with mp.Pool(processes=N_PROC) as pool:
        for k, out in enumerate(pool.imap_unordered(fit_worker, args, chunksize=1), 1):
            res.append(out)
            if k % 25 == 0 or k == len(args):
                P(f"  [{tag}] {k}/{len(args)} 起点完成（{time.time()-t0:.0f} s）")
    return res


# ---------------- 聚类 ----------------
def cluster(sols, nlls):
    """单链接聚类：β₀ 相对 10% & |Δt₀|<5 d & s 相对 20%。返回盆地列表。"""
    order = list(np.argsort(nlls))
    basins = []                                  # 每盆：dict(idx=[...])
    for i in order:
        th = sols[i]
        for b in basins:
            ok = True
            for j in b["idx"]:
                rb = sols[j]
                if (abs(th[0] - rb[0]) > TOL_BETA * max(abs(rb[0]), 1e-9)
                        or abs(th[2] - rb[2]) >= TOL_T0
                        or abs(th[3] - rb[3]) > TOL_S * max(abs(rb[3]), 1e-9)):
                    ok = False
                    break
            if ok:
                b["idx"].append(i)
                break
        else:
            basins.append(dict(idx=[i]))
    return basins


# ---------------- 盆地诊断 ----------------
def basin_diag(th, ivals, dmf, k0):
    nll, obs, times = V.make_nll(ivals, dmf, V.PRM, k0)
    n = float(nll(th))
    np_ = len(m._keys(dmf))
    s, c, idx, mods, tms = V.sim_pred(th, ivals, dmf)
    dev = [100.0 * (mo - ob) / ob for mo, ob in zip(mods, obs)]
    s2 = m.simulate(V.PRM, m.Xf0, th, V.T_END, 0.5,
                    t_grid=np.array([0., 5., 12., 152.]), dmax_fixed=dmf)
    ci = s2["cum_inf_at"]
    r_early = float(np.log(ci[2] / ci[1]) / 7.0)
    b0, rh = th[0], th[1]
    re0 = float(m.R0_ngm(b0, V.PRM, x=float(m.Xf0(0.0))))
    re148 = float(m.R0_ngm(b0 * np.exp(-rh * 148.0), V.PRM, x=float(m.Xf0(148.0))))
    dvals = [float(s["dfun"](t)) for t in DQ]
    return dict(theta=[float(v) for v in th], nll=n, aic=2 * n + 2 * np_,
                re0=re0, re148=re148, r_early=r_early,
                cum_inf_152=float(ci[3]), conf_152=float(c[-1]),
                d_at=dvals, dq=DQ,
                dev=[float(x) for x in dev],
                iv=[[float(a), float(b), float(v)] for a, b, v in ivals],
                obs=[float(x) for x in obs], mod=[float(x) for x in mods],
                re0_gt3=bool(re0 > 3.0))


def summarize_basin(name, diag, n_starts):
    th = diag["theta"]
    dev = np.array(diag["dev"])
    P(f"- 盆地 {name}：起点数 {n_starts}")
    P(f"    θ=(β₀={th[0]:.6f}, ρ={th[1]:.6f}, t₀={th[2]:.4f}, s={th[3]:.4f})"
      f"  NLL={diag['nll']:.4f}  AIC={diag['aic']:.4f}  Rₑ₀={diag['re0']:.4f}"
      f"  Rₑ(148)={diag['re148']:.4f}"
      + ("  [Rₑ₀>3，判不可信]" if diag["re0_gt3"] else ""))
    P(f"    d(t)@0/12/32/60/92/148 = " + "/".join(f"{x:.4f}" for x in diag["d_at"]))
    P(f"    早段真实感染增速 r_early={diag['r_early']:.4f} /d；"
      f"cum_inf(152)={diag['cum_inf_152']:,.0f}；conf(152)={diag['conf_152']:,.0f}")
    P(f"    逐区间偏差%：" + " ".join(f"{x:+.1f}" for x in dev))


# ---------------- T1 ----------------
def phase_t1():
    P("## T1  E1 V-C 全局多起点盆地搜索（主盒：LHS 256 + 定向）")
    P(f"   区间数={len(IV_E1)}；k_obs={K0}；d_max=0.75 固定；容差 "
      f"β₀±10% / |Δt₀|<5 d / s±20%")
    starts = lhs_starts(256, LO_SPEC, HI_SPEC, 20261101) + directed_spec()
    P(f"   起点总数 {len(starts)}（LHS 256 + 定向 {len(directed_spec())}）")
    args = [(i, st, IV_E1, K0, 0.75, 600, 1e-6, 1e-8) for i, st in enumerate(starts)]
    import os
    if os.path.exists("_basins_t1_coarse.json"):
        with open("_basins_t1_coarse.json", encoding="utf-8") as f:
            ck = json.load(f)
        res = [(r["i"], tuple(r["theta"]), r["nll"], r["nit"], r["nfev"]) for r in ck]
        P("   [断点] 复用已存粗筛结果 _basins_t1_coarse.json"
          f"（{len(res)} 起点，NLL 范围 [{min(r[2] for r in res):.2f},"
          f" {max(r[2] for r in res):.2f}]）")
    else:
        t0 = time.time()
        res = run_pool(args, "T1-coarse")
        P(f"   粗筛完成：{time.time()-t0:.0f} s")
        with open("_basins_t1_coarse.json", "w", encoding="utf-8") as f:
            json.dump([dict(i=r[0], theta=list(r[1]), nll=r[2], nit=r[3], nfev=r[4])
                       for r in res], f)
    sols = [r[1] for r in res]
    nlls = np.array([r[2] for r in res])
    ok = nlls < 1e8
    P(f"   有效起点 {int(ok.sum())}/{len(res)}；NLL 范围 "
      f"[{nlls[ok].min():.2f}, {np.sort(nlls[ok])[-1]:.2f}]")
    basins = cluster(sols, nlls)
    P(f"   粗筛独特盆地数：{len(basins)}")
    # 每盆地精炼（代表起点 + 2 扰动）
    ref_starts = []
    for b in basins:
        rep = sols[b["idx"][0]]                  # idx 按 NLL 升序，[0] 即代表
        ref_starts.append(rep)
        ref_starts.append((rep[0] * 0.9, rep[1], rep[2] - 3.0, rep[3] * 1.1))
        ref_starts.append((rep[0] * 1.1, rep[1] * 1.5, rep[2] + 3.0, rep[3] * 0.9))
    ref_args = [(i, st, IV_E1, K0, 0.75, 2500, 1e-10, 1e-12)
                for i, st in enumerate(ref_starts)]
    t0 = time.time()
    rres = run_pool(ref_args, "T1-refine")
    P(f"   精炼完成：{time.time()-t0:.0f} s")
    out_basins = []
    for bi, b in enumerate(basins):
        grp = rres[3 * bi: 3 * bi + 3]
        best = min(grp, key=lambda r: r[2])
        diag = basin_diag(best[1], IV_E1, 0.75, K0)
        diag["n_starts"] = len(b["idx"])
        diag["coarse_rep"] = [float(v) for v in sols[b["idx"][0]]]
        diag["coarse_nll"] = float(nlls[b["idx"][0]])
        out_basins.append(diag)
    out_basins.sort(key=lambda d: d["nll"])
    P("")
    P("### T1 主盒盆地（按 NLL 升序）")
    for bi, d in enumerate(out_basins):
        summarize_basin(f"T1-B{bi+1}", d, d["n_starts"])
    best = out_basins[0]
    P("")
    P(f"   与 v6 解对照：ΔNLL={best['nll']-130.10933525761075:+.4f}；"
      f"θ 差 β₀={best['theta'][0]-TH_V6[0]:+.2e}, ρ={best['theta'][1]-TH_V6[1]:+.2e}, "
      f"t₀={best['theta'][2]-TH_V6[2]:+.3f}, s={best['theta'][3]-TH_V6[3]:+.3f}")
    P(f"   晚爬坡（t₀>40）可比盆地（ΔNLL<2 vs 最优）存在？"
      f" {any(d['theta'][2] > 40 and d['nll'] - best['nll'] < 2.0 for d in out_basins)}")

    # 中间落盘（防止 T1b 阶段中断丢失主盒结果）
    with open("_basins_t1.json", "w", encoding="utf-8") as f:
        json.dump(dict(meta=dict(k0=K0, n_ivals=len(IV_E1),
                                 lo_spec=list(map(float, LO_SPEC)),
                                 hi_spec=list(map(float, HI_SPEC)),
                                 lo_ext=list(map(float, LO_EXT)),
                                 hi_ext=list(map(float, HI_EXT)),
                                 tol=[TOL_BETA, TOL_T0, TOL_S]),
                       n_starts_total=len(starts),
                       coarse=[dict(i=r[0], theta=list(r[1]), nll=r[2],
                                    nit=r[3], nfev=r[4]) for r in res],
                       basins=out_basins, ext_basins=[]), f, indent=1)

    # ---- 扩展盒稳健性 ----
    P("")
    P("## T1b  扩展盒稳健性（β₀≤3.0, ρ≤0.05, t₀≤220, s≤120；LHS 128 + 定向）")
    starts2 = lhs_starts(128, LO_EXT, HI_EXT, 20261102) + directed_ext()
    P(f"   起点总数 {len(starts2)}")
    args2 = [(i, st, IV_E1, K0, 0.75, 600, 1e-6, 1e-8) for i, st in enumerate(starts2)]
    t0 = time.time()
    res2 = run_pool(args2, "T1b-coarse")
    P(f"   粗筛完成：{time.time()-t0:.0f} s")
    sols2 = [r[1] for r in res2]
    nlls2 = np.array([r[2] for r in res2])
    basins2 = cluster(sols2, nlls2)
    P(f"   粗筛独特盆地数：{len(basins2)}")
    ref2 = []
    for j, b in enumerate(basins2):
        rep = sols2[b["idx"][0]]
        ref2.append((j, rep, IV_E1, K0, 0.75, 2500, 1e-10, 1e-12))
    rres2 = run_pool(ref2, "T1b-refine")
    out2 = []
    for bi, b in enumerate(basins2):
        d = basin_diag(rres2[bi][1], IV_E1, 0.75, K0)
        d["n_starts"] = len(b["idx"])
        out2.append(d)
    out2.sort(key=lambda d: d["nll"])
    P("")
    P("### T1b 扩展盒盆地（按 NLL 升序）")
    for bi, d in enumerate(out2):
        summarize_basin(f"T1b-B{bi+1}", d, d["n_starts"])

    with open("_basins_t1.json", "w", encoding="utf-8") as f:
        json.dump(dict(meta=dict(k0=K0, n_ivals=len(IV_E1),
                                 lo_spec=list(map(float, LO_SPEC)),
                                 hi_spec=list(map(float, HI_SPEC)),
                                 lo_ext=list(map(float, LO_EXT)),
                                 hi_ext=list(map(float, HI_EXT)),
                                 tol=[TOL_BETA, TOL_T0, TOL_S]),
                       n_starts_total=len(starts),
                       coarse=[dict(i=r[0], theta=list(r[1]), nll=r[2],
                                    nit=r[3], nfev=r[4]) for r in res],
                       basins=out_basins, ext_basins=out2), f, indent=1)
    P("")
    P("[T1 完成] _basins_t1.json 已写出")
    return out_basins, out2


# ---------------- T2（方案 E4） ----------------
def phase_t2():
    P("## T2  方案 E4（剔除 [0,5]/[5,12] 重分类污染；似然自 t=12 起）")
    P(f"   区间数={len(IV_E4)}（E1 23 − 2）；k_obs={K0} 固定保持可比；"
      f"C(0)=83 仍为 build_ic 锚点")
    # E1 主口径解对两个被剔区间的预测 vs 实测（污染评估）
    s, c, idx, mods, tms = V.sim_pred(TH_V6, IV_E1, 0.75)
    obs = np.array([iv[2] for iv in IV_E1])
    for iv, mo, ob in zip(IV_E1, mods, obs):
        if iv[0] < 12.0:
            P(f"   E1(v6) 预测 [{iv[0]:.0f},{iv[1]:.0f}]：模型 {mo:,.1f} vs 实测 "
              f"{ob:,.0f}（偏差 {100*(mo-ob)/ob:+.1f}%）")
    # ---- V-C 盆地搜索 ----
    starts = lhs_starts(200, LO_SPEC, HI_SPEC, 20261103) + directed_spec()
    P(f"   V-C 起点总数 {len(starts)}（LHS 200 + 定向 {len(directed_spec())}）")
    args = [(i, st, IV_E4, K0, 0.75, 600, 1e-6, 1e-8) for i, st in enumerate(starts)]
    t0 = time.time()
    res = run_pool(args, "T2-coarse")
    P(f"   粗筛完成：{time.time()-t0:.0f} s")
    sols = [r[1] for r in res]
    nlls = np.array([r[2] for r in res])
    basins = cluster(sols, nlls)
    P(f"   粗筛独特盆地数：{len(basins)}")
    ref_starts = []
    for b in basins:
        rep = sols[b["idx"][0]]
        ref_starts.append(rep)
        ref_starts.append((rep[0] * 0.9, rep[1], rep[2] - 3.0, rep[3] * 1.1))
        ref_starts.append((rep[0] * 1.1, rep[1] * 1.5, rep[2] + 3.0, rep[3] * 0.9))
    ref_args = [(i, st, IV_E4, K0, 0.75, 2500, 1e-10, 1e-12)
                for i, st in enumerate(ref_starts)]
    rres = run_pool(ref_args, "T2-refine")
    out_vc = []
    for bi, b in enumerate(basins):
        grp = rres[3 * bi: 3 * bi + 3]
        best = min(grp, key=lambda r: r[2])
        d = basin_diag(best[1], IV_E4, 0.75, K0)
        d["n_starts"] = len(b["idx"])
        out_vc.append(d)
    out_vc.sort(key=lambda d: d["nll"])
    P("")
    P("### T2 E4 V-C 盆地（按 NLL 升序）")
    for bi, d in enumerate(out_vc):
        summarize_basin(f"T2-B{bi+1}", d, d["n_starts"])
    # E4 最优解处 MoM k̂（诊断；拟合仍用 k=18.44）
    k4 = V.mom_k(IV_E4, out_vc[0]["theta"])
    P(f"   E4 最优解处 MoM k̂ = {k4:.2f}（诊断用；拟合固定 k={K0}）")
    # ---- V-B（5 参数，d_max 自由；嵌套保护锚点） ----
    th4 = out_vc[0]["theta"]
    starts5 = [(th4[0], th4[1], 0.75, th4[2], th4[3])]
    starts5 += lhs_starts5(64, np.array([0.03, 0.0, 0.30, -20.0, 3.0]),
                           np.array([0.60, 0.02, 0.95, 140.0, 60.0]), 20261104)
    for dm in (0.30, 0.60, 0.95):
        starts5.append((th4[0], th4[1], dm, th4[2], th4[3]))
        starts5.append((th4[0] * 0.85, th4[1] * 1.5, dm, th4[2] + 3.0, th4[3] * 1.1))
    P(f"   V-B 起点总数 {len(starts5)}（含 V-C 锚点 + LHS 64 + d_max 扫描）")
    args5 = [(i, st, IV_E4, K0, None, 600, 1e-6, 1e-8)
             for i, st in enumerate(starts5)]
    res5 = run_pool(args5, "T2-VB")
    best5 = min(res5, key=lambda r: r[2])
    # 锚点精炼（保证 ≥ V-C 的 NLL）
    r5 = fit_worker((0, tuple(best5[1]), IV_E4, K0, None, 2500, 1e-10, 1e-12))
    d5 = basin_diag(r5[1], IV_E4, None, K0)
    nest = d5["nll"] <= out_vc[0]["nll"] + 1e-6
    P(f"   E4 V-B 最优：θ=(β₀={d5['theta'][0]:.6f}, ρ={d5['theta'][1]:.6f}, "
      f"d_max={d5['theta'][2]:.4f}, t₀={d5['theta'][3]:.4f}, s={d5['theta'][4]:.4f})")
    P(f"   NLL={d5['nll']:.4f}  AIC={d5['aic']:.4f}  Rₑ₀={d5['re0']:.4f}；"
      f"嵌套保护 {'满足' if nest else '违例（已补跑锚点）'}")
    with open("_basins_t2.json", "w", encoding="utf-8") as f:
        json.dump(dict(meta=dict(k0=K0, n_ivals=len(IV_E4),
                                 iv_e4=[[float(a), float(b), float(v)]
                                        for a, b, v in IV_E4]),
                       vc_basins=out_vc, vb=d5, vb_nest=bool(nest),
                       mom_k_e4=float(k4)), f, indent=1)
    P("")
    P("[T2 完成] _basins_t2.json 已写出")
    return out_vc, d5


# ---------------- T3（条件性：双分量检测函数，7 参数） ----------------
def make_dfun2(d_max, w, t1, s1, t2, s2):
    def f(t):
        ta = np.asarray(t, float)
        v = d_max * (w / (1.0 + np.exp(-(ta - t1) / s1))
                     + (1.0 - w) / (1.0 + np.exp(-(ta - t2) / s2)))
        return float(v) if np.ndim(ta) == 0 else v
    return f


def simulate_custom(prm, Xf, beta0, rho, dfun, t_end, dt=0.5, t_grid=None):
    """复制 m.simulate 的积分逻辑，仅替换 dfun（不改主代码）。"""
    y0 = m.build_ic(float(dfun(0.0)), prm, Xf)
    beta_fn = lambda t: beta0 * np.exp(-rho * t)
    f = lambda t, y: m.rhs(t, y, prm, Xf, beta_fn, dfun, None, 0., 0., 0., False)
    ts, traj = m.rk4(f, y0, 0.0, t_end, dt)
    ci = m.cum_inf_of(traj)
    out = dict(ts=ts, traj=traj, cum_inf=ci, dfun=dfun, y0=y0)
    if t_grid is not None:
        tg = np.asarray(t_grid, float)
        ci_at = np.interp(tg, ts, ci)
        out["conf_at"] = ci_at * np.array([float(dfun(t)) for t in tg])
        out["cum_inf_at"] = ci_at
    return out


T3_KEYS = ("beta0", "rho", "w", "t1", "s1", "t2", "s2")
T3_LO = np.array([0.03, 0.0, 0.05, -20.0, 3.0, -20.0, 3.0])
T3_HI = np.array([0.60, 0.02, 0.95, 140.0, 60.0, 220.0, 120.0])


def t3_nll_of(ivals, k0):
    times = sorted({t for iv in ivals for t in iv[:2]})
    idx = {t: i for i, t in enumerate(times)}
    obs = np.array([iv[2] for iv in ivals], float)

    def nll(th):
        pen = 0.0
        for key, val in zip(T3_KEYS, th):
            lo, hi = m.BD.get(key, (T3_LO[T3_KEYS.index(key)], T3_HI[T3_KEYS.index(key)]))
            if val < lo:
                pen += (lo - val) ** 2 * 1e6
            elif val > hi:
                pen += (val - hi) ** 2 * 1e6
        if pen > 0:
            return 1e9 + pen
        b0, rh, w, t1, s1, t2, s2 = th
        if not (t2 > t1):                          # 排序约定，避免标签互换
            return 1e9
        try:
            s = simulate_custom(V.PRM, m.Xf0, b0, rh,
                                make_dfun2(0.75, w, t1, s1, t2, s2),
                                V.T_END, 0.5, t_grid=np.array(times, float))
        except Exception:
            return 1e9
        c = s["conf_at"]
        if not np.all(np.isfinite(c)) or np.any(np.diff(c) < -1e-9) or np.any(c <= 0):
            return 1e9
        mods = np.array([c[idx[b]] - c[idx[a]] for a, b, _ in ivals])
        if np.any(mods <= 0):
            return 1e9
        return m.nb_nll(obs, mods, k0)
    return nll


def t3_worker(arg):
    i, th0, ivals, k0, maxiter, xatol, fatol = arg
    nll = t3_nll_of(ivals, k0)
    st = tuple(min(max(v, T3_LO[j]), T3_HI[j]) for j, v in enumerate(th0))
    r = minimize(nll, st, method="Nelder-Mead",
                 options=dict(xatol=xatol, fatol=fatol, maxiter=maxiter,
                              maxfev=maxiter, adaptive=True))
    return (i, tuple(float(v) for v in r.x), float(r.fun), int(r.nit), int(r.nfev))


def t3_diag(th, ivals, k0):
    b0, rh, w, t1, s1, t2, s2 = th
    dfun = make_dfun2(0.75, w, t1, s1, t2, s2)
    times = sorted({t for iv in ivals for t in iv[:2]})
    idx = {t: i for i, t in enumerate(times)}
    obs = np.array([iv[2] for iv in ivals], float)
    s = simulate_custom(V.PRM, m.Xf0, b0, rh, dfun, V.T_END, 0.5,
                        t_grid=np.array(times, float))
    c = s["conf_at"]
    mods = np.array([c[idx[b]] - c[idx[a]] for a, b, _ in ivals])
    n = float(m.nb_nll(obs, mods, k0))
    s2 = simulate_custom(V.PRM, m.Xf0, b0, rh, dfun, V.T_END, 0.5,
                         t_grid=np.array([0., 5., 12., 152.]))
    ci = s2["cum_inf_at"]
    re0 = float(m.R0_ngm(b0, V.PRM, x=float(m.Xf0(0.0))))
    return dict(theta=[float(v) for v in th], nll=n, aic=2 * n + 2 * 7,
                re0=re0, r_early=float(np.log(ci[2] / ci[1]) / 7.0),
                cum_inf_152=float(ci[3]),
                dev=[float(100.0 * (mo - ob) / ob) for mo, ob in zip(mods, obs)],
                d_at=[float(dfun(t)) for t in DQ],
                iv=[[float(a), float(b), float(v)] for a, b, v in ivals])


def phase_t3():
    P("## T3  双分量检测函数（7 参数；d_max=0.75 固定；E1 23 区间）")
    P("   d(t)=d_max·[w·σ((t−t₁)/s₁)+(1−w)·σ((t−t₂)/s₂)]；"
      "约定 t₂>t₁（避免标签互换）")
    eng = qmc.LatinHypercube(d=7, seed=20261105)
    u = eng.random(96)
    starts = [tuple(float(T3_LO[j] + u[i, j] * (T3_HI[j] - T3_LO[j]))
                    for j in range(7)) for i in range(96)]
    # 定向：以 T1 各盆地代表为第一分量起点，第二分量错后 20–60 d
    with open("_basins_t1.json", encoding="utf-8") as f:
        t1 = json.load(f)
    for b in t1["basins"] + t1["ext_basins"]:
        th = b["theta"]
        for dt2 in (20., 40., 60.):
            starts.append((th[0], th[1], 0.7, th[2], th[3], th[2] + dt2, 15.))
            starts.append((th[0], th[1], 0.3, th[2], th[3], th[2] + dt2, 30.))
    P(f"   起点总数 {len(starts)}（LHS 96 + 定向 {len(starts)-96}）")
    args = [(i, st, IV_E1, K0, 800, 1e-6, 1e-8) for i, st in enumerate(starts)]
    t0 = time.time()
    with mp.Pool(processes=N_PROC) as pool:
        res = []
        for k, out in enumerate(pool.imap_unordered(t3_worker, args, chunksize=1), 1):
            res.append(out)
            if k % 20 == 0 or k == len(args):
                P(f"  [T3] {k}/{len(args)} 起点完成（{time.time()-t0:.0f} s）")
    res.sort(key=lambda r: r[2])
    best3 = res[0]
    r3 = t3_worker((0, best3[1], IV_E1, K0, 3000, 1e-10, 1e-12))
    d3 = t3_diag(r3[1], IV_E1, K0)
    aic_vc = 2 * 130.10933525761075 + 8
    P(f"   T3 最优：θ=(β₀={r3[1][0]:.6f}, ρ={r3[1][1]:.6f}, w={r3[1][2]:.4f}, "
      f"t₁={r3[1][3]:.3f}, s₁={r3[1][4]:.3f}, t₂={r3[1][5]:.3f}, s₂={r3[1][6]:.3f})")
    P(f"   NLL={d3['nll']:.4f}  AIC={d3['aic']:.4f}；单 logistic V-C AIC={aic_vc:.4f}；"
      f"ΔAIC={d3['aic']-aic_vc:+.4f}；ΔNLL={d3['nll']-130.10933525761075:+.4f}")
    P(f"   Rₑ₀={d3['re0']:.4f}；r_early={d3['r_early']:.4f}；"
      f"cum_inf(152)={d3['cum_inf_152']:,.0f}")
    P("   d(t)@0/12/32/60/92/148 = " + "/".join(f"{x:.4f}" for x in d3["d_at"]))
    P("   逐区间偏差%：" + " ".join(f"{x:+.1f}" for x in d3["dev"]))
    with open("_basins_t3.json", "w", encoding="utf-8") as f:
        json.dump(dict(meta=dict(k0=K0, aic_vc_single=float(aic_vc)),
                       best=d3, top5=[dict(theta=list(r[1]), nll=r[2]) for r in res[:5]]),
                  f, indent=1)
    P("")
    P("[T3 完成] _basins_t3.json 已写出")
    return d3


# ---------------- 图 ----------------
def dedup_basins(basins, tol_beta=0.02, tol_t0=3.0, tol_s=0.05):
    """合并精炼后落到同一点的聚类分裂（按 β₀ 相对 2% / |Δt₀|<3 d / s 相对 5%）。"""
    out = []
    for b in sorted(basins, key=lambda d: d["nll"]):
        th = b["theta"]
        for o in out:
            rb = o["theta"]
            if (abs(th[0] - rb[0]) <= tol_beta * max(abs(rb[0]), 1e-9)
                    and abs(th[2] - rb[2]) < tol_t0
                    and abs(th[3] - rb[3]) <= tol_s * max(abs(rb[3]), 1e-9)):
                o["n_starts"] = o.get("n_starts", 1) + b.get("n_starts", 1)
                break
        else:
            out.append(dict(b))
    return out


def phase_fig():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with open("_basins_t1.json", encoding="utf-8") as f:
        t1 = json.load(f)
    basins = dedup_basins(t1["basins"])[:4]
    P(f"[fig] 去重后独特盆地 {len(dedup_basins(t1['basins']))} 个，画前 "
      f"{len(basins)} 个：t₀ = " + ", ".join(f"{b['theta'][2]:.1f}" for b in basins))
    fig, ax = plt.subplots(1, 2, figsize=(12.6, 4.8))
    cols = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]
    # (a) 仅画两个竞争盆地（其余盆地偏差达 1e2–1e6%，线性轴下不可见；其 d(t) 见 (b)）
    for k, b in enumerate(basins[:2]):
        a = np.array([iv[0] for iv in b["iv"]], float)
        bb = np.array([iv[1] for iv in b["iv"]], float)
        mid = 0.5 * (a + bb)
        ax[0].plot(mid, b["dev"], "o-", ms=4, lw=1.4, color=cols[k],
                   label=f"basin {k+1}: t0={b['theta'][2]:.1f}, NLL={b['nll']:.2f}")
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_xlabel("interval midpoint t (days)")
    ax[0].set_ylabel("model-obs relative deviation (%)")
    ax[0].set_title("(a) Per-interval deviation, competitive basins (E1 V-C)")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3)
    for k, b in enumerate(basins):
        tt = np.linspace(0, 152, 400)
        t0d, sd = b["theta"][2], b["theta"][3]
        ax[1].plot(tt, 0.75 / (1.0 + np.exp(-(tt - t0d) / sd)), lw=1.6,
                   color=cols[k], label=f"basin {k+1}: t0={t0d:.1f}, s={sd:.1f}")
    ax[1].set_xlabel("t (days)")
    ax[1].set_ylabel("d(t) (d_max=0.75 fixed)")
    ax[1].set_title("(b) Detection ramp by basin")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig("fig_v6_basins.png", dpi=150)
    plt.close(fig)
    P("[fig] fig_v6_basins.png 已写出")


def main():
    t_all = time.time()
    ph = "t1"
    for a in sys.argv[1:]:
        if a.startswith("--phase"):
            ph = a.split("=")[-1] if "=" in a else sys.argv[sys.argv.index(a) + 1]
    P(f"# v6 全局盆地搜索（phase={ph}）")
    P(f"python={sys.version.split()[0]} numpy={np.__version__} "
      f"scipy={__import__('scipy').__version__}")
    P("> 所有数值均标注「待用户数值确认」，须由用户用自有已校验代码复算后方可采信。")
    P("")
    if ph == "t1":
        phase_t1()
    elif ph == "t2":
        phase_t2()
    elif ph == "t3":
        phase_t3()
    elif ph == "fig":
        phase_fig()
    else:
        raise SystemExit(f"unknown phase {ph}")
    P("")
    P(f"[完成] phase={ph} 总耗时 {time.time()-t_all:.1f} s")


if __name__ == "__main__":
    main()
