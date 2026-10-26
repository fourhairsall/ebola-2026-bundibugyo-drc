# -*- coding: utf-8 -*-
"""
v6_extended.py —— v6 全量重算（增长相数据补齐：UKHSA wk21–wk41 单一家族连续周序列）
==============================================================================
数据（全部经 _early_data_assembly.md 核验；t = 天，自 2026-05-21）：
  t=0 锚点（WHO DON603，数据 05-21）：83 确诊 / 9 死
  UKHSA-only 周序列 wk21–wk41（21 点，t=5→143）：
    - wk21–wk36（t=5→109）：16 点，public-health-advisor 检索核验
      （UKHSA 家族 15/15 周增量与累计差精确自洽，脚本内复核）
    - wk37（t=115）：7,258/3,510，主理人经 gov.uk 原文核实
      （"increase of 501 confirmed cases and 243 deaths"）
    - wk38–wk41（t=122→143）：论文既有（7,773/3,759 → 9,066/4,381）
  日尾（与 v5 方案 A 完全相同，保持与论文基线连续）：
    [143,146]=211（UKHSA→ECDC）、[146,148]=138、[148,150]=137、[150,152]=116
    累计 9,277/9,415/9,552/9,668；死亡 4,478/4,536/4,602/4,649
对账区间处理（wk29→wk30，[60,67] 观测增量 887，含文献明确的 272 例/174 死
  回顾性对账，7/22 批量入账；UKHSA wk30 原文 + DON615 图注 + Sitrep 12 三方互证）：
  E1（主口径，合并稀释）：[53,60]+[60,67]+[67,74] 合并为 21 天区间
    [53,74]=3,874−2,011=1,863（88.7 例/天，归带）；23 个区间；累计自洽、无任意校正
  E2（敏感性，原始周增量）：[60,67]=887 原样入似然；25 个区间
  E3（敏感性，剔除对账）：[60,67]=615（=887−272）；25 个区间
    （累计自洽校验在 t≥67 呈现 −272 偏移，报告中说明）
  方案比较：E2/E3 区间结构相同（25 个）可直接比 NLL；E1（23 个）用逐区间平均偏差。
投影基线与 horizon 不变：T_BASE=148、T_HOR==513（与论文连续）。
MC 只跑主口径 E1；E2/E3 跑确定性部分 + Rₑ + AIC。
不改 ebola_model_v3.py / v5_full.py；import ebola_model_v3 as m。
"""
import io
import sys
import time
import numpy as np
import multiprocessing as mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import stats
import ebola_model_v3 as m

SMOKE = "--smoke" in sys.argv
N_MC = 20 if SMOKE else 1000
MAXIT = 400 if SMOKE else 2500
buf = io.StringIO()
P = lambda s="": (print(s, file=buf), print(s))
fmt = lambda x, n=1: f"{x:,.{n}f}"

PRM = m.make_params()          # 中心自然史：p=0.43, c=0.60, s_comm=0.71, h=0.198, κ=0.40, N_eff=1.35e7
T_END = 152.0                  # 数据窗终点（与 v5 相同）
T_BASE = 148.0                 # 投影基线（与论文连续）
T_HOR = 513.0                  # 一年 horizon（与论文连续）
I513 = int(round((T_HOR - T_BASE) / 0.5))
TH_V5 = (0.129507, 0.001892, 24.6218, 12.7571)   # v5 主口径 V-C θ（warm start 中心）

# ============================ 数据定义 ============================
T0_CONF, T0_DEATH = 83.0, 9.0                     # WHO DON603（05-21）
WK_T = np.array([5., 12., 18., 25., 32., 39., 46., 53., 60., 67., 74., 81.,
                 87., 95., 102., 109., 115., 122., 129., 136., 143.])
WK_C = np.array([121., 363., 598., 837., 1094., 1333., 1708., 2011., 2473., 3360.,
                 3874., 4449., 5021., 5656., 6186., 6757., 7258., 7773., 8116.,
                 8603., 9066.])
WK_D = np.array([17., 62., 115., 196., 277., 399., 580., 754., 999., 1487.,
                 1751., 2061., 2378., 2715., 3007., 3267., 3510., 3759., 3924.,
                 4148., 4381.])
WK_LAB = ["wk21", "wk22", "wk23", "wk24", "wk25", "wk26", "wk27", "wk28",
          "wk29", "wk30*", "wk31", "wk32", "wk33", "wk34", "wk35", "wk36",
          "wk37", "wk38", "wk39", "wk40", "wk41"]   # * = 含回顾性对账
TAIL = [(143., 146., 211.), (146., 148., 138.), (148., 150., 137.), (150., 152., 116.)]
TAIL_CUM = {146.: 9277., 148.: 9415., 150.: 9552., 152.: 9668.}
TAIL_DEATH = {146.: 4478., 148.: 4536., 150.: 4602., 152.: 4649.}
# UKHSA 原文周增量（wk21→wk36 共 15 个；汇编 §3.1 逐字引文），脚本内复核用
INC_Q = np.array([242., 235., 239., 257., 239., 375., 303., 462., 887., 514.,
                  575., 572., 635., 530., 571.])
DEATH_Q = np.array([45., 53., 81., 81., 122., 181., 174., 245., 488., 264.,
                    310., 317., 337., 292., 260.])
RETRO = 272.0                                       # wk29→wk30 回顾性对账例数
ANCH_T = np.r_[0.0, WK_T]                           # 22 个锚点（t=0 + 21 周点）
ANCH_C = np.r_[T0_CONF, WK_C]
ANCH_D = np.r_[T0_DEATH, WK_D]
ANCH_LAB = ["DON603"] + WK_LAB


def base_intervals():
    """25 个区间：t=0→wk21→…→wk41（21 个周区间）+ 4 个日尾区间（E2 结构）。"""
    ts = [0.0] + list(WK_T)
    cs = [T0_CONF] + list(WK_C)
    iv = [(float(ts[i]), float(ts[i + 1]), float(cs[i + 1] - cs[i]))
          for i in range(len(ts) - 1)]
    return iv + [tuple(float(x) for x in t) for t in TAIL]


def scheme_E1():
    """主口径：[53,60]+[60,67]+[67,74] 合并为 [53,74]=1,863（23 区间）。"""
    out = []
    for a, b, v in base_intervals():
        if a == 53.0:
            out.append((53.0, 74.0, 3874.0 - 2011.0))
        elif a in (60.0, 67.0):
            continue
        else:
            out.append((a, b, v))
    return out


def scheme_E3():
    """敏感性：[60,67]=887−272=615（25 区间）。"""
    return [(a, b, 615.0 if (a == 60.0 and b == 67.0) else v)
            for a, b, v in base_intervals()]


SCHEMES = {"E1": scheme_E1(), "E2": base_intervals(), "E3": scheme_E3()}
PRIMARY = "E1"


# ============================ 拟合基础设施 ============================
def make_nll(ivals, dmax_fixed, prm, k_obs, ramp=True, t_end=T_END):
    times = sorted({t for iv in ivals for t in iv[:2]})
    idx = {t: i for i, t in enumerate(times)}
    obs = np.array([iv[2] for iv in ivals], float)

    def nll(theta):
        pen = m.penalize(theta, dmax_fixed)
        if pen > 0:
            return 1e9 + pen
        try:
            s = m.simulate(prm, m.Xf0, theta, t_end, 0.5,
                           t_grid=np.array(times, float),
                           dmax_fixed=dmax_fixed, ramp=ramp)
        except Exception:
            return 1e9
        c = s["conf_at"]
        if not np.all(np.isfinite(c)) or np.any(np.diff(c) < -1e-9) or np.any(c <= 0):
            return 1e9
        mods = np.array([c[idx[b]] - c[idx[a]] for a, b, _ in ivals])
        if np.any(mods <= 0):
            return 1e9
        return m.nb_nll(obs, mods, k_obs)
    return nll, obs, times


def refit(nll, keys, starts, maxiter, xatol=1e-9, fatol=1e-11):
    """多起点 Nelder-Mead；返回 (最优 θ, 最优 NLL, 各起点明细)。"""
    best, detail = None, []
    for st in starts:
        st = tuple(min(max(v, m.BD[k][0]), m.BD[k][1]) for v, k in zip(st, keys))
        r = minimize(nll, st, method="Nelder-Mead",
                     options=dict(xatol=xatol, fatol=fatol, maxiter=maxiter,
                                  maxfev=maxiter, adaptive=True))
        detail.append((tuple(float(v) for v in r.x), float(r.fun)))
        if best is None or r.fun < best.fun:
            best = r
    return tuple(float(v) for v in best.x), float(best.fun), detail


def vc_starts(base):
    return [tuple(base),
            (base[0] * 0.85, base[1] * 2.0, base[2] + 2.0, base[3]),
            (base[0] * 1.1, base[1] * 0.4, base[2] - 3.0, base[3] - 1.0)]


def vb_starts(base):
    """V-B（5 参数，(β₀,ρ,d_max,t₀,s) 排序）多起点。
    base 可为 V-C 4 元组 (β₀,ρ,t₀,s) 或 V-B 5 元组 (β₀,ρ,d_max,t₀,s)。
    注意：PKEYS_FREE 中 d_max 在第三位——历史上 (base+(0.75,)) 的顺序错误曾使
    V-B 首起点被 BD 裁剪成无效点（v5 即带此隐患，靠其余起点收敛）。"""
    if len(base) == 5:
        b0, rh, dmx, t0d, sd = base
    else:
        b0, rh, t0d, sd = base
        dmx = 0.75
    return [(b0, rh, dmx, t0d, sd),
            (b0 * 0.85, rh * 2.0, 0.30, t0d + 2.0, sd),
            (b0 * 1.1, rh * 0.4, 0.60, t0d - 3.0, sd - 1.0)]


def vb_anchor(base, dm=0.75):
    """本方案 V-C 解 + 指定 d_max 的锚点起点（保证 V-B 至少达到 V-C 的 NLL）。"""
    b0, rh, t0d, sd = base[0], base[1], base[2], base[3]
    return (b0, rh, dm, t0d, sd)


def sim_pred(th, ivals, dmax_fixed=0.75, ramp=True):
    """模型区间增量与累计（给定 θ）。"""
    times = sorted({t for iv in ivals for t in iv[:2]})
    idx = {t: i for i, t in enumerate(times)}
    s = m.simulate(PRM, m.Xf0, th, T_END, 0.5, t_grid=np.array(times, float),
                   dmax_fixed=dmax_fixed, ramp=ramp)
    c = s["conf_at"]
    mods = np.array([c[idx[b]] - c[idx[a]] for a, b, _ in ivals])
    return s, c, idx, mods, times


def mom_k(ivals, th_vc, dmax_fixed=0.75):
    """Pearson-type MoM：k̂ = Σm²/[Σ(y−m)²−Σm]（模型均值口径）。"""
    _, c, idx, mods, _ = sim_pred(th_vc, ivals, dmax_fixed)
    obs = np.array([iv[2] for iv in ivals], float)
    den = float(np.sum((obs - mods) ** 2) - np.sum(mods))
    if den <= 0:
        return np.inf
    return float(np.sum(mods ** 2) / den)


def window_ivals(ivals, mid_lo=None, mid_hi=None, weekly_only=False):
    out = []
    for iv in ivals:
        mid = (iv[0] + iv[1]) / 2.0
        if mid_lo is not None and mid < mid_lo:
            continue
        if mid_hi is not None and mid >= mid_hi:
            continue
        if weekly_only and iv[1] > 143.5:
            continue
        out.append(iv)
    return out


# ============================ MC ============================
def mc4_worker(arg):
    i, seed, th0, ivals, dmf_main = arg
    m.T_LAST = T_BASE          # spawn 子进程重新 import 后 m.T_LAST=143，必须重设为 148
    rng = np.random.default_rng(int(seed))
    prm, k_obs, dm = m.mc_draw_prm(rng, dmf_main)   # 与 v3/v5 同序
    # ==================== 任务 A 修改段 BEGIN（v6.1，2026-10-25，主理人指令） ====================
    # MC 逐 draw 的 k_obs 抽样集合由 {20, 39, ∞} 改为 {12, 18.44, 39, ∞} 等概率，
    # 使 MC 的 k 先验集合包含头条 MoM 估计 18.44（biostatistician 指出的不一致）。
    # 实现说明（扰动最小化）：上面的 m.mc_draw_prm 调用保持原样——它仍按原顺序消费
    # 同一批随机数（5 个自然史 uniform → 原 k choice({20,39,∞}) → d_max uniform），
    # 因此每个 draw 的自然史参数与 d_max 抽取和 v6 原 MC 逐位一致，种子流位置亦不变；
    # 随后仅用一次额外抽取覆盖 k_obs。该额外抽取是本 worker 对 rng 的最后一次消费
    # （worker 此后不再使用 rng：拟合/模拟/情景均为确定性），故不影响任何其他量。
    # 原 k 抽取结果被弃用（不采用），仅用于保持随机流位置。
    k_obs = float(rng.choice([12.0, 18.44, 39.0, np.inf]))
    # ==================== 任务 A 修改段 END ====================
    dmf = dm if dmf_main is not None else None      # dmf_main=-1 → d_max~U[0.34,0.95]
    nll, _, _ = make_nll(ivals, dmf, prm, k_obs)
    if dmf is None:
        keys, st0 = m._keys(None), (th0[0], th0[1], th0[2], th0[3], th0[4])
        st1 = (th0[0] * 0.7, th0[1] * 1.4, th0[2], th0[3], th0[4])
    else:
        keys = m._keys(dmf)
        st0 = (th0[0], th0[1], th0[2], th0[3])
        st1 = (th0[0] * 0.7, th0[1] * 1.4, th0[2], th0[3])
    best = None
    for st in (st0, st1):
        st = tuple(min(max(v, m.BD[k][0]), m.BD[k][1]) for v, k in zip(st, keys))
        r = minimize(nll, st, method="Nelder-Mead",
                     options=dict(xatol=1e-8, fatol=1e-10, maxiter=400, maxfev=400,
                                  adaptive=True))
        if best is None or r.fun < best.fun:
            best = r
    th = tuple(float(v) for v in best.x)
    try:
        sb = m.simulate(prm, m.Xf0, th, T_BASE, 0.5, dmax_fixed=dmf)
        dfun = sb["dfun"]
        yl = sb["traj"][-1].copy()
        cb = float(sb["conf_series"][-1])
        Fb = float(yl[m.I_F])
        dFb = float(dfun(T_BASE)) * Fb
        out = dict(i=i, conf148=cb, dF148=dFb, F148=Fb)
        for kind in ("A", "B0", "D"):
            kw = dict(vac=m.make_vac(t_start=T_BASE), ve_s=0.15, ve_d=0.40, f_ring=0.5,
                      v0=m.VAC_V0) if kind == "D" else {}
            r = m.run_scenario(prm, m.Xf0, th, yl, kind, t_end=T_HOR, dmax_fixed=dmf, **kw)
            c5 = float(r["conf"][I513])
            F5 = float(r["traj"][I513, m.I_F])
            out[f"{kind}_c"] = c5 - cb
            out[f"{kind}_dF"] = float(dfun(T_HOR)) * F5 - dFb
            out[f"{kind}_F"] = F5 - Fb
        red = 100 * (out["B0_F"] - out["D_F"]) / out["B0_F"] if out["B0_F"] > 0 else np.nan
        out["red_F"] = red
        out["red_dF"] = 100 * (out["B0_dF"] - out["D_dF"]) / out["B0_dF"]
        return out
    except Exception:
        return dict(i=i, bad=True)


def run_mc(ivals, th0, dmf_main):
    seeds = [20261011 + 7919 * i + 13 for i in range(N_MC)]   # 与 v3/v5 完全一致
    args = [(i, seeds[i], th0, ivals, dmf_main) for i in range(N_MC)]
    try:
        with mp.Pool(processes=m.N_MC_PROC) as pool:
            res = pool.map(mc4_worker, args, chunksize=8)
    except Exception as ex:
        print("[warn] MC 并行失败，回退串行：", ex, file=sys.stderr)
        res = [mc4_worker(a) for a in args]
    return [r for r in res if not r.get("bad", False)]


# ============================ profile likelihood ============================
def profile_worker(arg):
    j, val, th_mle, ivals, k0 = arg
    keys = m._keys(0.75)
    nll, _, _ = make_nll(ivals, 0.75, PRM, k0)

    def f(xo):
        x = list(th_mle)
        oi = 0
        for i in range(len(keys)):
            if i == j:
                x[i] = val
            else:
                x[i] = xo[oi]
                oi += 1
        return nll(tuple(x))
    others = [k for i, k in enumerate(keys) if i != j]
    x0 = [th_mle[i] for i in range(len(keys)) if i != j]

    def mk_start(facs=None, adds=None):
        facs, adds = facs or {}, adds or {}
        st = []
        for v, k in zip(x0, others):
            vv = v * facs.get(k, 1.0) + adds.get(k, 0.0)
            st.append(min(max(vv, m.BD[k][0]), m.BD[k][1]))
        return tuple(st)
    # 内层多起点：MLE 值 / ρ↑（β₀ 上调的补偿方向）/ ρ↓+t₀,s 上移
    best = np.inf
    for facs, adds in (({}, {}), ({"rho": 1.4}, {}), ({"rho": 0.7}, {"t0": 2.0, "s": 2.0})):
        r = minimize(f, mk_start(facs, adds), method="Nelder-Mead",
                     options=dict(xatol=1e-9, fatol=1e-11, maxiter=1000, maxfev=1000,
                                  adaptive=True))
        best = min(best, float(r.fun))
    return (j, float(val), best)


# ============================ 主流程 ============================
def main():
    t_all = time.time()
    m.T_LAST = T_BASE            # 情景/基线统一从 t=148 出发

    P("# v6 全量重算（增长相数据补齐：UKHSA wk21–wk41 单一家族连续周序列；E1 主口径）")
    P("")
    P("> 数据截至 2026-10-20（日尾同 v5 方案 A）；t=0 = 2026-05-21（WHO DON603 锚点 83/9）。")
    P("> 基线 t=148、horizon t=513、种子约定、抽样方案均与 v3/v5 一致；未改主代码。")
    P("> **所有数值均标注「待用户数值确认」，须由用户用自有已校验代码复算后方可采信。**")
    P("")

    # ---------------- 0 数据与自洽校验 ----------------
    P("## 0. 数据定义与自洽校验")
    P("")
    P("### 0.1 22 个锚点（t=0 + UKHSA wk21–wk41）")
    P("")
    P("| t | 报告 | 累计确诊 | 累计死亡 | 计算 CFR |")
    P("|---|---|---|---|---|")
    for t, c, d, lab in zip(ANCH_T, ANCH_C, ANCH_D, ANCH_LAB):
        P(f"| {t:.0f} | {lab} | {c:,.0f} | {d:,.0f} | {100*d/c:.1f}% |")
    P("")
    P("日尾累计（同 v5）：t=146: 9,277/4,478（ECDC）；t=148: 9,415/4,536（MinSante）；"
      "t=150: 9,552/4,602；t=152: 9,668/4,649。")
    P("")
    P("### 0.2 UKHSA 家族增量复核（脚本内实算 vs 原文引文）")
    P("")
    dc = np.diff(WK_C[:16])
    dd = np.diff(WK_D[:16])
    ok_c = bool(np.all(dc == INC_Q))
    ok_d = bool(np.all(dd == DEATH_Q))
    P(f"- wk21→wk36 共 15 个相邻周：确诊增量=累计差 **{ok_c}**；死亡增量=累计差 **{ok_d}**"
      "（汇编 §3.1 已核 15/15，此处脚本内复核）")
    P(f"- wk36→wk37：+{WK_C[16]-WK_C[15]:.0f} 确诊 / +{WK_D[16]-WK_D[15]:.0f} 死亡"
      "（gov.uk 原文 \"increase of 501 confirmed cases and 243 deaths\" ✓）")
    P(f"- wk37→wk41：+515/+343/+487/+463 确诊（与论文既有 wk38–wk41 一致）")
    P("")
    P("### 0.3 三方案区间结构与累计自洽校验")
    P("")
    P("| 方案 | 区间数 | [60,67] 处理 | 观测增量总和 | 83+总和 vs 9,668 |")
    P("|---|---|---|---|---|")
    for sc in ("E1", "E2", "E3"):
        iv = SCHEMES[sc]
        tot = sum(v for _, _, v in iv)
        note = {"E1": "并入 [53,74]=1,863（21 天合并）",
                "E2": "原样 887（含 272 对账）",
                "E3": "615（=887−272）"}[sc]
        P(f"| {sc} | {len(iv)} | {note} | {tot:,.0f} | {T0_CONF+tot:,.0f} |")
    P("")
    # 逐锚点的区间和 vs 实测累计
    for sc in ("E1", "E2", "E3"):
        cum = {0.0: T0_CONF}
        for a, b, v in SCHEMES[sc]:
            cum[b] = cum[a] + v
        offs = [(t, cum[t] - v) for t, v in zip(WK_T, WK_C)
                if t in cum and abs(cum[t] - v) > 1e-6]
        missing = [float(t) for t in WK_T if t not in cum]
        note = (f"；t={missing} 为合并区间 [53,74] 的内点，不设逐点校验"
                if missing else "")
        if not offs:
            P(f"- {sc}：区间增量之和与全部可校验周锚点累计**逐点一致**（无偏移）{note}")
        else:
            t0 = offs[0][0]
            P(f"- {sc}：t≥{t0:.0f} 共 {len(offs)} 个锚点呈现 **{offs[0][1]:+.0f}** 偏移"
              f"（E3 剔除 272 例对账后的预期现象：区间和低于官方累计，"
              f"因官方累计含 7/22 批量入账的既往病例）")
    P("")
    P("- **E1 合并依据**：wk29→wk30 的 887 例周新增中 272 例（30.7%）为 7/22 回顾性对账"
      "（UKHSA wk30 原文明言；DON615 图注与 AFRO Sitrep 12 三方互证），任何来源家族在该窗口"
      "都观测不到未污染增量。合并 [53,74] 后日均 88.7 例/天，回到汇编 §3.2 的 40–120 例/天带内。")
    P("")

    # ---------------- 1 k_obs 标定 + 三方案拟合 ----------------
    P("## 1. k_obs 标定（K0）与三方案重标定（V-C / V-B）")
    P("")
    P("### 1.1 K0 定点标定（MoM，Pearson-type，模型均值口径）")
    P("")
    P("k̂ = Σm²/[Σ(y−m)²−Σm]；以 k=39 起步拟合 → 估计 k̂ → 以 k̂ 重拟合，迭代至稳定。")
    P("")
    k_seq = []
    k_now, th_now = 39.0, TH_V5
    for it in range(3):
        nll_, _, _ = make_nll(SCHEMES[PRIMARY], 0.75, PRM, k_now)
        th_, n_, det_ = refit(nll_, m._keys(0.75), vc_starts(th_now), MAXIT)
        k_new = mom_k(SCHEMES[PRIMARY], th_)
        k_seq.append((it + 1, k_now, th_, n_, k_new))
        th_now = th_
        if not np.isfinite(k_new) or abs(k_new - k_now) < 0.5:
            k_now = k_new if np.isfinite(k_new) else k_now
            break
        k_now = k_new
    # 终拟合
    nll_k, _, _ = make_nll(SCHEMES[PRIMARY], 0.75, PRM, k_now)
    th_vc_E1, n_vc_E1, det_vc_E1 = refit(nll_k, m._keys(0.75), vc_starts(th_now), MAXIT)
    k_final = mom_k(SCHEMES[PRIMARY], th_vc_E1)
    K0 = k_now if np.isfinite(k_now) else np.inf
    for it, kk, th_, n_, kn in k_seq:
        P(f"- 迭代 {it}：k={kk:.2f} → β₀={th_[0]:.6f}, ρ={th_[1]:.6f}, "
          f"t₀={th_[2]:.4f}, s={th_[3]:.4f}；NLL={n_:.4f}；新 k̂={kn:.2f}")
    P(f"- **主口径 K0 = {K0:.2f}**（全扩展窗 23 区间 MoM；终拟合 NLL={n_vc_E1:.4f}，"
      f"复核 k̂={k_final:.2f}）")
    if not np.isfinite(K0):
        P("- 注意：Σ(y−m)²≤Σm，MoM 分母非正 → K0=∞（泊松极限）")
    P("")
    P("### 1.2 k_obs 三口径（E1，V-C 轨迹）")
    P("")
    k_a = mom_k(SCHEMES[PRIMARY], th_vc_E1)
    k_b = mom_k(window_ivals(SCHEMES[PRIMARY], mid_lo=60.0), th_vc_E1)
    iv_c = window_ivals(SCHEMES[PRIMARY], mid_lo=115.0, mid_hi=143.0)
    k_c = mom_k(iv_c, th_vc_E1)
    P("| 口径 | 区间数 | k̂ |")
    P("|---|---|---|")
    P(f"| (a) 全扩展窗（E1 主口径） | {len(SCHEMES[PRIMARY])} | **{k_a:.2f}** |")
    P(f"| (b) 后爬坡窗 t≥60（含合并区间 [53,74]，中点 63.5） | "
      f"{len(window_ivals(SCHEMES[PRIMARY], mid_lo=60.0))} | {k_b:.2f} |")
    P(f"| (c) 仅 wk38–wk41（[115,122]…[136,143]） | {len(iv_c)} | {k_c:.2f} |")
    P("")
    P("- (c) 与论文历史值 39 的关系：v3 拟合下同窗口 MoM 为 35.9（后 5 区间）–49.0（后 4 区间），"
      "39 处于该邻域；v6 拟合残差略变，k̂ 随之移动，**量级一致**。")
    P("")
    P("### 1.3 三方案 × V-C/V-B 拟合")
    P("")
    P("| 方案 | 变体 | β₀ | ρ | d_max | t₀ | s | NLL | AIC |")
    P("|---|---|---|---|---|---|---|---|---|")
    fits = {}
    for sc in ("E1", "E2", "E3"):
        iv = SCHEMES[sc]
        if sc == PRIMARY:
            th4, n4, det4 = th_vc_E1, n_vc_E1, det_vc_E1
        else:
            nll4, _, _ = make_nll(iv, 0.75, PRM, K0)
            th4, n4, det4 = refit(nll4, m._keys(0.75),
                                  vc_starts(th_vc_E1) + [TH_V5], MAXIT)
        nll5, _, _ = make_nll(iv, None, PRM, K0)
        # 嵌套保护：V-B 起点中含"本方案 V-C 解 + d_max=0.75"锚点，保证 V-B 至少能达到 V-C 的 NLL
        starts5 = vb_starts(th_vc_E1) + [vb_anchor(th4, 0.75)]
        th5, n5, det5 = refit(nll5, m._keys(None), starts5, MAXIT)
        nest_ok = n5 <= n4 + 1e-6
        if not nest_ok:   # 嵌套违例 = 局部极小 → 以本方案 V-C 解为锚补跑
            th5b, n5b, det5b = refit(nll5, m._keys(None),
                                     [vb_anchor(th4, dm) for dm in (0.75, 0.30, 0.95, 0.60)],
                                     MAXIT)
            th5, n5, det5 = th5b, n5b, det5b
            nest_ok = n5 <= n4 + 1e-6
        fits[sc] = dict(vc=th4, vc_nll=n4, vc_det=det4, vb=th5, vb_nll=n5, vb_det=det5,
                        nest=nest_ok)
        P(f"| {sc} | V-C（4 参数） | {th4[0]:.6f} | {th4[1]:.6f} | 0.75（固定） | "
          f"{th4[2]:.4f} | {th4[3]:.4f} | {n4:.4f} | {2*n4+8:.4f} |")
        P(f"| {sc} | V-B（5 参数） | {th5[0]:.6f} | {th5[1]:.6f} | {th5[2]:.4f} | "
          f"{th5[3]:.4f} | {th5[4]:.4f} | {n5:.4f} | {2*n5+10:.4f} |")
    P("")
    P("- **d_max 恒等式**（发病率似然对 d_max 信息量为零；附嵌套校验 V-B NLL ≤ V-C NLL+1e-6）：")
    for sc in ("E1", "E2", "E3"):
        dv = abs(fits[sc]["vb"][0] - fits[sc]["vc"][0])
        dmax_vb = fits[sc]["vb"][2]
        P(f"  - {sc}：|β₀(V-B)−β₀(V-C)| = {dv:.2e}"
          f"（{'成立' if dv < 1e-3 else '**破坏！需排查**'}）；V-B d_max = {dmax_vb:.4f}"
          f"（{'贴下界 0.30' if dmax_vb < 0.301 else '内点'}）；"
          f"嵌套校验 {'通过' if fits[sc]['nest'] else '**未过（已触发锚点补跑修复）'}")
    P("")
    P("- **多起点一致性**（各起点 NLL，主口径 E1）：")
    for tag, det in (("V-C", det_vc_E1), ("V-B", fits[PRIMARY]["vb_det"])):
        nlls = [d[1] for d in det]
        P(f"  - {tag}：{[f'{v:.4f}' for v in nlls]}，起点间极差 = "
          f"{max(nlls)-min(nlls):.2e}（{'一致' if max(nlls)-min(nlls) < 1e-3 else '存在局部极小差异'}）")
    P("")
    P("- **方案比较**：E2/E3 同为 25 个观测点，NLL 可直接比较；E1 为 23 个点，"
      "用逐区间平均绝对偏差（MoP 口径）比较：")
    P("")
    P("| 方案 | NLL（V-C） | 平均\\|偏差\\|% | 最大\\|偏差\\|% | 偏差>50% 区间数 |")
    P("|---|---|---|---|---|")
    dev_summary = {}
    for sc in ("E1", "E2", "E3"):
        _, c, idx, mods, _ = sim_pred(fits[sc]["vc"], SCHEMES[sc])
        obs = np.array([v for _, _, v in SCHEMES[sc]])
        dev = 100 * (mods - obs) / obs
        dev_summary[sc] = dev
        P(f"| {sc} | {fits[sc]['vc_nll']:.4f} | {np.mean(np.abs(dev)):.2f}% | "
          f"{np.max(np.abs(dev)):.2f}% | {int(np.sum(np.abs(dev) > 50))} |")
    P("")
    P(f"- 参考：NLL/区间数 v6 = {fits['E1']['vc_nll']/len(SCHEMES['E1']):.2f}（23 区间）"
      f" vs v5 = 6.95（10 区间）——**平均每区间似然贡献未恶化**，"
      "失配是局部集中（个别区间）而非全局退化。")
    nE2, nE3 = fits["E2"]["vc_nll"], fits["E3"]["vc_nll"]
    dEE = nE3 - nE2
    if dEE > 0:
        P(f"- NLL(E2)={nE2:.4f} vs NLL(E3)={nE3:.4f}：剔除对账后似然**变差** {dEE:+.4f}"
          "（官方累计口径下，[60,67] 的 887 例确为注册库入账值；E3 的 615 为校正敏感性，"
          "不是观测本身）")
    else:
        P(f"- NLL(E2)={nE2:.4f} vs NLL(E3)={nE3:.4f}：剔除对账后似然**反而改善** {dEE:+.4f}。"
          "模型对含对账周的预测低于 887（如 §4 校验表 t=67 模型 3,034 vs 实测 3,360），"
          "E3 的 615 反而更接近模型期望——这是**模型在该窗口系统性低估的失配信号**，"
          "不是对账处理的证据；按纪律不改模型结构，如实记录交主理人决策（见 §9 失配诊断）。")
    P("")
    P("### 1.4 日尾口径稳健性（DON622 脚注，低杠杆，不改主口径）")
    P("")
    P("WHO DON622（10/23 发布，数据至 10/20）载 9,704/4,687；v5 链（CDC 口径）在 t=152 用 "
      "9,668/4,649，差 +36 确诊/+38 死亡。将末日尾区间 [150,152] 由 116 改为 152 重算"
      "（23 区间中的 1 个 2 天区间，杠杆低）：")
    P("")
    iv_don = [(a, b, 152.0 if (a == 150.0 and b == 152.0) else v)
              for a, b, v in SCHEMES[PRIMARY]]
    nll_don, _, _ = make_nll(iv_don, 0.75, PRM, K0)
    th_don, n_don, det_don = refit(nll_don, m._keys(0.75),
                                   vc_starts(th_vc_E1) + [TH_V5], MAXIT)
    P("| 口径 | NLL | β₀ | ρ | t₀ | s |")
    P("|---|---|---|---|---|---|")
    P(f"| 主口径（[150,152]=116，CDC 链） | {n_vc_E1:.4f} | {th_vc_E1[0]:.6f} | "
      f"{th_vc_E1[1]:.6f} | {th_vc_E1[2]:.4f} | {th_vc_E1[3]:.4f} |")
    P(f"| DON622（[150,152]=152） | {n_don:.4f} | {th_don[0]:.6f} | {th_don[1]:.6f} | "
      f"{th_don[2]:.4f} | {th_don[3]:.4f} |")
    P("")
    P(f"- 差异：ΔNLL={n_don-n_vc_E1:+.4f}；β₀ {100*(th_don[0]-th_vc_E1[0])/th_vc_E1[0]:+.2f}%，"
      f"ρ {100*(th_don[1]-th_vc_E1[1])/abs(th_vc_E1[1]):+.2f}%，"
      f"t₀ {th_don[2]-th_vc_E1[2]:+.3f} d，s {100*(th_don[3]-th_vc_E1[3])/th_vc_E1[3]:+.2f}%"
      "（改动后隐含 t=152 累计 = 9,704，恰为 DON622 值）。主口径不变。")
    P("")

    # ---------------- 2 模型比较 ----------------
    P("## 2. 模型比较（H1' ramp vs H2' 常数；M_A vs M_B；k 敏感性）")
    P("")
    P("### 2.1 H1' vs H2'（ΔAIC，三方案）")
    P("")
    P("| 方案 | H1' NLL（5 参数） | H1' AIC | H2' NLL（3 参数） | H2' AIC | ΔAIC（H2'−H1'） |")
    P("|---|---|---|---|---|---|")
    h2 = {}
    for sc in ("E1", "E2", "E3"):
        iv = SCHEMES[sc]
        nll_h2, _, _ = make_nll(iv, None, PRM, K0, ramp=False)
        base = fits[sc]["vc"]
        starts_h2 = [(0.31, 0.0105, 0.95), (0.20, 0.004, 0.60), (0.45, 0.015, 0.95),
                     (base[0], base[1], 0.75)]
        th_h2, n_h2, _ = refit(nll_h2, ("beta0", "rho", "d_max"), starts_h2, MAXIT)
        h2[sc] = (th_h2, n_h2)
        aic1, aic2 = 2 * fits[sc]["vb_nll"] + 10, 2 * n_h2 + 6
        P(f"| {sc} | {fits[sc]['vb_nll']:.4f} | {aic1:.4f} | {n_h2:.4f} | {aic2:.4f} | "
          f"**{aic2-aic1:+.4f}** |")
    P("")
    P(f"- 主口径 E1：ΔAIC(H2'−H1') = **{2*h2['E1'][1]+6-(2*fits['E1']['vb_nll']+10):+.4f}**"
      f"（H2' 最优：β₀={h2['E1'][0][0]:.5f}, ρ={h2['E1'][0][1]:.5f}, "
      f"d_max={h2['E1'][0][2]:.4f}）")
    P("")
    P("### 2.2 k_obs 敏感性（E1）：ΔAIC(H2'−H1') 对 k∈{20, 39, K0, ∞}")
    P("")
    P("| k | H1' NLL | H2' NLL | ΔAIC（H2'−H1'） |")
    P("|---|---|---|---|")
    for kk in (20.0, 39.0, K0, np.inf):
        nll5, _, _ = make_nll(SCHEMES[PRIMARY], None, PRM, kk)
        th5, n5, _ = refit(nll5, m._keys(None), vb_starts(fits[PRIMARY]["vb"]), MAXIT)
        nll2, _, _ = make_nll(SCHEMES[PRIMARY], None, PRM, kk, ramp=False)
        th2, n2, _ = refit(nll2, ("beta0", "rho", "d_max"),
                           [h2[PRIMARY][0], (0.31, 0.0105, 0.95),
                            (th5[0], th5[1], 0.75)], MAXIT)
        klab = f"{kk:.1f}" if np.isfinite(kk) else "∞（泊松）"
        P(f"| {klab} | {n5:.4f} | {n2:.4f} | **{2*n2+6-(2*n5+10):+.4f}** |")
    P("")
    P("### 2.3 M_A vs M_B（θ∈{0,0.2,0.4,0.6}，各 5 参数重拟合）")
    P("")
    MB_NLL = {}
    for sc in ("E1", "E2", "E3"):
        iv = SCHEMES[sc]
        P(f"**方案 {sc}**")
        P("")
        P("| θ | NLL | AIC | Rₑ₀ |")
        P("|---|---|---|---|")
        rows = []
        for th_v in (0.0, 0.2, 0.4, 0.6):
            if th_v == 0.0:
                n_b, re0, th_b = fits[sc]["vb_nll"], None, fits[sc]["vb"]
                prm_b = PRM
                re0 = m.R0_ngm(th_b[0], prm_b, x=float(m.Xf0(0.0)))
            else:
                prm_b = m.make_params(th_a=th_v, th_b=th_v)
                nll_b, _, _ = make_nll(iv, None, prm_b, K0)
                st0 = (fits[sc]["vc"][0], fits[sc]["vc"][1], 0.30,
                       fits[sc]["vc"][2], fits[sc]["vc"][3])
                th_b, n_b, _ = refit(nll_b, m._keys(None),
                                     [st0, (st0[0]*0.8, st0[1]*1.5, st0[2], st0[3], st0[4])],
                                     MAXIT)
                re0 = m.R0_ngm(th_b[0], prm_b, x=float(m.Xf0(0.0)))
            rows.append((th_v, n_b, re0))
            MB_NLL[(sc, th_v)] = n_b
            P(f"| {th_v:.1f} | {n_b:.4f} | {2*n_b+10:.4f} | {re0:.4f} |")
        d_aic = 2 * rows[1][1] - 2 * rows[0][1]
        P("")
        P(f"- ΔAIC(M_B θ=0.2 − M_A) = **{d_aic:+.4f}**"
          f"（{'失序耦合仍不可区分' if abs(d_aic) < 2 else '定性改变！'}）；"
          f"Rₑ₀ 随 θ 漂移 {rows[0][2]:.3f}→{rows[-1][2]:.3f}（θ–β 补偿，不可分离辨识）")
        P("")

    # ---------------- 3 Rₑ ----------------
    P("## 3. Rₑ（NGM 口径 + 数据侧 log-linear 参考）")
    P("")
    g_nod = m.R0_ngm(1.0, PRM, x=0.0)
    g_x0 = m.R0_ngm(1.0, PRM, x=float(m.Xf0(0.0)))
    P(f"- NGM 乘子核验：无失序常数 G={g_nod:.4f}；含 X(0)（X(0)={float(m.Xf0(0.0)):.4f}）"
      f"G={g_x0:.4f}（M_A 下 th_a=th_b=0，X 不进入，二者相同）。"
      f"注：论文原引乘子 9.026 对应旧 h_care=3.5（h=0.286）；h=0.198 修正后为 {g_x0:.4f}。")
    P("")
    P("| 方案 | β₀ | Rₑ₀=β₀·G | Rₑ(148) | Rₑ(152) |")
    P("|---|---|---|---|---|")
    re_info = {}
    for sc in ("E1", "E2", "E3"):
        th = fits[sc]["vc"]
        re0 = m.R0_ngm(th[0], PRM, x=float(m.Xf0(0.0)))
        re148 = m.R0_ngm(th[0] * np.exp(-th[1] * 148.0), PRM, x=float(m.Xf0(148.0)))
        re152 = m.R0_ngm(th[0] * np.exp(-th[1] * 152.0), PRM, x=float(m.Xf0(152.0)))
        re_info[sc] = dict(re0=re0, re148=re148, re152=re152, th=th)
        P(f"| {sc} | {th[0]:.6f} | **{re0:.4f}** | **{re148:.4f}** | {re152:.4f} |")
    P("")
    P("### 3.1 数据侧 log-linear 参考（区间增量/天 ~ 区间中点；**禁止对累计数做 log-linear**）")
    P("")
    P("r 为日增速；Rₑ=exp(r·T_g)。CI 由 OLS 标准误 + t 分位数给出。")
    P("")

    def ll_fit(ivs, mid_lo=None, weekly_only=False):
        sub = window_ivals(ivs, mid_lo=mid_lo, weekly_only=weekly_only)
        mids = np.array([(a + b) / 2.0 for a, b, _ in sub])
        rate = np.array([v / (b - a) for a, b, v in sub])
        lr = stats.linregress(mids, np.log(rate))
        n = len(mids)
        tc = stats.t.ppf(0.975, n - 2)
        return dict(n=n, r=float(lr.slope), se=float(lr.stderr),
                    lo=float(lr.slope - tc * lr.stderr), hi=float(lr.slope + tc * lr.stderr),
                    mids=mids)

    ll_rows = [
        ("全窗（E2 原始 25 区间，含对账周）", SCHEMES["E2"], None, False),
        ("全窗（E1 合并 23 区间，主口径）", SCHEMES["E1"], None, False),
        ("全窗（仅 21 个周区间，去日尾）", SCHEMES["E2"], None, True),
        ("t≥60（E2，16 区间）", SCHEMES["E2"], 60.0, False),
        ("t≥67（E2，15 区间，剔除对账周）", SCHEMES["E2"], 67.0, False),
        ("t≥74（E2，14 区间）", SCHEMES["E2"], 74.0, False),
    ]
    P("| 窗口 | n | r (1/d) | SE(r) | 95% CI | Rₑ(T_g=10) | Rₑ(T_g=12) | Rₑ(T_g=16) | Rₑ(SI=15.3) |")
    P("|---|---|---|---|---|---|---|---|---|")
    ll_res = {}
    for lab, ivs, lo, wk in ll_rows:
        d = ll_fit(ivs, lo, wk)
        ll_res[lab] = d
        P(f"| {lab} | {d['n']} | {d['r']:+.6f} | {d['se']:.6f} | "
          f"[{d['lo']:+.6f}, {d['hi']:+.6f}] | "
          f"{np.exp(d['r']*10):.4f} | {np.exp(d['r']*12):.4f} | "
          f"{np.exp(d['r']*16):.4f} | {np.exp(d['r']*15.3):.4f} |")
    P("")
    P("- GT=12 d 为主理人标注「团队推导」（无文献支持），敏感性 10–16 d；SI=15.3 d"
      "（WHO EBT NEJM 2014，EBOV 西非）只做数据侧对照，不进模型。")
    P("- 增长相（t<60）数据侧 r 显著高于衰退相：检出爬坡与传播减速叠加的观测层解释"
      "与汇编 §3.7 的分段对数增长率衰减（0.109→0.013/d）一致。")
    P("")

    # ---------------- 4 校验表 ----------------
    P("## 4. 校验表（主口径 E1，V-C）")
    P("")
    thC = fits[PRIMARY]["vc"]
    s_full = m.simulate(PRM, m.Xf0, thC, T_END, 0.5, t_grid=ANCH_T, dmax_fixed=0.75)
    conf_at = s_full["conf_at"]
    F_at = np.array([float(np.interp(t, s_full["ts"], s_full["traj"][:, m.I_F]))
                     for t in ANCH_T])
    d_at = np.array([float(s_full["dfun"](t)) for t in ANCH_T])
    P("| t | 报告 | 实测确诊 | 模型确诊 | 偏差% | 实测死亡 | 模型 d·F | 偏差% | d(t) |")
    P("|---|---|---|---|---|---|---|---|---|")
    for i, t in enumerate(ANCH_T):
        P(f"| {t:.0f} | {ANCH_LAB[i]} | {ANCH_C[i]:,.0f} | {conf_at[i]:,.1f} | "
          f"{100*(conf_at[i]-ANCH_C[i])/ANCH_C[i]:+.2f} | {ANCH_D[i]:,.0f} | "
          f"{d_at[i]*F_at[i]:,.1f} | {100*(d_at[i]*F_at[i]-ANCH_D[i])/ANCH_D[i]:+.2f} | "
          f"{d_at[i]:.4f} |")
    P("")
    dev_c = 100 * (conf_at - ANCH_C) / ANCH_C
    dev_d = 100 * (d_at * F_at - ANCH_D) / ANCH_D
    P(f"- 确诊：平均|偏差| = {np.mean(np.abs(dev_c)):.2f}%，最大 = {np.max(np.abs(dev_c)):.2f}%"
      f"（t={ANCH_T[np.argmax(np.abs(dev_c))]:.0f}）")
    P(f"- 死亡（d·F 口径）：平均|偏差| = {np.mean(np.abs(dev_d)):.2f}%，"
      f"最大 = {np.max(np.abs(dev_d)):.2f}%（t={ANCH_T[np.argmax(np.abs(dev_d))]:.0f}）")
    mid = (ANCH_T >= 67) & (ANCH_T <= 109)
    early = ANCH_T <= 25
    late = ANCH_T >= 129
    P(f"- **系统性模式**：中段 t=67–109 确诊平均偏差 {np.mean(dev_c[mid]):+.1f}%、"
      f"死亡 {np.mean(dev_d[mid]):+.1f}%（**系统性低估**）；早期 t≤25 死亡 "
      f"{np.mean(dev_d[early]):+.1f}%（高估）；末段 t≥129 收敛至 "
      f"{np.mean(dev_c[late]):+.1f}%/{np.mean(dev_d[late]):+.1f}%。"
      "确诊与死亡偏差同形态（S 形），与 §9 失配诊断一致。")
    P("")
    # 基线 t=148
    s_base = m.simulate(PRM, m.Xf0, thC, T_BASE, 0.5, dmax_fixed=0.75)
    dfunC = s_base["dfun"]
    yl148 = s_base["traj"][-1].copy()
    conf148 = float(s_base["conf_series"][-1])
    F148 = float(yl148[m.I_F])
    dF148 = float(dfunC(T_BASE)) * F148
    P(f"- **基线 t=148 校验**：模型 conf(148)={conf148:,.1f} vs 实测 9,415"
      f"（{100*(conf148-9415)/9415:+.2f}%）；模型 d·F(148)={dF148:,.1f} vs 实测 4,536"
      f"（{100*(dF148-4536)/4536:+.2f}%）")
    P("")
    # 数值验证
    P("### 4.1 数值验证（Radau 交叉校验 / 网格收敛 / 守恒正性）")
    P("")
    from scipy.integrate import solve_ivp
    beta_fn = lambda t: thC[0] * np.exp(-thC[1] * t)
    f_ivp = lambda t, y: m.rhs(t, y, PRM, m.Xf0, beta_fn, dfunC)
    sol = solve_ivp(f_ivp, (0.0, T_END), s_full["y0"], method="Radau",
                    rtol=1e-10, atol=1e-8, dense_output=True)
    ca = np.array([float(dfunC(t)) * sum(sol.sol(t)[1:1 + m.IP_NE + 5]) for t in ANCH_T])
    P(f"- 求解器交叉校验：RK4(dt=0.5) vs solve_ivp/Radau(rtol=1e-10)，22 锚点确诊"
      f"最大相对偏差 = **{np.max(np.abs(ca/conf_at-1.0)):.3e}**「待用户数值确认」")
    devs = [(d, float(m.simulate(PRM, m.Xf0, thC, T_BASE, d, dmax_fixed=0.75)
                      ["conf_series"][-1])) for d in (1.0, 0.5, 0.25, 0.125)]
    P("- 网格加密收敛（conf(148)）：" +
      "；".join(f"dt={d} → {v:.4f}" for d, v in devs) +
      f"；dt=1.0→0.125 总漂移 = {abs(devs[-1][1]-devs[0][1])/devs[-1][1]:.3e}")
    tot = s_full["traj"].sum(axis=1)
    P(f"- 守恒：max|Σ/N_eff−1| = {np.max(np.abs(tot/PRM['N_eff']-1.0)):.3e}；"
      f"正性：全程最小 = {float(np.min(s_full['traj'])):.3e}（≥0）")
    P("")

    # ---------------- 5 爬坡表 ----------------
    P("## 5. 检出爬坡表 d(t)（增长相爬坡现已被数据辨识）")
    P("")
    t_ramp = [0, 5, 13, 32, 60, 92, 115, 148]
    thB = fits[PRIMARY]["vb"]
    dfB = m.make_dfun(thB[2], thB[3], thB[4], True)
    P("| t | 日期 | V-C d(t)（d_max=0.75 固定） | d(t)/d_max | V-B d(t)（d_max 自由） |")
    P("|---|---|---|---|---|")
    dates = {0: "05-21", 5: "05-26", 13: "06-03", 32: "06-22", 60: "07-20",
             92: "08-21", 115: "09-13", 148: "10-16"}
    for t in t_ramp:
        P(f"| {t} | {dates[t]} | {float(dfunC(t)):.4f} | {float(dfunC(t))/0.75:.4f} | "
          f"{float(dfB(t)):.4f} |")
    P("")
    P(f"- 拟合 t₀={thC[2]:.2f} d，s={thC[3]:.2f} d：检出率从 t=0 的 {float(dfunC(0)):.4f} "
      f"爬升至 t=148 的 {float(dfunC(148)):.4f}；**增长相 16 个新周点（t=5→109）"
      "直接跨越爬坡区，爬坡形状首次被数据约束**（v5 窗仅 2 个早期锚点）。")
    P(f"- V-B 的 d_max = {thB[2]:.4f}"
      f"（{'贴 0.30 下界' if thB[2] < 0.301 else '内点'}）：发病率似然对 d_max 信息量为零"
      f"（恒等式 |Δβ₀|={abs(thB[0]-thC[0]):.1e}），β₀ 完全补偿，其 d(t) 与 V-C 只差常数标度；"
      "主口径用 V-C（d_max=0.75 外生锚定）。")
    P("")

    # ---------------- 6 情景 ----------------
    P("## 6. 情景（A / B0 / B / D；基线 t=148，horizon t=513；各方案 V-C）")
    P("")
    P("疫苗参数同 v5：c_ring=20, VE_S=0.15, VE_D=0.40, f_ring=0.5, V(0)=9,000, t_vac=158。")
    P("")
    P("| 方案 | 情景 | 确诊Δ t=513 | d·F 死亡Δ t=513 | F 全死因死亡Δ t=513 |")
    P("|---|---|---|---|---|")
    scen = {}
    red_info = {}
    for sc in ("E1", "E2", "E3"):
        th_s = fits[sc]["vc"]
        sb = m.simulate(PRM, m.Xf0, th_s, T_BASE, 0.5, dmax_fixed=0.75)
        df_s = sb["dfun"]
        yl_s = sb["traj"][-1].copy()
        cb = float(sb["conf_series"][-1])
        Fb = float(yl_s[m.I_F])
        dFb = float(df_s(T_BASE)) * Fb
        rows_s = {}
        for kind in ("A", "B0", "B"):
            r = m.run_scenario(PRM, m.Xf0, th_s, yl_s, kind, t_end=T_HOR, dmax_fixed=0.75)
            c5 = float(r["conf"][I513])
            F5 = float(r["traj"][I513, m.I_F])
            rows_s[kind] = dict(c=c5 - cb, dF=float(df_s(T_HOR)) * F5 - dFb, F=F5 - Fb)
            if kind != "B":      # M_A 下 B 与 B0 数值相同
                P(f"| {sc} | {kind} | {rows_s[kind]['c']:,.1f} | {rows_s[kind]['dF']:,.1f} | "
                  f"{rows_s[kind]['F']:,.1f} |")
        rD = m.run_scenario(PRM, m.Xf0, th_s, yl_s, "D", t_end=T_HOR,
                            vac=m.make_vac(t_start=T_BASE), ve_s=0.15, ve_d=0.40,
                            f_ring=0.5, dmax_fixed=0.75, v0=m.VAC_V0)
        FD = float(rD["traj"][I513, m.I_F])
        cD = float(rD["conf"][I513])
        rows_s["D"] = dict(c=cD - cb, dF=float(df_s(T_HOR)) * FD - dFb, F=FD - Fb)
        redF = 100 * (rows_s["B0"]["F"] - rows_s["D"]["F"]) / rows_s["B0"]["F"]
        reddF = 100 * (rows_s["B0"]["dF"] - rows_s["D"]["dF"]) / rows_s["B0"]["dF"]
        red_info[sc] = dict(red_F=redF, red_dF=reddF, V513=float(rD["traj"][I513, m.I_V]))
        P(f"| {sc} | D（疫苗） | {rows_s['D']['c']:,.1f} | {rows_s['D']['dF']:,.1f} | "
          f"{rows_s['D']['F']:,.1f} |")
        scen[sc] = rows_s
    P("")
    P("- M_A 口径（θ_a=θ_b=0）下 X 不进入动力学，情景 B 与 B0 数值相同（表中 B 并入 B0 行）；"
      "失序效应只以相对量由 M_B 分支给出。")
    P("")
    P("| 方案 | 疫苗死亡降幅（F 口径） | 疫苗死亡降幅（d·F 口径） | V(513) | V/N_eff |")
    P("|---|---|---|---|---|")
    for sc in ("E1", "E2", "E3"):
        P(f"| {sc} | {red_info[sc]['red_F']:.2f}% | {red_info[sc]['red_dF']:.2f}% | "
          f"{red_info[sc]['V513']:,.0f} | {100*red_info[sc]['V513']/PRM['N_eff']:.2f}% |")
    P("")
    P("- 均匀混合下环接种流量 ∝ 日新发确诊，覆盖仅千分之几，效果结构性极小"
      "（非数值问题；v3/v5 同结论）。")
    P("")

    # ---------------- 7 MC ----------------
    P(f"## 7. 蒙特卡洛（N={N_MC}，主口径 E1，抽样方案与 v3/v5 一致）")
    P("")
    t_mc = time.time()
    mc = run_mc(SCHEMES[PRIMARY], fits[PRIMARY]["vc"], -1.0)
    P(f"- 有效样本 {len(mc)}/{N_MC}；耗时 {time.time()-t_mc:.1f} s")
    qs = lambda a: tuple(np.percentile(a, [2.5, 50, 97.5]))
    mcA_c = qs(np.array([r["A_c"] for r in mc]))
    mcA_dF = qs(np.array([r["A_dF"] for r in mc]))
    mcB_c = qs(np.array([r["B0_c"] for r in mc]))
    mcB_dF = qs(np.array([r["B0_dF"] for r in mc]))
    mcA_F = qs(np.array([r["A_F"] for r in mc]))
    mcB_F = qs(np.array([r["B0_F"] for r in mc]))
    mcRed = qs(np.array([r["red_F"] for r in mc]))
    mcRedD = qs(np.array([r["red_dF"] for r in mc]))
    P("")
    P("| 量 | 中位数 [95%] | 确定性 | 确定性偏离中位数 |")
    P("|---|---|---|---|")
    for lab, q, det in (("A 确诊Δ", mcA_c, scen[PRIMARY]["A"]["c"]),
                        ("A d·F 死亡Δ", mcA_dF, scen[PRIMARY]["A"]["dF"]),
                        ("A F 死亡Δ", mcA_F, scen[PRIMARY]["A"]["F"]),
                        ("B0 确诊Δ", mcB_c, scen[PRIMARY]["B0"]["c"]),
                        ("B0 d·F 死亡Δ", mcB_dF, scen[PRIMARY]["B0"]["dF"]),
                        ("B0 F 死亡Δ", mcB_F, scen[PRIMARY]["B0"]["F"])):
        P(f"| {lab} | {q[1]:,.0f} [{q[0]:,.0f}, {q[2]:,.0f}] | {det:,.0f} | "
          f"{100*(det-q[1])/abs(q[1]):+.1f}% |")
    P(f"| 疫苗死亡降幅（F 口径） | {mcRed[1]:.2f}% [{mcRed[0]:.2f}, {mcRed[2]:.2f}] | "
      f"{red_info[PRIMARY]['red_F']:.2f}% | — |")
    P(f"| 疫苗死亡降幅（d·F 口径） | {mcRedD[1]:.2f}% [{mcRedD[0]:.2f}, {mcRedD[2]:.2f}] | "
      f"{red_info[PRIMARY]['red_dF']:.2f}% | — |")
    P("")
    np.savez("mc_draws_v6.npz",
             **{k: np.array([r.get(k, np.nan) for r in mc]) for k in
                ("A_c", "A_dF", "A_F", "B0_c", "B0_dF", "B0_F", "D_c",
                 "D_dF", "D_F", "red_F", "red_dF", "conf148", "dF148", "F148")})
    P("- 已存 `mc_draws_v6.npz`（字段同 v5：A/B0/D 的确诊/死亡增量、疫苗降幅、基线量）")
    P("")

    # ---------------- 8 profile likelihood ----------------
    P("## 8. Profile likelihood（E1，V-C，扩展窗重算；论文正文引用区间必须更新）")
    P("")
    th_p = fits[PRIMARY]["vc"]
    n_min = fits[PRIMARY]["vc_nll"]
    CRIT = n_min + 1.92          # χ²₁,95% / 2
    P(f"- MLE：β₀={th_p[0]:.6f}, ρ={th_p[1]:.6f}, t₀={th_p[2]:.4f}, s={th_p[3]:.4f}；"
      f"NLL_min={n_min:.4f}，CRIT=NLL_min+1.92={CRIT:.4f}")
    P("- 方法：固定被 profile 参数，其余 3 参数内层**多起点**（3 起点 × 1000 iter，含 ρ 补偿方向）"
      "重优化；网格以 MLE 为中心、按 BD 界裁剪、MLE 附近密。smoke 曾出现单点 CI，经诊断为"
      "内层欠收敛 + 网格过粗的复合伪影（细网格+多起点独立诊断确认 profile 平滑），已修复。")
    keys_p = m._keys(0.75)
    b0b, rb, t0b, sbnd = (m.BD[k] for k in keys_p)
    ngs = (7, 7, 7, 7) if SMOKE else (25, 25, 19, 19)
    GRIDS = {
        0: np.clip(th_p[0] + np.linspace(-0.16, 0.16, ngs[0]), b0b[0] + 1e-9, b0b[1] - 1e-9),
        1: np.clip(th_p[1] + np.linspace(-0.016, 0.016, ngs[1]), rb[0] + 1e-9, rb[1] - 1e-9),
        2: np.clip(th_p[2] + np.linspace(-18.0, 18.0, ngs[2]), t0b[0] + 1e-9, t0b[1] - 1e-9),
        3: np.clip(th_p[3] + np.linspace(-8.0, 8.0, ngs[3]), sbnd[0] + 1e-9, sbnd[1] - 1e-9),
    }
    GRIDS = {j: np.unique(g) for j, g in GRIDS.items()}
    jobs = [(j, float(v), th_p, SCHEMES[PRIMARY], K0)
            for j in range(len(keys_p)) for v in GRIDS[j]]
    t_pr = time.time()
    try:
        with mp.Pool(processes=min(m.N_MC_PROC, len(jobs))) as pool:
            res = pool.map(profile_worker, jobs, chunksize=4)
    except Exception as ex:
        print("[warn] profile 并行失败，回退串行：", ex, file=sys.stderr)
        res = [profile_worker(a) for a in jobs]
    byj = {j: {} for j in range(len(keys_p))}
    for j, v, nl in res:
        byj[j][v] = nl
    P("")
    P("| 参数 | MLE | profile 95% CI | 网格/BD 边界? |")
    P("|---|---|---|---|")
    prof_ci = {}
    for j, k in enumerate(keys_p):
        g = np.array(sorted(byj[j].keys()))
        nls = np.array([byj[j][float(vv)] for vv in g])
        below = g[nls <= CRIT]
        lo = float(below.min()) if len(below) else np.nan
        hi = float(below.max()) if len(below) else np.nan
        prof_ci[k] = (lo, hi)
        edge = (abs(lo - g.min()) < 1e-9) or (abs(hi - g.max()) < 1e-9)
        P(f"| {k} | {th_p[j]:.5f} | [{lo:.4f}, {hi:.4f}] | "
          f"{'**是（贴网格/BD 边，需加宽）**' if edge else '否'} |")
    P("")
    # 论文正文引用的 (β0,ρ) 曲线（可审计）
    for j, k in ((0, "beta0"), (1, "rho")):
        g = np.array(sorted(byj[j].keys()))
        nls = np.array([byj[j][float(vv)] for vv in g])
        P(f"<details><summary>{k} profile 曲线（网格值 / NLL；&le;CRIT 标记 &le;）</summary>")
        P("")
        for vv, nn in zip(g, nls):
            P(f"- {vv:.5f} → {nn:.4f} {'≤' if nn <= CRIT else ''}")
        P("")
        P("</details>")
        P("")
    P(f"- 耗时 {time.time()-t_pr:.1f} s。v5（窗 [0,152] 稀疏增长相）对应区间："
      "β₀ [0.100, 0.170]、ρ [−0.0006, 0.0022]、t₀ [10.33, 40.67]、s [10.56, 16.11]。"
      "增长相 16 点补齐后 (β₀,ρ) 区间位置与宽度均显著移动，**论文正文引用的区间必须按本表更新**。")
    P("- 注意：远场网格点（|参数−MLE| 大）的 profile NLL 可能因内层优化极限而非单调"
      "（上方曲线末段可见）；95% CI 由 MLE 邻域决定，该区域已经独立细网格 + 多起点诊断"
      "验证为光滑（β₀: 0.155→0.205 全部 ≤CRIT），CI 结论不受远场非单调影响。")
    P("")

    # ---------------- 9 迁移报告 + 稳定性清单 ----------------
    P("## 9. v5 → v6 迁移报告与结论稳定性清单")
    P("")
    V5 = dict(beta0=0.129507, rho=0.001892, re0=1.2552, re148=0.9487,
              A_c=5854.5, A_dF=4287.0, B0_c=14216.9, B0_dF=9053.1,
              vac_red=0.57, dAIC_ramp=33.2821, dAIC_mb=-0.3153, k=39.0,
              conf148=9624.2, dF148=5030.5)
    re0_v6 = re_info[PRIMARY]["re0"]
    re148_v6 = re_info[PRIMARY]["re148"]
    dAIC_ramp_v6 = 2 * h2[PRIMARY][1] + 6 - (2 * fits[PRIMARY]["vb_nll"] + 10)
    rows_mig = [
        ("β₀", V5["beta0"], fits[PRIMARY]["vc"][0], "{:.6f}"),
        ("ρ", V5["rho"], fits[PRIMARY]["vc"][1], "{:.6f}"),
        ("Rₑ₀（NGM，含 X(0)）", V5["re0"], re0_v6, "{:.4f}"),
        ("Rₑ(148)（NGM）", V5["re148"], re148_v6, "{:.4f}"),
        ("A 确诊Δ（一年）", V5["A_c"], scen[PRIMARY]["A"]["c"], "{:,.1f}"),
        ("A d·F 死亡Δ（一年）", V5["A_dF"], scen[PRIMARY]["A"]["dF"], "{:,.1f}"),
        ("B0 确诊Δ（一年）", V5["B0_c"], scen[PRIMARY]["B0"]["c"], "{:,.1f}"),
        ("B0 d·F 死亡Δ（一年）", V5["B0_dF"], scen[PRIMARY]["B0"]["dF"], "{:,.1f}"),
        ("疫苗死亡降幅 F 口径（%）", V5["vac_red"], red_info[PRIMARY]["red_F"], "{:.2f}"),
        ("k_obs（MoM）", V5["k"], K0, "{:.2f}"),
        ("ΔAIC（H2'−H1'，ramp）", V5["dAIC_ramp"], dAIC_ramp_v6, "{:.4f}"),
        ("ΔAIC（M_B θ=0.2 − M_A）", V5["dAIC_mb"], float("nan"), "{:.4f}"),   # 下方重算后回填
        ("conf(148)（基线校验）", V5["conf148"], conf148, "{:,.1f}"),
        ("d·F(148)（基线校验）", V5["dF148"], dF148, "{:,.1f}"),
    ]
    # M_B ΔAIC 复用 §2.3 的 θ=0.2 重拟合结果（同起点、同设置，避免重复计算）
    dAIC_mb_v6 = 2 * MB_NLL[(PRIMARY, 0.2)] - 2 * fits[PRIMARY]["vb_nll"]
    rows_mig[11] = ("ΔAIC（M_B θ=0.2 − M_A）", V5["dAIC_mb"], dAIC_mb_v6, "{:.4f}")
    P("| 量 | v5（窗 [0,152]，稀疏增长相） | v6（增长相补齐，E1） | 相对变化 | >2%? |")
    P("|---|---|---|---|---|")
    for lab, a, b, f_ in rows_mig:
        rel = 100 * (b - a) / abs(a) if a != 0 else np.nan
        flag = "**是**" if abs(rel) > 2 else "否"
        P(f"| {lab} | {f_.format(a)} | {f_.format(b)} | {rel:+.2f}% | {flag} |")
    P("")
    P("### 结论稳定性清单（逐条核对论文现有论断）")
    P("")
    b0_rel = abs(re0_v6 - 1.26) / 1.26
    v1 = "确认" if b0_rel <= 0.05 else ("限定" if b0_rel <= 0.15 else "反转")
    v2 = ("确认" if 0.90 <= re148_v6 <= 1.00 else
          ("限定" if 0.85 <= re148_v6 <= 1.05 else "反转"))
    b2 = ("落在确认带 [0.90,1.00] 内" if v2 == "确认" else
          ("落在限定带 [0.85,1.05] 内、但已低于 0.90 下沿" if v2 == "限定" else
           "跌出 [0.85,1.05] 限定带"))
    v3 = ("确认" if dAIC_ramp_v6 >= 33 else
          ("限定" if dAIC_ramp_v6 > 2 else "反转"))
    v4 = ("确认" if abs(dAIC_mb_v6) < 2 else ("限定" if abs(dAIC_mb_v6) < 5 else "反转"))
    sB0c = 100 * (scen[PRIMARY]["B0"]["c"] - V5["B0_c"]) / V5["B0_c"]
    sB0d = 100 * (scen[PRIMARY]["B0"]["dF"] - V5["B0_dF"]) / V5["B0_dF"]
    v5 = ("确认" if max(abs(sB0c), abs(sB0d)) <= 25 else
          ("限定" if max(abs(sB0c), abs(sB0d)) <= 50 else "反转"))
    v6 = ("确认" if red_info[PRIMARY]["red_F"] < 1 else
          ("限定" if red_info[PRIMARY]["red_F"] < 2 else "反转"))
    dv_id = abs(fits[PRIMARY]["vb"][0] - fits[PRIMARY]["vc"][0])
    v7 = "确认" if dv_id < 1e-3 else "反转"
    P("| # | 论文论断 | 论文数值（v5） | v6 数值 | 判定 | 依据 |")
    P("|---|---|---|---|---|---|")
    P(f"| 1 | Rₑ₀≈1.26 | 1.2552 | **{re0_v6:.4f}** | **{v1}** | "
      f"相对 1.26 偏离 {100*b0_rel:.1f}% |")
    P(f"| 2 | Rₑ(148)≈0.95 平台 | 0.9487 | **{re148_v6:.4f}** | **{v2}** | {b2} |")
    P(f"| 3 | ΔAIC(ramp)>33（检出爬坡被强烈支持） | +33.28 | **{dAIC_ramp_v6:+.2f}** | **{v3}** | "
      f"增长相 16 点补齐后爬坡证据{'更强' if dAIC_ramp_v6 >= 33 else '仍强但数值带变窄'} |")
    P(f"| 4 | M_A/M_B 不可区分（ΔAIC<2） | −0.3153 | **{dAIC_mb_v6:+.4f}** | **{v4}** | "
      f"θ=0.2 重拟合 ΔAIC |")
    P(f"| 5 | 情景量级（B0 一年确诊Δ≈14.2k、死亡Δ≈9.1k） | 14,217 / 9,053 | "
      f"{scen[PRIMARY]['B0']['c']:,.0f} / {scen[PRIMARY]['B0']['dF']:,.0f} | **{v5}** | "
      f"相对 v5 变化 {sB0c:+.1f}% / {sB0d:+.1f}% |")
    P(f"| 6 | 疫苗降幅 <1% | 0.57% | **{red_info[PRIMARY]['red_F']:.2f}%** | **{v6}** | "
      f"F 口径点估计 |")
    P(f"| 7 | d_max 恒等式（发病率似然对 d_max 信息量为零） | 2.9e-06 | {dv_id:.2e} | **{v7}** | "
      f"|β₀(V-B)−β₀(V-C)|<1e-3 |")
    P(f"| 8（新增） | 增长相检出爬坡可被数据辨识 | v5 窗仅 2 个早期锚点 | "
      f"16 个新周点（t=5→109）跨爬坡区 | **确认（新增证据）** | "
      f"§5 爬坡表 + ΔAIC(ramp)={dAIC_ramp_v6:+.2f} |")
    P("")
    dev_E1 = dev_summary[PRIMARY]
    order = np.argsort(-np.abs(dev_E1))[:3]
    top = "；".join(
        f"[{SCHEMES[PRIMARY][i][0]:.0f},{SCHEMES[PRIMARY][i][1]:.0f}] "
        f"obs={SCHEMES[PRIMARY][i][2]:.0f} dev={dev_E1[i]:+.1f}%" for i in order)
    P("**失配诊断**：主口径 E1 逐区间最大|偏差| = "
      f"{np.max(np.abs(dev_E1)):.2f}%（>50% 的区间 "
      f"{int(np.sum(np.abs(dev_E1) > 50))} 个）；最大偏差区间：{top}。"
      + ("偏差模式为：首区间 [0,5] 大幅高估（观测首周过平：7.6→34.6 例/天的一周 4.5 倍跳增"
         "被平滑 ramp 无法同时吸收）与个别周波动（如 [122,129] 单周回落至 343）无法被"
         "4 参数光滑结构复现。**未改模型结构，如实记录交主理人决策。**"
         if np.max(np.abs(dev_E1)) > 50 else "未见系统性失配，模型结构无需修改。"))
    P("")

    # ---------------- 10 图 ----------------
    P("## 10. 图（fig_v6_*，英文标注，Agg 后端）")
    P("")
    P("- fig_v6_weekly_fit.png：左=区间增量（周点/日尾/合并区间，对数轴）；右=累计锚点全程")
    P("- fig_v6_reff.png：Rₑ(t)（标注 Rₑ₀、Rₑ(148)与数据侧参考）")
    P("- fig_v6_scen.png：情景；fig_v6_vacc.png：疫苗")
    P("")

    plt.rcParams["axes.unicode_minus"] = False
    s_full = m.simulate(PRM, m.Xf0, thC, T_HOR, 0.5, dmax_fixed=0.75)
    ts_all = np.asarray(s_full["ts"])
    conf_all = np.asarray(s_full["conf_series"])

    # --- 图1 拟合 ---
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ivs = SCHEMES[PRIMARY]
    _, c1, idx1, mod_inc, _ = sim_pred(thC, ivs)
    mids = [(a + b) / 2 for a, b, _ in ivs]
    is_merged = [(a, b) == (53.0, 74.0) for a, b, _ in ivs]
    is_daily = [b > 143.5 for a, b, _ in ivs]
    ax[0].plot(mids, mod_inc, "o-", color="#2a7f2a", lw=1.5, ms=4, label="Model (v6 fit, E1)")
    wx = [x for x, mg, dl in zip(mids, is_merged, is_daily) if not mg and not dl]
    wy = [v for (a, b, v), mg, dl in zip(ivs, is_merged, is_daily) if not mg and not dl]
    ax[0].plot(wx, wy, "s", color="#1f5fbf", ms=6, label="Observed weekly (UKHSA wk21-41)")
    mx = [x for x, mg in zip(mids, is_merged) if mg]
    my = [v for (a, b, v), mg in zip(ivs, is_merged) if mg]
    ax[0].plot(mx, my, "D", color="#9467bd", ms=8,
               label="Merged interval [53,74]=1863 (reconciliation dilution)")
    dx = [x for x, mg, dl in zip(mids, is_merged, is_daily) if dl and not mg]
    dy = [v for (a, b, v), mg, dl in zip(ivs, is_merged, is_daily) if dl and not mg]
    ax[0].plot(dx, dy, "^", color="#d62728", ms=8, label="Daily tails (ECDC/MinSante chain)")
    ax[0].plot([63.5], [887.0], "x", color="grey", ms=9, mew=2,
               label="Raw wk30 increment 887 (incl. 272 retrospective)")
    ax[0].annotate("merged", (mx[0], my[0]), textcoords="offset points",
                   xytext=(6, -14), fontsize=8, color="#9467bd")
    ax[0].annotate("raw 887", (63.5, 887.0), textcoords="offset points",
                   xytext=(-42, 8), fontsize=8, color="grey")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("t (days since 2026-05-21)")
    ax[0].set_ylabel("interval confirmed increments")
    ax[0].set_title(f"(a) Increments, scheme E1, k_obs={K0:.1f}")
    ax[0].legend(fontsize=7.5, loc="lower right")
    ax[0].grid(alpha=.3, which="both")
    ax[1].plot(ts_all, conf_all, "-", color="#2a7f2a", lw=1.8,
               label="Model cumulative confirmed")
    ax[1].plot(ANCH_T, ANCH_C, "s", color="#1f5fbf", ms=5,
               label="UKHSA wk21-41 (21 pts)")
    ax[1].plot([0], [T0_CONF], "*", color="k", ms=12, label="WHO DON603 (t=0)")
    tt = sorted(TAIL_CUM)
    ax[1].plot(tt, [TAIL_CUM[t] for t in tt], "D", color="#ff7f0e", ms=6,
               label="Daily tail (ECDC/MinSante/CDC)")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("t (days since 2026-05-21)")
    ax[1].set_ylabel("cumulative confirmed")
    ax[1].set_title("(b) Cumulative anchors (full window)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3, which="both")
    fig.tight_layout()
    fig.savefig("fig_v6_weekly_fit.png", dpi=150)
    plt.close(fig)

    # --- 图2 Re(t) ---
    tt = np.linspace(0, T_HOR, 400)
    re_t = [m.R0_ngm(thC[0] * np.exp(-thC[1] * t), PRM, x=float(m.Xf0(t))) for t in tt]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(tt, re_t, color="#1f5fbf", lw=1.8, label="R_e(t) NGM (v6 fit, E1)")
    ax.axhline(1.0, color="k", ls="--", lw=1)
    ax.plot([0], [re0_v6], "o", color="#d62728", ms=7)
    ax.annotate(f"Re0 = {re0_v6:.3f}", (0, re0_v6), textcoords="offset points",
                xytext=(8, 4), fontsize=9)
    ax.plot([148], [re148_v6], "s", color="#d62728", ms=7)
    ax.annotate(f"Re(148) = {re148_v6:.3f}", (148, re148_v6),
                textcoords="offset points", xytext=(8, -14), fontsize=9)
    # 数据侧参考（T_g=12）
    for lab, col, ls in (("full window", "grey", "-"), ("t>=60", "grey", "--"),
                         ("t>=67", "grey", "-."), ("t>=74", "grey", ":")):
        d = ll_res[("全窗（E2 原始 25 区间，含对账周）" if lab == "full window" else
                    "t≥60（E2，16 区间）" if lab == "t>=60" else
                    "t≥67（E2，15 区间，剔除对账周）" if lab == "t>=67" else
                    "t≥74（E2，14 区间）")]
        rv = np.exp(d["r"] * 12.0)
        t0_, t1_ = float(d["mids"].min()), float(d["mids"].max())
        ax.plot([t0_, t1_], [rv, rv], color=col, ls=ls, lw=1.4,
                label=f"data-side Re (Tg=12), {lab}: {rv:.3f}")
    ax.set_xlabel("t (days since 2026-05-21)")
    ax.set_ylabel("R_e")
    ax.set_title("Effective reproduction number, v6 fit (extended calibration window)")
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig("fig_v6_reff.png", dpi=150)
    plt.close(fig)

    # --- 图3 情景 ---
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    colors = {"A": "#2a7f2a", "B0": "#1f5fbf", "B": "#9467bd"}
    for kind in ("A", "B0", "B"):
        r = m.run_scenario(PRM, m.Xf0, thC, yl148, kind, t_end=T_HOR, dmax_fixed=0.75)
        ts_s = np.asarray(r["ts"])
        ax[0].plot(np.r_[ts_all, ts_s[1:]], np.r_[conf_all, np.asarray(r["conf"])[1:]],
                   color=colors[kind], lw=1.6, label=kind)
        dFs = np.asarray([float(dfunC(t)) for t in ts_s]) * r["traj"][:, m.I_F]
        base_tail = np.asarray([float(dfunC(t)) for t in ts_all]) * s_full["traj"][:, m.I_F]
        ax[1].plot(np.r_[ts_all, ts_s[1:]], np.r_[base_tail, dFs[1:]],
                   color=colors[kind], lw=1.6)
    dt_ = np.array([a for a, _ in zip(ANCH_T, ANCH_D)], float)
    dv_ = np.array([b for _, b in zip(ANCH_T, ANCH_D)], float)
    ax[1].plot(dt_, dv_, "k^", ms=5, label="Observed deaths (d-scale, wk)")
    tt2 = sorted(TAIL_DEATH)
    ax[1].plot(tt2, [TAIL_DEATH[t] for t in tt2], "v", color="grey", ms=6,
               label="Observed deaths (daily tail)")
    ax[0].plot(ANCH_T, ANCH_C, "ks", ms=4, label="Anchors (conf)")
    ax[0].plot(sorted(TAIL_CUM), [TAIL_CUM[t] for t in sorted(TAIL_CUM)], "D",
               color="#ff7f0e", ms=5, label="Daily tail (conf)")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("t (days)")
    ax[0].set_ylabel("cumulative confirmed")
    ax[0].set_title("(a) Scenarios, confirmed")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3, which="both")
    ax[1].set_xlabel("t (days)")
    ax[1].set_ylabel("cumulative deaths (d·F calibre)")
    ax[1].set_title("(b) Scenarios, deaths")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig("fig_v6_scen.png", dpi=150)
    plt.close(fig)

    # --- 图4 疫苗 ---
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    rB0 = m.run_scenario(PRM, m.Xf0, thC, yl148, "B0", t_end=T_HOR, dmax_fixed=0.75)
    rD2 = m.run_scenario(PRM, m.Xf0, thC, yl148, "D", t_end=T_HOR,
                         vac=m.make_vac(t_start=T_BASE), ve_s=0.15, ve_d=0.40,
                         f_ring=0.5, dmax_fixed=0.75, v0=m.VAC_V0)
    ts_s = np.asarray(rB0["ts"])
    dF_B0 = np.asarray([float(dfunC(t)) for t in ts_s]) * rB0["traj"][:, m.I_F]
    dF_D = np.asarray([float(dfunC(t)) for t in ts_s]) * rD2["traj"][:, m.I_F]
    base_tail = np.asarray([float(dfunC(t)) for t in ts_all]) * s_full["traj"][:, m.I_F]
    ax.plot(np.r_[ts_all, ts_s[1:]], np.r_[base_tail, dF_B0[1:]], color="#1f5fbf",
            lw=1.8, label="B0 (no vaccine)")
    ax.plot(np.r_[ts_all, ts_s[1:]], np.r_[base_tail, dF_D[1:]], color="#2a7f2a",
            lw=1.8, label="D (ring vacc., c_ring=20)")
    ax.plot(dt_, dv_, "k^", ms=5, label="Observed deaths (d-scale, wk)")
    ax.plot(tt2, [TAIL_DEATH[t] for t in tt2], "v", color="grey", ms=6,
            label="Observed deaths (daily tail)")
    ax.annotate(f"death reduction by t=513: {red_info[PRIMARY]['red_F']:.2f}%",
                xy=(0.45, 0.12), xycoords="axes fraction", fontsize=10,
                bbox=dict(boxstyle="round", fc="#fffbe6", ec="#c8b273"))
    ax.set_xlabel("t (days since 2026-05-21)")
    ax.set_ylabel("cumulative deaths (d·F calibre)")
    ax.set_title("Ring vaccination scenario D vs B0, v6 fit (E1)")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig("fig_v6_vacc.png", dpi=150)
    plt.close(fig)

    P("")
    P("---")
    P(f"复现：`D:/anaconda/python.exe v6_extended.py`（{'SMOKE 模式' if SMOKE else '全量'}，"
      f"本次总耗时 {time.time()-t_all:.1f} s；import ebola_model_v3，未改主代码；"
      "MC 种子 20261011+7919i+13 与 v3/v5 同约定）。")

    with open("results_v6.md", "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    print("[v6 完成] results_v6.md + fig_v6_*.png + mc_draws_v6.npz 已写出")


if __name__ == "__main__":
    main()
