# -*- coding: utf-8 -*-
"""
_v6_mc_boot.py —— 任务 A（MC k 集合对齐头条 MoM 18.44）+ 任务 B（200 次参数化 bootstrap）
========================================================================================
任务 A（主理人指令，biostatistician 指出 k 集合不一致）：
  results_v6.md §7 的 MC（N=1000，E1，种子 20261011+7919i+13）原逐 draw 抽
  k_obs∈{20,39,∞}（ebola_model_v3.mc_draw_prm 内置），现改为 {12,18.44,39,∞}
  等概率。改动位于 v6_extended.py 的 mc4_worker（已用"任务 A 修改段"注释标注），
  本脚本通过 V.mc4_worker 调用它，其余（ seeds、自然史/d_max 抽取、4 参数逐 draw
  重拟合、三情景、T_BASE=148/T_HOR=513、m.T_LAST=148、12 进程）完全不变。
  与 V.run_mc 的唯一区别：本脚本自建 pool.map 保留 draw 索引（V.run_mc 会过滤
  bad draw 导致索引错位），便于与旧 MC 逐 draw 对照；bad draw 记 NaN。

任务 B（v6 profile CI 有网格端点伪影；biostatistician 120 次 bootstrap 部分证实偏窄）：
  E1 V-C 主口径 200 次参数化 bootstrap：以 MLE + NB(k=18.44) 生成 23 区间伪观测
  → 2 起点 × maxiter=1500 重拟合（v6 全策略为 3 起点 × 2500；前 20 次复制同时跑
  全策略以量化差异）→ 记录 (β₀,ρ,t₀,s) → 经验 2.5/50/97.5 分位，与 profile CI 并列。

复现：D:/anaconda/python.exe _v6_mc_boot.py --phase all [--smoke]
      --smoke：MC 20 draws + bootstrap 10 次（+3 次全策略对照）。
产物：mc_draws_v6.npz（覆盖更新，旧版备份 mc_draws_v6_k20_39_inf_backup.npz）、
      _mc_boot_mc.json、_mc_boot_boot.json、results_v6_mc_boot.md、stdout 日志。
所有数值「待用户数值确认」。不改 ebola_model_v3.py / v5_full.py 既有逻辑。
"""
import sys
import time
import json
import numpy as np
import multiprocessing as mp
from scipy.optimize import minimize
import ebola_model_v3 as m
import v6_extended as V

m.T_LAST = V.T_BASE
SMOKE = "--smoke" in sys.argv
K0 = 18.44                                    # 头条 MoM（v6 §1.1）
IV_E1 = V.SCHEMES["E1"]
N_MC = 20 if SMOKE else 1000
N_BOOT = 10 if SMOKE else 200
N_CMP = 3 if SMOKE else 20                    # 全策略对照的复制数
BOOT_SEED = 20261110                          # bootstrap 种子约定（文档化）
MC_FIELDS = ["A_c", "A_dF", "A_F", "B0_c", "B0_dF", "B0_F",
             "D_c", "D_dF", "D_F", "red_F", "red_dF", "conf148", "dF148", "F148"]
P = lambda s="": print(s, flush=True)


def get_mle():
    """复现 v6 §1.1 终拟合（3 起点 × 2500），交叉核验头条值。"""
    nll, _, _ = V.make_nll(IV_E1, 0.75, V.PRM, K0)
    th, n, det = V.refit(nll, m._keys(0.75), V.vc_starts(V.TH_V5), 2500)
    return tuple(float(v) for v in th), float(n)


def doc_k_draws(n):
    """文档化：逐 draw 复现 worker 的随机数消费，记录新 k 集合的实际抽取值。"""
    ks = []
    for i in range(n):
        rng = np.random.default_rng(20261011 + 7919 * i + 13)
        m.mc_draw_prm(rng, -1.0)              # 与 worker 消费完全相同的随机数
        ks.append(float(rng.choice([12.0, 18.44, 39.0, np.inf])))
    return np.array(ks)


# ================= 任务 A：MC =================
def mc_phase():
    P("## 任务 A：MC 重跑（k_obs 逐 draw ~ {12, 18.44, 39, ∞} 等概率）")
    th_mle, nll_mle = get_mle()
    P(f"- MLE 复核：θ=({th_mle[0]:.6f}, {th_mle[1]:.6f}, {th_mle[2]:.4f}, "
      f"{th_mle[3]:.4f})，NLL={nll_mle:.4f}（v6 头条：0.178386/0.004781/5.5730/7.9013，"
      f"130.1091）")
    seeds = [20261011 + 7919 * i + 13 for i in range(N_MC)]
    args = [(i, seeds[i], th_mle, IV_E1, -1.0) for i in range(N_MC)]
    t0 = time.time()
    with mp.Pool(processes=m.N_MC_PROC) as pool:
        res = pool.map(V.mc4_worker, args, chunksize=8)   # 保序，保留索引
    P(f"- MC 完成：{len(res)} draws，耗时 {time.time()-t0:.1f} s")
    bad = [r["i"] for r in res if r.get("bad", False)]
    P(f"- bad draws：{len(bad)}" + (f"（索引 {bad[:10]}）" if bad else ""))
    arrs = {}
    for k in MC_FIELDS:
        col = np.array([r.get(k, np.nan) for r in res], float)
        arrs[k] = col
    np.savez("mc_draws_v6.npz", **arrs)
    P("- 已写出 mc_draws_v6.npz（14 字段同 v5/v6；bad → NaN）")
    kdraw = doc_k_draws(N_MC)
    ks = kdraw[np.isfinite(kdraw)]
    P(f"- 新 k 集合实际分布：12 → {int(np.sum(kdraw == 12.0))}；18.44 → "
      f"{int(np.sum(kdraw == 18.44))}；39 → {int(np.sum(kdraw == 39.0))}；"
      f"∞ → {int(np.sum(~np.isfinite(kdraw)))}（期望各 {N_MC//4}）")

    # ---- 与旧 MC（k∈{20,39,∞}）对照 ----
    old = np.load("mc_draws_v6_k20_39_inf_backup.npz")
    qs = lambda a: tuple(np.nanpercentile(a, [2.5, 50, 97.5]))
    rows = []
    P("")
    P("| 量 | 新中位数 [95%] | 旧中位数 [95%] | 中位数相对差 | 95% 宽度相对差 | k 集合影响 <1%? |")
    P("|---|---|---|---|---|---|")
    for k in MC_FIELDS:
        a, b = arrs[k], old[k]
        qa, qb = qs(a), qs(b)
        dmed = 100 * (qa[1] - qb[1]) / abs(qb[1]) if qb[1] != 0 else np.nan
        dwid = 100 * ((qa[2] - qa[0]) - (qb[2] - qb[0])) / abs(qb[2] - qb[0]) \
            if (qb[2] - qb[0]) != 0 else np.nan
        small = abs(dmed) < 1.0
        rows.append(dict(field=k, new=[float(x) for x in qa], old=[float(x) for x in qb],
                         dmed=float(dmed), dwid=float(dwid)))
        P(f"| {k} | {qa[1]:,.4g} [{qa[0]:,.4g}, {qa[2]:,.4g}] | "
          f"{qb[1]:,.4g} [{qb[0]:,.4g}, {qb[2]:,.4g}] | {dmed:+.2f}% | {dwid:+.2f}% | "
          f"{'是' if small else '否'} |")
    # 逐 draw 相关性（同 seeds、同自然史/d_max，仅 k 不同）
    # 长度守卫：smoke（新 20 draws）与旧备份（1000 draws）不等长时不可逐 draw 对齐，
    # 截断到公共长度；全量时两者均为 1000，不受影响。
    cors = {}
    n_common = min(len(arrs["A_c"]), len(old["A_c"]))
    if len(arrs["A_c"]) != len(old["A_c"]):
        P(f"- 注意：新旧 MC 长度不一致（新 {len(arrs['A_c'])} vs 旧 "
          f"{len(old['A_c'])}），逐 draw 相关性截断到 {n_common}")
    for k in MC_FIELDS:
        x, y = arrs[k][:n_common], old[k][:n_common]
        ok = np.isfinite(x) & np.isfinite(y)
        cors[k] = float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 10 else np.nan
    P("")
    P("- 逐 draw 相关性（新 vs 旧，同 seeds）：" +
      "；".join(f"{k}: {v:.3f}" for k, v in cors.items()))
    with open("_mc_boot_mc.json", "w", encoding="utf-8") as f:
        json.dump(dict(meta=dict(n=N_MC, k_set=[12.0, 18.44, 39.0, "inf"],
                                 mle=list(th_mle), nll_mle=nll_mle,
                                 bad=[int(b) for b in bad],
                                 k_counts={str(c): int(np.sum(kdraw == c))
                                           for c in (12.0, 18.44, 39.0)}
                                 | {"inf": int(np.sum(~np.isfinite(kdraw)))}),
                       rows=rows, corr=cors,
                       new_q={k: [float(x) for x in qs(arrs[k])] for k in MC_FIELDS},
                       old_q={k: [float(x) for x in qs(old[k])] for k in MC_FIELDS}),
                  f, indent=1)
    P("[任务 A 完成] _mc_boot_mc.json 已写出")
    return th_mle, nll_mle


# ================= 任务 B：参数化 bootstrap =================
def boot_worker(arg):
    r, th_mle, mods, k0, maxiter, mode = arg
    rng = np.random.default_rng(BOOT_SEED + r)
    if np.isfinite(k0):                       # NB(mean=mods, dispersion=k0)
        y = rng.negative_binomial(k0, k0 / (k0 + mods))
    else:
        y = rng.poisson(mods)
    ivals_r = [(a, b, float(yi)) for (a, b, _), yi in zip(IV_E1, y)]
    nll, _, _ = V.make_nll(ivals_r, 0.75, V.PRM, k0)
    starts = [th_mle,
              (th_mle[0] * 0.85, th_mle[1] * 2.0, th_mle[2] + 2.0, th_mle[3])]
    if mode == "full":                        # v6 全策略第 3 起点
        starts.append((th_mle[0] * 1.1, th_mle[1] * 0.4, th_mle[2] - 3.0, th_mle[3] - 1.0))
    best = None
    for st in starts:
        st = tuple(min(max(v, m.BD[k][0]), m.BD[k][1])
                   for v, k in zip(st, m._keys(0.75)))
        r_ = minimize(nll, st, method="Nelder-Mead",
                      options=dict(xatol=1e-9, fatol=1e-11, maxiter=maxiter,
                                   maxfev=maxiter, adaptive=True))
        if best is None or r_.fun < best.fun:
            best = r_
    return dict(r=int(r), mode=mode, theta=[float(v) for v in best.x],
                nll=float(best.fun), nit=int(best.nit), nfev=int(best.nfev),
                y=[float(v) for v in y])


def boot_phase():
    P("## 任务 B：E1 V-C 参数化 bootstrap（NB(k=18.44) 伪观测）")
    th_mle, nll_mle = get_mle()
    P(f"- MLE：θ=({th_mle[0]:.6f}, {th_mle[1]:.6f}, {th_mle[2]:.4f}, "
      f"{th_mle[3]:.4f})，NLL={nll_mle:.4f}")
    _, _, _, mods, _ = V.sim_pred(th_mle, IV_E1, 0.75)
    P(f"- 模型区间增量（bootstrap 均值向量）：min={mods.min():.1f}, "
      f"max={mods.max():.1f}（23 区间）")
    # 主跑：2 起点 × 1500
    args = [(r, th_mle, mods, K0, 1500, "std") for r in range(N_BOOT)]
    t0 = time.time()
    with mp.Pool(processes=12) as pool:
        res = pool.map(boot_worker, args, chunksize=1)
    P(f"- 主跑 {N_BOOT} 次完成（2 起点 × maxiter=1500），耗时 {time.time()-t0:.1f} s")
    # 全策略对照：前 N_CMP 次复制用 3 起点 × 2500
    args_c = [(r, th_mle, mods, K0, 2500, "full") for r in range(N_CMP)]
    t0 = time.time()
    with mp.Pool(processes=12) as pool:
        res_c = pool.map(boot_worker, args_c, chunksize=1)
    P(f"- 全策略对照 {N_CMP} 次完成（3 起点 × maxiter=2500），耗时 {time.time()-t0:.1f} s")

    TH = np.array([r["theta"] for r in res])
    NL = np.array([r["nll"] for r in res])
    names = ["beta0", "rho", "t0", "s"]
    # 失败/异常检测
    flag_nll = NL > nll_mle + 50.0
    at_bd = [(names[j], m.BD[names[j]][0], m.BD[names[j]][1])
             for j in range(4)
             if np.any(TH[:, j] <= m.BD[names[j]][0] + 1e-6)
             or np.any(TH[:, j] >= m.BD[names[j]][1] - 1e-6)]
    P(f"- NLL > MLE+50 的复制：{int(flag_nll.sum())}/{N_BOOT}"
      + (f"（索引 {list(np.where(flag_nll)[0])[:8]}）" if flag_nll.any() else ""))
    P(f"- 贴 BD 界的参数：{at_bd if at_bd else '无'}")

    q = lambda a: tuple(float(x) for x in np.percentile(a, [2.5, 50, 97.5]))
    prof = dict(beta0=[0.1651, 0.2051], rho=[0.0034, 0.0061],
                t0=[1.5730, 11.5730], s=[5.2346, 12.3458])   # v6 §8 profile 95% CI
    P("")
    P("| 参数 | MLE | bootstrap 2.5% | bootstrap 50% | bootstrap 97.5% | "
      "profile 95% CI | bootstrap 宽度/profile 宽度 | 中位数偏离 MLE |")
    P("|---|---|---|---|---|---|---|---|")
    rows = []
    for j, nm in enumerate(names):
        qb = q(TH[:, j])
        pw = prof[nm][1] - prof[nm][0]
        bw = qb[2] - qb[0]
        dev = 100 * (qb[1] - th_mle[j]) / abs(th_mle[j])
        rows.append(dict(param=nm, mle=float(th_mle[j]), boot=qb, profile=prof[nm],
                         width_ratio=float(bw / pw), med_dev=float(dev)))
        P(f"| {nm} | {th_mle[j]:.6f} | {qb[0]:.6f} | {qb[1]:.6f} | {qb[2]:.6f} | "
          f"[{prof[nm][0]:.4f}, {prof[nm][1]:.4f}] | {bw/pw:.2f} | {dev:+.2f}% |")
    bias = [r for r in rows if abs(r["med_dev"]) > 5.0]
    P("")
    P(f"- 中位数偏离 MLE >5% 的参数：{[r['param'] for r in bias] if bias else '无'}")
    # 全策略 vs 2 起点差异
    P("")
    P("| 复制 | 2 起点 θ(β₀,ρ,t₀,s) / NLL | 全策略 θ / NLL | max\\|Δθ\\| | ΔNLL |")
    P("|---|---|---|---|---|")
    cmp_rows = []
    for rc, rs in zip(res_c, res[:N_CMP]):
        dth = max(abs(rc["theta"][j] - rs["theta"][j]) / max(abs(rs["theta"][j]), 1e-12)
                  for j in range(4))
        dnll = rc["nll"] - rs["nll"]
        cmp_rows.append(dict(r=rc["r"], std=list(rs["theta"]), std_nll=rs["nll"],
                             full=list(rc["theta"]), full_nll=rc["nll"],
                             max_rel_dtheta=float(dth), dnl1=float(dnll)))
        P(f"| {rc['r']} | ({rs['theta'][0]:.6f}, {rs['theta'][1]:.6f}, "
          f"{rs['theta'][2]:.4f}, {rs['theta'][3]:.4f}) / {rs['nll']:.4f} | "
          f"({rc['theta'][0]:.6f}, {rc['theta'][1]:.6f}, {rc['theta'][2]:.4f}, "
          f"{rc['theta'][3]:.4f}) / {rc['nll']:.4f} | {100*dth:.2f}% | {dnll:+.4f} |")
    md = max(r["max_rel_dtheta"] for r in cmp_rows)
    mn = max(abs(r["dnl1"]) for r in cmp_rows)
    P("")
    P(f"- 2 起点 vs 全策略：最大相对 θ 差 {100*md:.3f}%，最大 \\|ΔNLL\\| {mn:.2e}"
      f"（{'可忽略，2 起点策略可用于全量' if md < 0.02 else '需注意'}）")
    with open("_mc_boot_boot.json", "w", encoding="utf-8") as f:
        json.dump(dict(meta=dict(n_boot=N_BOOT, n_cmp=N_CMP, boot_seed=BOOT_SEED,
                                 k0=K0, mle=list(th_mle), nll_mle=nll_mle,
                                 n_flagged=int(flag_nll.sum()), at_bd=at_bd),
                       rows=rows, cmp=cmp_rows,
                       draws=[dict(r=r["r"], theta=r["theta"], nll=r["nll"])
                              for r in res]), f, indent=1)
    P("[任务 B 完成] _mc_boot_boot.json 已写出")
    return rows, cmp_rows


def main():
    t_all = time.time()
    ph = "all"
    for j, a in enumerate(sys.argv[1:]):
        if a.startswith("--phase"):
            ph = a.split("=")[-1] if "=" in a else sys.argv[j + 2]
    P(f"# v6 MC k 集合对齐 + 参数化 bootstrap（phase={ph}"
      f"{'，SMOKE' if SMOKE else ''}）")
    P(f"python={sys.version.split()[0]} numpy={np.__version__} "
      f"scipy={__import__('scipy').__version__}")
    P("> 所有数值均标注「待用户数值确认」。")
    P("")
    if ph in ("mc", "all"):
        mc_phase()
        P("")
    if ph in ("boot", "all"):
        boot_phase()
        P("")
    P(f"[完成] phase={ph} 总耗时 {time.time()-t_all:.1f} s")


if __name__ == "__main__":
    main()
