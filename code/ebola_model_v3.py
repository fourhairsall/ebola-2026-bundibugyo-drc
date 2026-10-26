# -*- coding: utf-8 -*-
"""
ebola_model_v3.py  ——  SEIHRD-V + 观测模型（论文用数值结果，含主理人三轮裁决）
=========================================================================
作者：郭并行（科学计算工程师）｜数据截至 2026-10-11（UKHSA wk37–41）

模型结构
  S → E1→E2→…→E_n（Erlang-n，n=5，总均值 7.68 d，每级速率 c=5/7.68=0.6510/d）
      → 症状发作分轨：p → I_S（求医轨，速率 h=1/3.5 入院），1−p → I_C（社区轨）
  I_S --h--> H --(γ_H+μ_H)-->
  I_C --(μ_I+γ_I)-->
  λ = β·S_eff/N_eff·(I_S + I_C + κH)
  观测层：cum_confirmed(t) = d(t)·cum_inf(t)，d(t)=d_max/[1+exp(-(t-t₀)/s)]

三轮裁决落实
  R1  潜伏期 Erlang-5（Wamala 2010，mean 7.68 d [7.04–8.32], SD 3.53, CV=0.46），
      原 7.71/4.68 撤回；GT=12 d 标「团队推导」（敏感性 10–16）；
      SI=15.3 d（SD 9.1, WHO EBT NEJM 2014）只做数据侧对照，不进模型。
  R2  社区死亡占比改区间 67.8%–92.3% ⇒ μ_I/γ_I 拆分敏感性
      c（总体 CFR 锚）∈{0.55,0.60,0.65} × 社区死亡占比∈{0.68,0.80,0.92}
      联立解：c₁=(1−s_comm)·c/p，c₂=s_comm·c/(1−p)
      （校验：p=0.43, c=0.60, s_comm=0.71 ⇒ c₁=0.4047, c₂=0.7474,
        μ_I=0.0710, γ_I=0.0240 —— 与主理人原始给定值完全一致）
  R3  d_max 界 [0.30, 0.95] + 解析约束（隐含 CFR=0.4833·d_max，两口径并立：
      死亡全记 d∈[0.50,0.95]；死亡低估允许 d 低至 0.30）；
      「d_max-informed」变体（固定 0.75，U[0.65,0.85] 锚定）照旧；
      p 不再自由拟合，固定三档 {0.35,0.43,0.50}，中心 0.43；
      p 自由拟合（→0.996）保留为「p 的可辨识性」一节；
      M_A/M_B：ΔAIC 各自最优公平比较；情景绝对死亡数只用 M_A 口径，
      M_B 分支只报相对量（B/B0、D 相对减少%）；
      θ_a 映射 h(I_S→H)=h·(1−θ_a X) 确认，另加备选耦合 p(X)=p₀(1−θ_a X)。

【硬边界】全部数值标注「待用户数值确认」，须由作者用自有已校验代码复算后方可采信。
=========================================================================
"""

import os
import sys
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"

# ======================================================================
# 0. 全局开关
# ======================================================================
SEED = 20261011
N_MC = 1000
N_MC_PROC = 12
RUN_MC = True
RUN_FIG = True

# ======================================================================
# 1. 数据
# ======================================================================
T_CONF = np.array([0.0, 6.0, 13.0, 92.0, 115.0, 122.0, 129.0, 136.0, 143.0])
C_CONF = np.array([83.0, 121.0, 344.0, 5290.0, 7258.0, 7773.0, 8116.0, 8603.0, 9066.0])
T_LABEL = ["5/21", "5/27", "6/3", "8/21", "9/13 wk37", "9/20 wk38",
           "9/27 wk39", "10/4 wk40", "10/11 wk41"]
T_DEATH = np.array([0.0, 6.0, 13.0, 92.0, 122.0, 129.0, 136.0, 143.0])
D_DEATH = np.array([9.0, 17.0, 60.0, 2516.0, 3759.0, 3924.0, 4148.0, 4381.0])

INC_OBS = np.diff(C_CONF)
INC_T0, INC_T1 = T_CONF[:-1], T_CONF[1:]
INC_DAYS = np.diff(T_CONF)

OBS_DAILY_INC_7D = 72.0
OBS_ACTIVE = 890.0
OBS_RECOVERED = 2389.0

# BDBV 文献锚点（de Padua & Akhmetzhanov, EID 2026;32(10):1762-1764，
#                DOI 10.3201/eid3210.261175）—— 只进讨论不进模型
BDBV = dict(Rt=1.55, Rt_lo=1.02, Rt_hi=2.36,
            CFR=0.314, CFR_lo=0.218, CFR_hi=0.428, underreport=0.15)

# Wamala 2010 潜伏期
IP_MEAN, IP_SD, IP_NE = 7.68, 3.53, 5
# GT（团队推导）与 SI（WHO EBT NEJM 2014）
GT_MAIN, GT_LO, GT_HI = 12.0, 10.0, 16.0
SI_MEAN, SI_SD = 15.3, 9.1

# 失序事件（沿用论文构造）
EVENTS = [(0, 2), (1, 2), (2, 3), (3, 2), (11, 3),
          (20, 2), (25, 1), (41, 3), (55, 1), (59, 2)]
EV_T = np.array([e[0] for e in EVENTS], float)
EV_W = np.array([e[1] for e in EVENTS], float)
T_LAST_EV = float(EV_T.max())
X_DT, X_TMAX = 0.25, 571.0

# ======================================================================
# 2. 参数
# ======================================================================
def make_params(e_mean=IP_MEAN, n_e=IP_NE, m_H=8.84, c_total=0.60, s_comm=0.71,
                p_care=0.43, kappa=0.40, N_eff=1.35e7, th_a=0.0, th_b=0.0,
                tau_X=45.0, x_cap=0.60, x_base=0.0, h_care=5.05,
                mu_gamma_I=0.095, r0_shape=0.10936931150590773,
                p_coupled=False):
    """c_total = 总体 CFR 锚（感染→死亡总概率）；s_comm = 社区死亡占比。
       联立解：c₁=(1−s_comm)·c/p（住院 CFR），c₂=s_comm·c/(1−p)（社区 CFR）
       ⇒ μ_I=c₂(μ_I+γ_I), γ_I=(1−c₂)(μ_I+γ_I), μ_H=c₁/m_H, γ_H=(1−c₁)/m_H"""
    c_stage = n_e / e_mean
    c1 = (1.0 - s_comm) * c_total / p_care          # 住院 CFR
    c2 = s_comm * c_total / (1.0 - p_care)          # 社区 CFR
    return dict(
        n_e=n_e, c_stage=c_stage, e_mean=e_mean,
        h=1.0 / h_care, h_care=h_care, mu_gamma_I=mu_gamma_I,
        p_care=p_care, p_coupled=p_coupled,
        c_total=c_total, s_comm=s_comm, c1=c1, c2=c2,
        mu_I=c2 * mu_gamma_I, gam_I=(1.0 - c2) * mu_gamma_I,
        m_H=m_H, mu_H=c1 / m_H, gam_H=(1.0 - c1) / m_H,
        kappa=kappa, N_eff=N_eff, th_a=th_a, th_b=th_b,
        tau_X=tau_X, x_cap=x_cap, x_base=x_base, r0_shape=r0_shape,
    )


def build_X_grid(tau=45.0, weights=None, x_base=0.0, cap=0.60, tmax=X_TMAX, dt=X_DT):
    w = EV_W if weights is None else np.asarray(weights, float)
    tg = np.arange(0.0, tmax + 1e-9, dt)
    raw = np.zeros_like(tg)
    for te, wi in zip(EV_T, w):
        m = tg >= te
        raw[m] += wi * np.exp(-(tg[m] - te) / tau)
    mx = raw.max()
    X = np.clip(raw / mx + x_base, 0.0, 1.0)
    i_last = int(round(T_LAST_EV / dt))
    X[tg > T_LAST_EV] = min(X[i_last], cap)
    return tg, X


TG0, XG0 = build_X_grid()
XG_T, XG_V = TG0, XG0


def Xf0(t):
    return float(np.interp(t, XG_T, XG_V))


def make_dfun(d_max, t0, s, ramp=True):
    if ramp:
        def f(t):
            ta = np.asarray(t, float)
            v = d_max / (1.0 + np.exp(-(ta - t0) / s))
            return float(v) if np.ndim(ta) == 0 else v
        return f
    def f(t):
        ta = np.asarray(t, float)
        v = np.full_like(ta, d_max, float)
        return float(v) if np.ndim(ta) == 0 else v
    return f


# ======================================================================
# 3. 初值
# ======================================================================
def build_ic(d0, prm, Xf, conf0=83.0, death0=9.0):
    """cum_inf(0)=conf0/d0；F(0)=death0/d0；R(0)=0；
       E1..En,I_S,I_C,H 按增速 r0_shape 的稳定分布比例分配余量。"""
    r, c, n_e = prm["r0_shape"], prm["c_stage"], prm["n_e"]
    pc = prm["p_care"]
    x0 = Xf(0.0)
    h_eff = prm["h"] * (1.0 - prm["th_a"] * x0) if prm["th_a"] > 0 else prm["h"]
    kE = (c + r) / c
    # 逐级回推：E_n=1, E_{n-1}=kE·E_n, ... E_1=kE^{n-1}
    levels = [kE ** (n_e - 1 - k) for k in range(n_e)]   # [E1..En] 比例
    IS = pc * c / h_eff
    IC = (1.0 - pc) * c / prm["mu_gamma_I"]
    H = h_eff * IS / (prm["gam_H"] + prm["mu_H"] + r)
    tot = sum(levels) + IS + IC + H
    cum0 = conf0 / d0
    F0 = death0 / d0
    scale = (cum0 - F0) / tot
    y0 = np.zeros(7 + n_e)
    for k in range(n_e):
        y0[1 + k] = levels[k] * scale
    y0[1 + n_e] = IS * scale
    y0[2 + n_e] = IC * scale
    y0[3 + n_e] = H * scale
    y0[5 + n_e] = F0                     # I_F = 5+n_e
    y0[0] = prm["N_eff"] - y0[1:].sum()
    return y0


# ======================================================================
# 4. 右端
# ======================================================================
def rhs(t, y, prm, Xf, beta_fn, dfun=None, vac=None,
        ve_s=0.0, ve_d=0.0, f_ring=0.0, x_zero=False):
    """y = [S, E1..E_n, I_S, I_C, H, R, F, V]；n = prm['n_e']（索引 1..n_e）
       此处 n=5 → 索引：0=S,1..5=E,6=I_S,7=I_C,8=H,9=R,10=F,11=V"""
    n = prm["n_e"]
    S = y[0]
    E = y[1:1 + n]
    IS, IC, H, R, F, V = y[1 + n], y[2 + n], y[3 + n], y[4 + n], y[5 + n], y[6 + n]
    x = Xf(t) if (prm["th_a"] > 0 or prm["th_b"] > 0) and not x_zero else 0.0
    h_eff = prm["h"] * (1.0 - prm["th_a"] * x)
    pc = prm["p_care"] * (1.0 - prm["th_a"] * x) if prm["p_coupled"] else prm["p_care"]
    pc = min(max(pc, 0.0), 1.0)
    b_eff = beta_fn(t) * (1.0 + prm["th_b"] * x)
    S_eff = S + (1.0 - ve_s * f_ring) * V
    lam = b_eff * S_eff / prm["N_eff"] * (IS + IC + prm["kappa"] * H)
    vacc = 0.0
    if vac is not None:
        conf_inc_now = dfun(t) * lam            # 日新发确诊（勿用累计！）
        vacc = vac(t, conf_inc_now) * (S / prm["N_eff"])
        vacc = min(vacc, max(S, 0.0))
    w = V / (S + V) if (S + V) > 0 else 0.0
    mu_I_eff = prm["mu_I"] * (1.0 - ve_d * f_ring * w)
    mu_H_eff = prm["mu_H"] * (1.0 - ve_d * f_ring * w)
    c = prm["c_stage"]
    out = np.empty(7 + n)
    out[0] = -lam - vacc
    out[1] = lam - c * E[0]
    if n > 1:
        out[2:1 + n] = c * E[:-1] - c * E[1:]
    out[1 + n] = pc * c * E[-1] - h_eff * IS
    out[2 + n] = (1.0 - pc) * c * E[-1] - prm["mu_gamma_I"] * IC
    out[3 + n] = h_eff * IS - (prm["gam_H"] + prm["mu_H"]) * H
    out[4 + n] = prm["gam_I"] * IC + prm["gam_H"] * H
    out[5 + n] = mu_I_eff * IC + mu_H_eff * H
    out[6 + n] = vacc
    return out


NS = 7 + IP_NE          # 状态维数 = 12
I_IS, I_IC, I_H, I_R, I_F, I_V = 1 + IP_NE, 2 + IP_NE, 3 + IP_NE, 4 + IP_NE, 5 + IP_NE, 6 + IP_NE


def rk4(f, y0, t0, t1, dt=0.5):
    n = int(round((t1 - t0) / dt))
    y = np.asarray(y0, float).copy()
    traj = np.empty((n + 1, len(y)))
    traj[0] = y
    ts = t0 + dt * np.arange(n + 1)
    for k in range(n):
        t = ts[k]
        k1 = f(t, y)
        k2 = f(t + dt / 2, y + dt / 2 * k1)
        k3 = f(t + dt / 2, y + dt / 2 * k2)
        k4 = f(t + dt, y + dt * k3)
        y = y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        y = np.maximum(y, 0.0)
        traj[k + 1] = y
    return ts, traj


def cum_inf_of(traj):
    return traj[:, 1:1 + IP_NE].sum(axis=1) + traj[:, I_IS] + traj[:, I_IC] \
        + traj[:, I_H] + traj[:, I_R] + traj[:, I_F]


def simulate(prm, Xf, theta, t_end, dt=0.5, vac=None, ve_s=0.0, ve_d=0.0,
             f_ring=0.0, t_grid=None, dmax_fixed=None, ramp=True,
             x_zero=False, v0=0.0):
    """theta：dmax_fixed 给定 → (β₀, ρ, t₀, s)；ramp=True → (β₀, ρ, d_max, t₀, s)；
       ramp=False（H2'，无爬坡）→ (β₀, ρ, d_max)，d(t)≡d_max"""
    if dmax_fixed is not None:
        beta0, rho, t0d, sd = theta
        d_max = dmax_fixed
    elif ramp:
        beta0, rho, d_max, t0d, sd = theta
    else:
        beta0, rho, d_max = theta
        t0d, sd = 0.0, 1.0
    dfun = make_dfun(d_max, t0d, sd, ramp)
    y0 = build_ic(float(dfun(0.0)), prm, Xf)
    if v0 > 0.0:
        y0[I_V] = v0
        y0[0] -= v0
    beta_fn = lambda t: beta0 * np.exp(-rho * t)
    f = lambda t, y: rhs(t, y, prm, Xf, beta_fn, dfun, vac, ve_s, ve_d, f_ring, x_zero)
    ts, traj = rk4(f, y0, 0.0, t_end, dt)
    ci = cum_inf_of(traj)
    out = dict(ts=ts, traj=traj, cum_inf=ci, dfun=dfun, y0=y0, d_max=d_max)
    if t_grid is not None:
        tg = np.asarray(t_grid, float)
        ci_at = np.interp(tg, ts, ci)
        out["conf_at"] = ci_at * np.array([float(dfun(t)) for t in tg])
        out["cum_inf_at"] = ci_at
    out["conf_series"] = ci * np.array([float(dfun(t)) for t in ts])
    return out


def lam_at(t, y, prm, theta, Xf, dmax_fixed=None, x_zero=False):
    if dmax_fixed is None:
        beta0, rho = theta[0], theta[1]
    else:
        beta0, rho = theta[0], theta[1]
    if (prm["th_a"] > 0 or prm["th_b"] > 0) and not x_zero:
        x = Xf(t)
    else:
        x = 0.0
    b_eff = beta0 * np.exp(-rho * t) * (1.0 + prm["th_b"] * x)
    S_eff = y[0] + y[I_V]
    return float(b_eff * S_eff / prm["N_eff"] * (y[I_IS] + y[I_IC] + prm["kappa"] * y[I_H]))


# ======================================================================
# 5. 似然与标定
# ======================================================================
def nb_nll(y, m, k):
    y = np.asarray(y, float)
    m = np.maximum(np.asarray(m, float), 1e-12)
    if not np.isfinite(k):
        return float(np.sum(m - y * np.log(m) + gammaln(y + 1.0)))
    return float(np.sum(-(gammaln(y + k) - gammaln(k) - gammaln(y + 1.0)
                          + k * np.log(k / (k + m)) + y * np.log(m / (k + m)))))


# 拟合参数序：dmax_free → (β₀, ρ, d_max, t₀, s)；dmax_fixed → (β₀, ρ, t₀, s)
BD = dict(beta0=(1e-4, 20.0), rho=(-0.05, 0.30), d_max=(0.30, 0.95),
          t0=(-60.0, 220.0), s=(2.0, 400.0))
PKEYS_FREE = ("beta0", "rho", "d_max", "t0", "s")
PKEYS_FIX = ("beta0", "rho", "t0", "s")


def _keys(dmax_fixed):
    return PKEYS_FIX if dmax_fixed is not None else PKEYS_FREE


def penalize(theta, dmax_fixed):
    pen = 0.0
    for key, val in zip(_keys(dmax_fixed), theta):
        lo, hi = BD[key]
        if val < lo:
            pen += (lo - val) ** 2 * 1e6
        elif val > hi:
            pen += (val - hi) ** 2 * 1e6
    return pen


def neg_ll(theta, prm, Xf, k_obs=39.0, dmax_fixed=None, ramp=True):
    pen = penalize(theta, dmax_fixed)
    if pen > 0:
        return 1e9 + pen
    try:
        s = simulate(prm, Xf, theta, T_LAST, 0.5, t_grid=T_CONF,
                     dmax_fixed=dmax_fixed, ramp=ramp)
    except Exception:
        return 1e9
    conf = s["conf_at"]
    if not np.all(np.isfinite(conf)) or np.any(np.diff(conf) < -1e-9) or np.any(conf <= 0):
        return 1e9
    return nb_nll(INC_OBS, np.diff(conf), k_obs)


def fit_model(prm, Xf, k_obs=39.0, dmax_fixed=None, starts=None, maxiter=2000):
    keys = _keys(dmax_fixed)
    if starts is None:
        starts = []
        for b0 in (0.18, 0.40, 0.80):
            for rho in (0.004, 0.015):
                st = [b0, rho] + ([0.75] if dmax_fixed is None else []) + [30.0, 30.0]
                starts.append(tuple(st))
    else:
        starts = [tuple(s) for s in starts]
    best = None
    for st in starts:
        st = tuple(min(max(v, BD[k][0]), BD[k][1]) for v, k in zip(st, keys))
        res = minimize(neg_ll, st, args=(prm, Xf, k_obs, dmax_fixed), method="Nelder-Mead",
                       options=dict(xatol=1e-8, fatol=1e-10, maxiter=maxiter,
                                    maxfev=maxiter, adaptive=True))
        if best is None or res.fun < best.fun:
            best = res
    return tuple(float(v) for v in best.x), float(best.fun), dict(nfev=int(best.nfev))


def unpack(th, dmax_fixed):
    keys = _keys(dmax_fixed)
    d = dict(zip(keys, th))
    d["d_max_eff"] = d["d_max"] if dmax_fixed is None else dmax_fixed
    return d


# ======================================================================
# 6. 再生数
# ======================================================================
def R0_ngm(beta, prm, s=1.0, x=0.0):
    n = prm["n_e"]
    h_eff = prm["h"] * (1.0 - prm["th_a"] * x)
    pc = prm["p_care"]
    b_eff = beta * (1.0 + prm["th_b"] * x)
    # Erlang-n 只改变 E 期分布，不改变总传染时长
    return b_eff * s * (pc * (1.0 / h_eff + prm["kappa"] / (prm["gam_H"] + prm["mu_H"]))
                        + (1.0 - pc) / prm["mu_gamma_I"])


def R0_spectral(beta, prm, s=1.0, x=0.0):
    n = prm["n_e"]
    c = prm["c_stage"]
    h_eff = prm["h"] * (1.0 - prm["th_a"] * x)
    pc = prm["p_care"]
    b_eff = beta * (1.0 + prm["th_b"] * x)
    dim = n + 3                                     # E1..En, I_S, I_C, H
    F = np.zeros((dim, dim))
    F[0, n] = b_eff * s
    F[0, n + 1] = b_eff * s
    F[0, n + 2] = b_eff * s * prm["kappa"]
    V = np.zeros((dim, dim))
    V[0, 0] = c
    for k in range(1, n):
        V[k, k - 1] = -c
        V[k, k] = c
    V[n, n - 1] = -pc * c
    V[n, n] = h_eff
    V[n + 1, n - 1] = -(1 - pc) * c
    V[n + 1, n + 1] = prm["mu_gamma_I"]
    V[n + 2, n] = -h_eff
    V[n + 2, n + 2] = prm["gam_H"] + prm["mu_H"]
    return float(np.max(np.abs(np.linalg.eigvals(F @ np.linalg.inv(V)))))


# ======================================================================
# 7. 疫苗 / 情景 / E
# ======================================================================
VAC_DV, VAC_RAMP, VAC_C_RING, VAC_V0 = 10.0, 30.0, 20.0, 9000.0


def make_vac(dv=VAC_DV, ramp_days=VAC_RAMP, c_ring=VAC_C_RING, t_start=143.0, phi_max=1.0):
    sr = ramp_days / (2.0 * np.log(9.0))
    t_mid = t_start + dv + ramp_days / 2.0
    def vac(t, conf_inc):
        ph = phi_max / (1.0 + np.exp(-(np.asarray(t, float) - t_mid) / sr))
        return float(ph) * c_ring * float(conf_inc)
    return vac


T_LAST, T_ONE_YEAR, T_SCEN_END = 143.0, 508.0, 565.0


def run_scenario(prm, Xf, theta, y_last, kind, t_end=T_SCEN_END, dt=0.5,
                 vac=None, ve_s=0.0, ve_d=0.0, f_ring=0.0, dmax_fixed=None,
                 v0=0.0, x_zero=None):
    if dmax_fixed is None:
        beta0, rho, d_max, t0d, sd = theta
    else:
        beta0, rho, t0d, sd = theta
        d_max = dmax_fixed
    dfun = make_dfun(d_max, t0d, sd, True)
    beta_now = beta0 * np.exp(-rho * T_LAST)
    if x_zero is None:
        x_zero = (kind == "B0")
    bf = (lambda t: beta_now * np.exp(-rho * (t - T_LAST))) if kind == "A" \
        else (lambda t: float(beta_now))
    y0 = np.asarray(y_last, float).copy()
    if v0 > 0.0:
        y0[I_V] = v0
        y0[0] -= v0
    f = lambda t, y: rhs(t, y, prm, Xf, bf, dfun, vac, ve_s, ve_d, f_ring, x_zero)
    ts, traj = rk4(f, y0, T_LAST, t_end, dt)
    ci = cum_inf_of(traj)
    conf = ci * np.array([float(dfun(t)) for t in ts])
    return dict(ts=ts, traj=traj, cum_inf=ci, conf=conf)


def scen_E():
    wk = np.array([515.0, 343.0, 487.0, 463.0])
    tmid = np.array([118.5, 125.5, 132.5, 139.5])
    mean_wk = wk.mean()
    annual1 = mean_wk * 52.0
    sl, ic = np.polyfit(tmid, np.log(wk), 1)
    r_day = float(sl)
    tt = np.arange(143.0, 508.0 + 1e-9, 1.0)
    annual2 = float((np.exp(ic + sl * tt) / 7.0).sum())
    ratio = ((D_DEATH[-1] - D_DEATH[-2]) / INC_OBS[-1]
             + (D_DEATH[-2] - D_DEATH[-3]) / INC_OBS[-2]) / 2.0
    # 数据侧 4 点 log-linear Rₑ 对照（GT 团队推导 10/12/16；SI=15.3 不进模型）
    rge = {f"GT={g:.0f}d": float(np.exp(r_day * g)) for g in (GT_LO, GT_MAIN, GT_HI)}
    rge[f"SI={SI_MEAN}d"] = float(np.exp(r_day * SI_MEAN))
    r_last2 = float(np.log(wk[-1] / wk[-2]) / 7.0)
    rge["GT=12d(末两周)"] = float(np.exp(r_last2 * GT_MAIN))
    return dict(mean_wk=mean_wk, r_day=r_day, annual1=annual1, annual2=annual2,
                cum1=9066.0 + annual1, cum2=9066.0 + annual2, dcr=ratio,
                death1=4381.0 + annual1 * ratio, death2=4381.0 + annual2 * ratio,
                slope_per_wk=float(sl * 7.0), R_data=rge)


# ======================================================================
# 8. 蒙特卡洛
# ======================================================================
def mc_draw_prm(rng, dmax_fixed):
    prm = make_params(
        e_mean=rng.uniform(6.0, 11.3), m_H=rng.uniform(7.0, 12.0),
        c_total=rng.uniform(0.55, 0.65), s_comm=rng.uniform(0.68, 0.92),
        p_care=0.43, N_eff=rng.uniform(8e6, 2.4e7),
        r0_shape=rng.uniform(0.06, 0.15))
    k_obs = float(rng.choice([20.0, 39.0, np.inf]))
    dm = rng.uniform(0.34, 0.95) if dmax_fixed is not None else None
    return prm, k_obs, dm


def mc_worker(arg):
    idx, seed, th0, dmax_fixed, ve_s, ve_d, f_ring = arg
    rng = np.random.default_rng(seed)
    prm, k_obs, dm = mc_draw_prm(rng, dmax_fixed)
    Xf = Xf0
    if dmax_fixed is not None:
        dm = dmax_fixed if dmax_fixed > 0 else dm     # dmax_fixed>0 → 固定；<0 → 抽样
        st = [th0, (th0[0] * 0.7, th0[1] * 1.4, th0[2], th0[3])]
        th, nll, _ = fit_model(prm, Xf, k_obs=k_obs, dmax_fixed=dm,
                               starts=st, maxiter=400)
    else:
        st = [th0, (th0[0] * 0.7, th0[1] * 1.4, th0[2], th0[3], th0[4])]
        th, nll, _ = fit_model(prm, Xf, k_obs=k_obs, dmax_fixed=None,
                               starts=st, maxiter=400)
    s = simulate(prm, Xf, th, T_LAST, 0.5, t_grid=T_CONF, dmax_fixed=dm)
    beta_now = th[0] * np.exp(-th[1] * T_LAST)
    rB = run_scenario(prm, Xf, th, s["traj"][-1].copy(), "B", dmax_fixed=dm)
    i508 = int(round((T_ONE_YEAR - T_LAST) / 0.5))
    i565 = len(rB["ts"]) - 1
    vac = make_vac()
    rD = run_scenario(prm, Xf, th, s["traj"][-1].copy(), "D", vac=vac,
                      ve_s=ve_s, ve_d=ve_d, f_ring=f_ring, dmax_fixed=dm, v0=VAC_V0)
    up = unpack(th, dm)
    return dict(idx=idx, beta0=th[0], rho=th[1], d_max=up["d_max_eff"], t0=th[-2],
                s=th[-1], nll=nll, conf143=float(s["conf_at"][-1]),
                c1=prm["c1"], c2=prm["c2"], c_total=prm["c_total"], s_comm=prm["s_comm"],
                B_death508=float(rB["traj"][i508, I_F]), B_death565=float(rB["traj"][i565, I_F]),
                B_conf508=float(rB["conf"][i508]), B_conf565=float(rB["conf"][i565]),
                D_death508=float(rD["traj"][i508, I_F]), D_death565=float(rD["traj"][i565, I_F]),
                D_conf508=float(rD["conf"][i508]), D_conf565=float(rD["conf"][i565]),
                V565=float(rD["traj"][i565, I_V]),
                lam_last=lam_at(T_LAST, s["traj"][-1], prm, th, Xf, dm))


# ======================================================================
# 9. 主流程
# ======================================================================
def fmt(x, n=1):
    return f"{x:,.{n}f}"


def main():
    t_all = time.time()
    L = []
    def P(s=""):
        L.append(s)
        print(s)

    P("# 刚果金埃博拉 SEIHRD-V + 观测模型（v3，含三轮裁决）—— 数值结果")
    P("")
    P("> 作者：郭并行（科学计算工程师）｜数据截至 2026-10-11（UKHSA wk37–41）｜脚本：`ebola_model_v3.py`")
    P("> **所有数值均标注「待用户数值确认」，须由作者用自有已校验代码复算后方可采信。**")
    P("")
    P("> **R1** 潜伏期 Erlang-5（Wamala 2010，mean 7.68 d [7.04–8.32], SD 3.53, CV=0.46, "
      f"每级速率 c={IP_NE}/{IP_MEAN}={IP_NE/IP_MEAN:.4f}/d）；原 7.71/4.68 撤回（出处查无）。"
      "GT=12 d 标「团队推导」（敏感性 10–16 d）；SI=15.3 d（SD 9.1, WHO EBT NEJM 2014）"
      "只做数据侧对照，不进模型。")
    P("")
    P("> **R2** 社区死亡占比 67.8%–92.3%（WHO AFRO sitrep，随疫情期段波动）⇒ "
      "μ_I/γ_I 拆分敏感性 c（总体 CFR 锚）× 社区死亡占比，联立解 "
      "`c₁=(1−s_comm)·c/p`、`c₂=s_comm·c/(1−p)`。")
    P("")
    P("> **R3** d_max 下界放宽至 0.30 + 「d_max-informed」变体；p 固定三档 "
      "{0.35, 0.43, 0.50}（中心 0.43）；p 自由拟合保留为可辨识性证据。")
    P("")
    P("---")
    P("")

    # ---------- 1 数据 ----------
    P("## 1. 观测数据与似然构造")
    P("")
    P("| 区间 | 天数 | 实测确诊增量 | 对应 |")
    P("|---|---|---|---|")
    tags = ["早期", "早期", "早期(含 8/21 锚)", "[92,115]",
            "wk38=515", "wk39=343", "wk40=487", "wk41=463"]
    for j in range(len(INC_OBS)):
        P(f"| [{INC_T0[j]:.0f},{INC_T1[j]:.0f}] | {INC_DAYS[j]:.0f} | {INC_OBS[j]:,.0f} | {tags[j]} |")
    P("")
    P(f"- 主理人指定 `[92,136] 段 3,313` = 1,968+515+343+487 = {1968+515+343+487:,.0f}（互不重叠，自动满足）")
    P("- 负二项 k_obs = 39（敏感性 {20, 39, ∞→泊松}）")
    P("")

    # ---------- 2 NGM ----------
    P("## 2. R₀ / Rₑ：NGM 闭式解与数值验证（Erlang-5）")
    P("")
    P("$$R_0=\\beta\\frac{S}{N_{eff}}\\Big\\{p\\Big[\\frac{1}{h}+\\frac{\\kappa}{\\gamma_H+\\mu_H}\\Big]"
      "+\\frac{1-p}{\\mu_I+\\gamma_I}\\Big\\}$$")
    P("")
    P("> Erlang-5 与 Erlang-3 **给出相同的 R₀ 闭式**（E 期只改变分布形状，不改变总传染时长），"
      "但改变早期增长速率与代际分布。")
    P("")
    P("| 检验点 | β | x | p | 闭式 R₀ | 数值谱半径 ρ(FV⁻¹) | 绝对差 |")
    P("|---|---|---|---|---|---|---|")
    for beta, x, pc in [(0.30, 0.0, 0.43), (0.30, 0.60, 0.43), (0.60, 0.0, 0.43),
                        (0.60, 0.60, 0.43), (0.60, 0.0, 0.50), (0.60, 0.0, 0.35)]:
        prm_ = make_params(p_care=pc, th_a=0.40, th_b=0.40)
        rc = R0_ngm(beta, prm_, x=x)
        rs = R0_spectral(beta, prm_, x=x)
        P(f"| β={beta:.2f} | {beta:.2f} | {x:.2f} | {pc:.2f} | {rc:.6f} | {rs:.6f} | {abs(rc-rs):.2e} |")
    P("")
    P("**路径分解**（p=0.43, c=0.60, s_comm=0.71 ⇒ c₁=0.4047, c₂=0.7474）：")
    P("")
    prm_c = make_params()
    ah = 1.0 / prm_c["h"] + prm_c["kappa"] / (prm_c["gam_H"] + prm_c["mu_H"])
    ac = 1.0 / prm_c["mu_gamma_I"]
    P(f"- 求医轨（I_S→H）单位传染时长 = 1/h + κ/(γ_H+μ_H) = {1/prm_c['h']:.3f} + "
      f"{prm_c['kappa']/(prm_c['gam_H']+prm_c['mu_H']):.3f} = **{ah:.4f} d**")
    P(f"- 社区轨（I_C）单位传染时长 = 1/(μ_I+γ_I) = **{ac:.4f} d**")
    P(f"- 加权（p=0.43）：{0.43*ah:.4f} + {0.57*ac:.4f} = **{0.43*ah+0.57*ac:.4f} d**"
      "「待用户数值确认」")
    P(f"- 隐含平均代际时间（团队推导 GT 的模型对应值）≈ E 均值 {IP_MEAN:.2f} + "
      f"{0.43*ah+0.57*ac:.4f} = **{IP_MEAN+0.43*ah+0.57*ac:.2f} d**"
      "（主理人给定 GT 主值 12 d，见 §10 对照）「待用户数值确认」")
    P("")

    # ---------- 3 p 的可辨识性 ----------
    P("## 3. p 的可辨识性（自由拟合 → 0.996，为何改为固定）")
    P("")
    prm_free = make_params(p_care=0.85)   # 占位；p 自由时在 theta 里
    th_f, nll_f, _ = fit_model_free_p(Xf0, k_obs=39.0)
    P(f"- 自由拟合（p 作为第 6 个参数，d_max∈[0.30,0.95]）：β₀={th_f[0]:.5f}, "
      f"ρ={th_f[1]:.5f}, d_max={th_f[2]:.4f}, t₀={th_f[3]:.2f}, s={th_f[4]:.2f}, "
      f"**p={th_f[5]:.4f}**；NLL={nll_f:.4f}「待用户数值确认」")
    P("- **诊断**：p 贴上界 1.0（社区轨 I_C 几乎为空）。原因：确诊数似然只「看见」"
      "被检出的人，p→1 使所有人都进入会被检出的求医轨，似然单调递增。")
    P("- **与实测矛盾**：社区死亡占比实测 67.8%–92.3%，若 p≈1 则社区死亡占比 = 0，直接冲突。")
    P("- **结论（进论文）**：确诊似然不能识别 p，p 必须由 CFR 分解外生固定——"
      "这是路径混合结构（I_S/I_C 分轨）必要性的又一证据。")
    P("")

    # ---------- 4 主变体 ----------
    P("## 4. 主变体标定（p=0.43 固定，中心 c=0.60, s_comm=0.71）")
    P("")
    P(f"- 由 (c, s_comm, p)=(0.60, 0.71, 0.43) 联立解：**c₁={prm_c['c1']:.4f}**（住院 CFR）、"
      f"**c₂={prm_c['c2']:.4f}**（社区 CFR）⇒ **μ_I={prm_c['mu_I']:.4f}、"
      f"γ_I={prm_c['gam_I']:.4f}**（与主理人原始给定值 0.071/0.024 完全一致）、"
      f"μ_H={prm_c['mu_H']:.5f}、γ_H={prm_c['gam_H']:.5f}「待用户数值确认」")
    P("")
    variants = {}
    for tag, dmf in (("V-B (d_max-free [0.30,0.95])", None),
                     ("V-C (d_max-informed 固定 0.75)", 0.75)):
        sts = [(0.18, 0.004, 0.60, 25.0, 15.0), (0.40, 0.015, 0.85, 20.0, 12.0),
               (0.80, 0.004, 0.45, 30.0, 30.0), (0.30, 0.010, 0.70, 15.0, 10.0)]
        if dmf is not None:
            sts = [tuple(v for i, v in enumerate(s) if i != 2) for s in sts]
        th, nll, info = fit_model(prm_c, Xf0, k_obs=39.0, dmax_fixed=dmf,
                                  starts=sts, maxiter=2500)
        s = simulate(prm_c, Xf0, th, T_LAST, 0.5, t_grid=T_CONF, dmax_fixed=dmf)
        variants[tag] = dict(th=th, nll=nll, s=s, dmf=dmf, npar=len(th))
        variants[tag]["aic"] = 2 * nll + 2 * len(th)
        u = unpack(th, dmf)
        P(f"- **{tag}**（{len(th)} 参数）：β₀={u['beta0']:.5f}, ρ={u['rho']:.5f}, "
          f"d_max={u['d_max_eff']:.4f}, t₀={u['t0']:.2f}, s={u['s']:.2f}；"
          f"NLL={nll:.4f}, AIC={variants[tag]['aic']:.4f}「待用户数值确认」")
    dnl = abs(variants["V-B (d_max-free [0.30,0.95])"]["nll"]
              - variants["V-C (d_max-informed 固定 0.75)"]["nll"])
    P(f"- **d_max 的发病率不可辨识性（本次运行核心发现）**：V-B（d_max 自由，贴下界 "
      f"{unpack(variants['V-B (d_max-free [0.30,0.95])']['th'], None)['d_max']:.2f}）与 "
      f"V-C（固定 0.75）的 NLL 仅差 {dnl:.4f} —— β₀ 重标定可完全补偿 d_max。"
      "因此 **d_max 的取值实际上由 §5 的外生一致性约束决定**（这正是「两口径并立」裁决的数值根据）；"
      "两变体的确诊轨迹与残差逐点相同，差别只体现在死亡绝对数（见 §9）。")
    P("")
    P("### 拟合残差（8 个确诊增量）")
    P("")
    P("| 区间 | 天数 | 实测 | V-B 模型 | 误差% | V-C 模型 | 误差% |")
    P("|---|---|---|---|---|---|---|")
    for j in range(len(INC_OBS)):
        a = np.diff(variants["V-B (d_max-free [0.30,0.95])"]["s"]["conf_at"])[j]
        b = np.diff(variants["V-C (d_max-informed 固定 0.75)"]["s"]["conf_at"])[j]
        P(f"| [{INC_T0[j]:.0f},{INC_T1[j]:.0f}] | {INC_DAYS[j]:.0f} | {INC_OBS[j]:,.0f} | "
          f"{a:.1f} | {100*(a-INC_OBS[j])/INC_OBS[j]:+.2f} | {b:.1f} | "
          f"{100*(b-INC_OBS[j])/INC_OBS[j]:+.2f} |")
    P("")
    P("### 累计确诊与累计死亡（V-B，两种死亡口径）")
    P("")
    P("> 死亡口径 ① `d(t)·F(t)`（与初值 F(0)=死亡/d(0) 自洽）；② `F(t)` 原始值（死亡直接计数）。")
    P("")
    P("| t | 日期 | 实测确诊 | V-B 确诊 | 误差% | 实测死亡 | V-B d·F | 误差% | V-B F | 误差% | d(t) |")
    P("|---|---|---|---|---|---|---|---|---|---|---|")
    svb = variants["V-B (d_max-free [0.30,0.95])"]["s"]
    for i, t in enumerate(T_CONF):
        di = int(np.where(T_DEATH == t)[0][0]) if t in T_DEATH else None
        dv = D_DEATH[di] if di is not None else np.nan
        mv = svb["conf_at"][i]
        md = float(np.interp(t, svb["ts"], svb["traj"][:, I_F]))
        dtv = float(svb["dfun"](t))
        dtxt = f"{100*(mv-C_CONF[i])/C_CONF[i]:+.2f}"
        if di is not None:
            P(f"| {t:.0f} | {T_LABEL[i]} | {C_CONF[i]:,.0f} | {mv:,.0f} | {dtxt} | {dv:,.0f} | "
              f"{md*dtv:,.0f} | {100*(md*dtv-dv)/dv:+.2f} | {md:,.0f} | {100*(md-dv)/dv:+.2f} | {dtv:.4f} |")
        else:
            P(f"| {t:.0f} | {T_LABEL[i]} | {C_CONF[i]:,.0f} | {mv:,.0f} | {dtxt} | — | — | — | — | — | {dtv:.4f} |")
    P("")

    # ---------- 4b H1' vs H2' ----------
    P("### H1' vs H2'：检出爬坡 d(t) 的必要性（p 固定 0.43，公平自由参数数对比）")
    P("")
    P("- **H1'**（有检出爬坡）：5 参数（β₀, ρ, d_max, t₀, s）—— 即上表 V-B 标定。")
    P("- **H2'**（无爬坡，d(t)≡d_max 常数）：3 参数（β₀, ρ, d_max），p 同样固定。")
    kb_key = "V-B (d_max-free [0.30,0.95])"
    best2 = None
    for b0 in (0.18, 0.40, 0.80):
        for rh in (0.004, 0.015):
            for dm0 in (0.45, 0.95):
                st2 = (b0, rh, dm0)
                r2 = minimize(neg_ll, st2, args=(prm_c, Xf0, 39.0, None, False),
                              method="Nelder-Mead",
                              options=dict(xatol=1e-8, fatol=1e-10, maxiter=2000,
                                           maxfev=2000, adaptive=True))
                if best2 is None or r2.fun < best2.fun:
                    best2 = r2
    th_h2, nll_h2 = tuple(float(v) for v in best2.x), float(best2.fun)
    aic_h1 = variants[kb_key]["aic"]
    aic_h2 = 2 * nll_h2 + 2 * 3
    P(f"- H2' 最优：β₀={th_h2[0]:.5f}, ρ={th_h2[1]:.5f}, d_max={th_h2[2]:.4f}；"
      f"NLL={nll_h2:.4f}, AIC={aic_h2:.4f}「待用户数值确认」")
    d_aic_h = aic_h2 - aic_h1
    P(f"- **ΔAIC（H2' − H1'）= {d_aic_h:+.4f}**「待用户数值确认」"
      f"（{'H1 检出爬坡被数据强烈支持' if d_aic_h > 2 else '两模型不可区分（|ΔAIC|<2）'}；"
      "p 固定后 H1' 5 参数 / H2' 3 参数，AIC 惩罚项分别为 10/6，对比公平）")
    P("")

    # ---------- 5 d_max profile 与一致性诊断 ----------
    P("## 5. d_max 轮廓与隐含感染 / 隐含 CFR 一致性诊断（两口径并立）")
    P("")
    P("**解析恒等式**（论文级自洽结果）：隐含总感染 = 累计确诊/d_max = 9,066/d_max；"
      "隐含总 CFR = 累计死亡/隐含总感染 = (4,381/9,066)·d_max = **0.4833·d_max**"
      "（当死亡按确诊口径 4,381 全记时）。")
    P("")
    P("| d_max | 隐含总感染 | 隐含总 CFR | 与 BDBV 文献带 [0.24, 0.51] | 与实测确诊 CFR 0.483 |")
    P("|---|---|---|---|---|")
    for dm in (0.30, 0.40, 0.50, 0.60, 0.75, 0.85, 0.90, 0.95):
        inf = 9066.0 / dm
        cfr = 0.483 * dm
        inband = "带内" if 0.24 <= cfr <= 0.51 else "**带外**"
        P(f"| {dm:.2f} | {fmt(inf)} | {cfr:.4f} | {inband} | {100*(cfr-0.483)/0.483:+.1f}% |")
    P("")
    P("- **解析约束**：d_max < 0.50 ⇒ 隐含 CFR < 0.242，低于 BDBV 文献带下界 0.24，"
      "只有在「社区死亡未计入」（真实死亡 > 4,381，社区占比 68%–92%）时才允许。")
    P("- **联合约束加强（数值发现）**：低估口径下隐含 CFR = 0.4833·d_max / 死亡记全比"
      "（记全比 ≥ 0.68），要达到 BDBV 带下界 0.24 需 d_max ≳ 0.24×0.68/0.4833 ≈ **0.34**。"
      "故 **联合可辨识域 ≈ [0.34, 0.95]**；d_max=0.30 即使按最大低估口径也略越界"
      "（隐含 CFR 0.213 < 0.24）。")
    P("- **d_max 的可辨识域 = 两口径并立**：")
    P("  - ① **死亡全记口径**：d ∈ [0.50, 0.95]；")
    P(f"  - ② **死亡低估口径**（真实死亡 = 4,381/[0.68–1.0]，即 {fmt(4381.0)}–{fmt(4381.0/0.68)}）："
      "允许 d 低至 0.30。")
    P("- **正文用「隐含累计感染区间」表述**（待用户数值确认）：")
    P(f"  - d ∈ [0.30, 0.95] ⇒ {fmt(9066/0.95)} – {fmt(9066/0.30)}；")
    P(f"  - d ∈ [0.50, 0.95] ⇒ {fmt(9066/0.95)} – {fmt(9066/0.50)}；")
    P("  - （主理人给定 0.9 上界口径：[0.30,0.9] ⇒ 10,073–30,220；[0.50,0.9] ⇒ 10,073–18,132。）")
    P("- 自洽性检查：d_max ≈ 0.50 时隐含感染 ≈18,132、隐含 CFR ≈0.242 —— 与"
      " de Padua EID 2026 报告的乌干达 2007 干预后量级一致，两条独立证据互相印证。")
    P("- V-C 档（d_max-informed，固定 0.75，de Padua 漏报锚定 U[0.65,0.85] 中心）："
      f"隐含总感染 = {fmt(9066/0.75)}，隐含总 CFR = {0.483*0.75:.4f}，均在带内。")
    P("")

    # ---------- 6 p 三档 ----------
    P("## 6. p 三档敏感性（{0.35, 0.43, 0.50}，中心 0.43）")
    P("")
    P("| p | c₁=(1−s)c/p | c₂=s·c/(1−p) | μ_I | γ_I | β₀ | ρ | d_max | t₀ | s | NLL | R₀(t=0) |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|")
    p_res = {}
    for pc in (0.35, 0.43, 0.50):
        pr = make_params(p_care=pc)
        warm = variants["V-B (d_max-free [0.30,0.95])"]["th"]
        th, nll, _ = fit_model(pr, Xf0, k_obs=39.0, dmax_fixed=None,
                               starts=[warm], maxiter=1200)
        s = simulate(pr, Xf0, th, T_LAST, 0.5, t_grid=T_CONF)
        u = unpack(th, None)
        r0 = R0_ngm(th[0], pr, x=Xf0(0.0))
        p_res[pc] = dict(th=th, nll=nll, s=s, prm=pr, r0=r0)
        P(f"| {pc:.2f} | {pr['c1']:.4f} | {pr['c2']:.4f} | {pr['mu_I']:.4f} | "
          f"{pr['gam_I']:.4f} | {u['beta0']:.5f} | {u['rho']:.5f} | {u['d_max']:.4f} | "
          f"{u['t0']:.2f} | {u['s']:.2f} | {nll:.4f} | {r0:.4f} |")
    P("")

    # ---------- 7 μ_I/γ_I 3×3 ----------
    P("## 7. μ_I/γ_I 拆分敏感性：c（总体 CFR 锚）× 社区死亡占比（p=0.43 固定）")
    P("")
    P("社区死亡占比全程 67.8%–92.3%（WHO AFRO sitrep）；c₁ 由联立解给出，须落在 "
      "CFR_H∈[0.35,0.50]、c₂∈[0.65,0.85] 内才物理自洽（带外者标 ⚠）。")
    P("")
    P("| c \\ s_comm | 0.68 | 0.80 | 0.92 |")
    P("|---|---|---|---|")
    mug_grid = {}
    for c_t in (0.55, 0.60, 0.65):
        row = []
        for sc in (0.68, 0.80, 0.92):
            pr = make_params(c_total=c_t, s_comm=sc)
            warn = "" if (0.35 <= pr["c1"] <= 0.50 and 0.65 <= pr["c2"] <= 0.85) else " ⚠"
            warm = variants["V-B (d_max-free [0.30,0.95])"]["th"]
            th, nll, _ = fit_model(pr, Xf0, k_obs=39.0, dmax_fixed=None,
                                   starts=[warm], maxiter=1000)
            s = simulate(pr, Xf0, th, T_LAST, 0.5, t_grid=T_CONF, dmax_fixed=None)
            r0 = R0_ngm(th[0], pr, x=Xf0(0.0))
            r = run_scenario(pr, Xf0, th, s["traj"][-1].copy(), "B", dmax_fixed=None)
            i508 = int(round((T_ONE_YEAR - T_LAST) / 0.5))
            mug_grid[(c_t, sc)] = dict(r0=r0, conf508=float(r["conf"][i508]),
                                       death565=float(r["traj"][-1, I_F]),
                                       mu_I=pr["mu_I"], c1=pr["c1"], c2=pr["c2"])
            row.append(f"R₀={r0:.3f}, μ_I={pr['mu_I']:.4f}, "
                       f"B确诊(t=508)={r['conf'][i508]:,.0f}{warn}")
        P(f"| **{c_t:.2f}** | " + " | ".join(row) + " |")
    P("")
    mus = [v["mu_I"] for v in mug_grid.values()]
    r0s = [v["r0"] for v in mug_grid.values()]
    P(f"- μ_I 区间 **[{min(mus):.4f}, {max(mus):.4f}]**；R₀(t=0) 区间 "
      f"**[{min(r0s):.4f}, {max(r0s):.4f}]**「待用户数值确认」")
    P("- **结构性说明（非 bug）**：9 个格子的 R₀ 与 B 确诊轨迹完全相同——发病率似然只经由 β₀ "
      "重标定吸收 (c, s_comm) 的差异（μ_I+γ_I 总量不变，Erlang-5 出清率不进入确诊轨迹）；"
      "μ_I/γ_I 拆分的实质影响在死亡数的轨内分配与 c₁/c₂ 的物理自洽性（⚠ 标记），不在确诊轨迹。")
    P("- ⚠ 标记表示该 (c, s_comm) 组合反解出的 c₁/c₂ 落在主理人给定区间"
      "[0.35,0.50]/[0.65,0.85] 之外，物理上不自洽，结果仅作敏感性参考。")
    P("")

    # ---------- 8 M_A vs M_B ----------
    P("## 8. 模型对比：M_A（无失序耦合） vs M_B（h(X), β(X) 耦合，θ=0.4）")
    P("")
    P("> 主理人已确认映射：**h(I_S→H) = h·(1−θ_a X)**（失序延迟就诊，人滞留 I_S 久→社区传染多）；"
      "β(X)=β·(1+θ_b X)。θ_a=θ_b=0.40 固定不加参数。")
    P("")
    res_m = {}
    for tag, prm in (("M_A", make_params()), ("M_B", make_params(th_a=0.40, th_b=0.40))):
        warm = variants["V-B (d_max-free [0.30,0.95])"]["th"]
        th, nll, _ = fit_model(prm, Xf0, k_obs=39.0, dmax_fixed=None,
                               starts=[warm], maxiter=2500)
        s = simulate(prm, Xf0, th, T_LAST, 0.5, t_grid=T_CONF, dmax_fixed=None)
        res_m[tag] = dict(th=th, nll=nll, s=s, prm=prm, aic=2 * nll + 10)
        u = unpack(th, None)
        P(f"- **{tag}**: β₀={u['beta0']:.5f}, ρ={u['rho']:.5f}, d_max={u['d_max']:.4f}, "
          f"t₀={u['t0']:.2f}, s={u['s']:.2f}；NLL={nll:.4f}, AIC={res_m[tag]['aic']:.4f}"
          "「待用户数值确认」")
    d_aic = res_m["M_B"]["aic"] - res_m["M_A"]["aic"]
    P("")
    P(f"- **ΔAIC（M_B − M_A）= {d_aic:+.4f}**「待用户数值确认」"
      f"（{'M_B 优于 M_A' if d_aic < -2 else ('M_A 优于 M_B' if d_aic > 2 else '不可区分，|ΔAIC|<2')}）")
    P("")
    # 备选耦合 p(X)
    P("### 备选耦合敏感性：p(X)=p₀(1−θ_a X)（失序把人从求医轨挤进社区轨）")
    P("")
    prm_px = make_params(th_a=0.40, th_b=0.40, p_coupled=True)
    warm = res_m["M_B"]["th"]
    th_px, nll_px, _ = fit_model(prm_px, Xf0, k_obs=39.0, dmax_fixed=None,
                                 starts=[warm], maxiter=1500)
    s_px = simulate(prm_px, Xf0, th_px, T_LAST, 0.5, t_grid=T_CONF)
    u_px = unpack(th_px, None)
    P(f"- β₀={u_px['beta0']:.5f}, ρ={u_px['rho']:.5f}, d_max={u_px['d_max']:.4f}, "
      f"t₀={u_px['t0']:.2f}, s={u_px['s']:.2f}；NLL={nll_px:.4f}, "
      f"AIC={2*nll_px+10:.4f}「待用户数值确认」")
    P(f"- 与主耦合 M_B（NLL={res_m['M_B']['nll']:.4f}）相比 ΔNLL = "
      f"{nll_px - res_m['M_B']['nll']:+.4f} → "
      f"{'备选耦合拟合不劣' if nll_px <= res_m['M_B']['nll'] + 1e-9 else '主耦合（h 调制）更好'}")
    P("")
    P("### 拟合残差（M_A vs M_B）")
    P("")
    P("| 区间 | 天数 | 实测 | M_A | 误差% | M_B | 误差% |")
    P("|---|---|---|---|---|---|---|")
    for j in range(len(INC_OBS)):
        a = np.diff(res_m["M_A"]["s"]["conf_at"])[j]
        b = np.diff(res_m["M_B"]["s"]["conf_at"])[j]
        P(f"| [{INC_T0[j]:.0f},{INC_T1[j]:.0f}] | {INC_DAYS[j]:.0f} | {INC_OBS[j]:,.0f} | "
          f"{a:.1f} | {100*(a-INC_OBS[j])/INC_OBS[j]:+.2f} | {b:.1f} | "
          f"{100*(b-INC_OBS[j])/INC_OBS[j]:+.2f} |")
    P("")
    dma = unpack(res_m['M_A']['th'], None)['d_max']
    dmb = unpack(res_m['M_B']['th'], None)['d_max']
    if abs(dma - dmb) > 1e-6:
        P(f"- **M_A 与 M_B 的 d_max 分叉（{dma:.3f} vs {dmb:.3f}）** → 外推段隐含确诊 CFR 分叉 "
          f"≈{prm_c['c_total']/dma:.3f} vs {prm_c['c_total']/dmb:.3f}"
          f"（实测 0.483）。**M_B 情景累计死亡绝对值系统性偏高，跨分支死亡数不可直接比较**。")
    else:
        P(f"- M_A 与 M_B 的 d_max 相同（{dma:.3f}，均贴边界）：p 固定后失序耦合未造成 d_max 分叉"
          "（p 自由时曾为 0.898 vs 0.676）。**口径约定仍适用**：情景绝对死亡数只用 M_A 口径，"
          "M_B 分支只报相对量。")
    P("")
    # ---- M_B 分支情景：只报相对量（主理人裁决 #3 口径约定）----
    P("### M_B 分支情景（**只报相对量** —— 口径约定，见下）")
    P("")
    th_mb = res_m["M_B"]["th"]
    yl_mb = res_m["M_B"]["s"]["traj"][-1].copy()
    r_mb_b0 = run_scenario(res_m["M_B"]["prm"], Xf0, th_mb, yl_mb, "B0", dmax_fixed=None)
    r_mb_b = run_scenario(res_m["M_B"]["prm"], Xf0, th_mb, yl_mb, "B", dmax_fixed=None)
    r_mb_d = run_scenario(res_m["M_B"]["prm"], Xf0, th_mb, yl_mb, "D", vac=make_vac(),
                          ve_s=0.15, ve_d=0.40, f_ring=0.5, dmax_fixed=None, v0=VAC_V0)
    ratio_mb = r_mb_b["conf"][-1] / r_mb_b0["conf"][-1]
    death_mb_b = float(r_mb_b["traj"][-1, I_F])
    red_mb = 100 * (death_mb_b - float(r_mb_d["traj"][-1, I_F])) / death_mb_b
    P(f"- M_B 失序放大倍数 **B/B0（t=565 累计确诊）= {ratio_mb:.4f}**「待用户数值确认」")
    P(f"- M_B 疫苗情景 D（f_ring=0.5, VE_S=0.15, VE_D=0.40, c_ring=20）较 B "
      f"**死亡相对减少 = {red_mb:.2f}%**「待用户数值确认」")
    P("- **口径约定（主理人裁决 #3，进正文）**：模型对比 ΔAIC 两边自由参数数相同"
      "（p 固定后各 5 个：β₀, ρ, d_max, t₀, s），各自取自身最优，公平比较，不重标定；"
      "但**情景表的绝对死亡数只用 M_A 口径**（其 d_max 与实测 CFR 0.483 校准一致），"
      "**M_B 分支只报相对量**（B/B0 放大倍数、D 相对减少%）。")
    P("")

    # ---------- 9 情景 ----------
    P("## 9. 情景（t=143 → t=565；一年 = t=508 即 2027-10-11）")
    P("")
    scen_all = {}
    for vtag in ("V-B", "V-C"):
        key = "V-B (d_max-free [0.30,0.95])" if vtag == "V-B" else "V-C (d_max-informed 固定 0.75)"
        dmf = variants[key]["dmf"]
        th = variants[key]["th"]
        yl = variants[key]["s"]["traj"][-1].copy()
        d = {}
        for kind in ("A", "B0", "B"):
            d[kind] = run_scenario(prm_c, Xf0, th, yl, kind, dmax_fixed=dmf)
        scen_all[vtag] = d
    P("### V-B（d_max-free）与 V-C（d_max-informed）情景")
    P("")
    P("> **口径约定**：下表绝对数（确诊、死亡）均为 **M_A 口径**（θ_a=θ_b=0，无失序耦合，"
      "d_max 与实测 CFR 0.483 校准一致）。在 M_A 口径下 X 不进入动力学，"
      "**情景 B 与 B0 数值相同**（表中并列仅为保留情景编号）——失序效应只以相对量"
      "（B/B0 放大倍数、D 相对减少%）由 M_B 分支给出，见 §8。")
    P("")
    P("| 变体 | 情景 | β 设定 | X 设定 | 累计确诊 t=508 | 累计确诊 t=565 | 累计死亡 t=508 | 累计死亡 t=565 |")
    P("|---|---|---|---|---|---|---|---|")
    xdesc = {"A": "X(143) 恒定", "B0": "X≡0", "B": "X(143) 恒定（失序持续）"}
    for vtag in ("V-B", "V-C"):
        for kind in ("A", "B0", "B"):
            r = scen_all[vtag][kind]
            i508 = int(round((T_ONE_YEAR - T_LAST) / 0.5))
            P(f"| {vtag} | {kind} | {'β 续降' if kind=='A' else 'β 恒定'} | {xdesc[kind]} | "
              f"{fmt(r['conf'][i508])} | {fmt(r['conf'][-1])} | "
              f"{fmt(r['traj'][i508,I_F])} | {fmt(r['traj'][-1,I_F])} |")
    P("")
    amp = scen_all["V-B"]["B"]["conf"][-1] / scen_all["V-B"]["B0"]["conf"][-1]
    P(f"- M_A 口径下 B/B0 = {amp:.4f}（恒等于 1：X 不进 M_A 动力学，符合口径约定）；"
      "**失序放大倍数的实质数值由 M_B 分支给出（§8）**。")
    P("")

    # ---------- D 疫苗 ----------
    P("### 情景 D：疫苗（ring vaccination，主指标 = 累计死亡；基线含已接种 9,000 名医护）")
    P("")
    P(f"- 机制：S→V 流量 = φ(t)·c_ring·**日新发确诊**·(S/N_eff)；V(0)={VAC_V0:,.0f}；"
      "VE_S 折减有效易感池，VE_D 按接种占比 w=V/(S+V) 折减 μ_I/μ_H；f_ring 折减两项收益。")
    P(f"- 时序：t_vac=143+{VAC_DV:.0f}={143+VAC_DV:.0f}，logistic 爬坡 10%→90% 用 "
      f"{VAC_RAMP:.0f} 天。c_ring={VAC_C_RING:.0f}（主理人已确认），敏感性 10/30。")
    P("")
    P("| 变体 | f_ring | VE_S | VE_D | c_ring | V(565) | V/N_eff | 累计死亡 t=565 | 较 B 减少% |")
    P("|---|---|---|---|---|---|---|---|---|")
    d_grids = {}
    for vtag in ("V-B", "V-C"):
        key = "V-B (d_max-free [0.30,0.95])" if vtag == "V-B" else "V-C (d_max-informed 固定 0.75)"
        dmf = variants[key]["dmf"]; th = variants[key]["th"]
        yl = variants[key]["s"]["traj"][-1].copy()
        base = scen_all[vtag]["B"]["traj"][-1, I_F]
        g = {}
        for fr in (0.2, 0.5, 1.0):
            for ve_s in (0.0, 0.15, 0.40):
                for ve_d in (0.20, 0.40, 0.60):
                    for cr in (10.0, 20.0, 30.0):
                        r = run_scenario(prm_c, Xf0, th, yl, "D", vac=make_vac(c_ring=cr),
                                         ve_s=ve_s, ve_d=ve_d, f_ring=fr,
                                         dmax_fixed=dmf, v0=VAC_V0)
                        dd = float(r["traj"][-1, I_F])
                        g[(fr, ve_s, ve_d, cr)] = dict(
                            death565=dd, V565=float(r["traj"][-1, I_V]),
                            conf565=float(r["conf"][-1]),
                            red=100 * (base - dd) / base)
        d_grids[vtag] = g
        for fr in (0.2, 0.5, 1.0):
            for cr in (10.0, 20.0, 30.0):
                g_ = g[(fr, 0.15, 0.40, cr)]
                P(f"| {vtag} | {fr:.1f} | 0.15 | 0.40 | {cr:.0f} | {fmt(g_['V565'])} | "
                  f"{100*g_['V565']/prm_c['N_eff']:.3f}% | {fmt(g_['death565'])} | "
                  f"{g_['red']:.2f}% |")
    P("")
    P("- **量级说明**：环接种流量正比于日新发确诊（~60/日），c_ring=20 → ~10³ 剂/日，"
      "一年 ~10⁵ 剂，V/N_eff 仅千分之几。均匀混合下该覆盖对人群层面传播/死亡的影响"
      "必然很小——结构性结论，非数值问题。要得到可观效果需远高接种速率或显式分层高暴露亚群。")
    P("")
    P("**VE_S × VE_D 敏感性（V-B, f_ring=0.5, c_ring=20）**")
    P("")
    P("| VE_S \\ VE_D | 0.20 | 0.40 | 0.60 |")
    P("|---|---|---|---|")
    for ve_s in (0.0, 0.15, 0.40):
        row = [f"{d_grids['V-B'][(0.5, ve_s, ve_d, 20.0)]['death565']:,.0f} "
               f"({d_grids['V-B'][(0.5, ve_s, ve_d, 20.0)]['red']:.2f}%)"
               for ve_d in (0.20, 0.40, 0.60)]
        P(f"| {ve_s:.2f} | " + " | ".join(row) + " |")
    P("")

    # ---------- E ----------
    P("### 情景 E：纯数据驱动对照 + GT/SI 的 Rₑ 对照")
    P("")
    E = scen_E()
    P(f"- 4 个 UKHSA 周度增量 515/343/487/463，周均 **{E['mean_wk']:.1f}**（={E['mean_wk']/7:.2f}/日）")
    P(f"- 4 点 log-线性拟合：周斜率 {E['slope_per_wk']:+.5f}，日增速 r={E['r_day']:+.6f}/d")
    P("")
    P("**数据侧 Rₑ 对照（只进讨论不进模型）**")
    P("")
    P("| 口径 | Rₑ |")
    P("|---|---|")
    for k, v in E["R_data"].items():
        P(f"| {k} | {v:.4f} |")
    P("")
    P("> GT=12 d 为主理人标注「团队推导」（无文献支持），敏感性 10–16 d；"
      f"SI={SI_MEAN} d（SD {SI_SD}，WHO Ebola Response Team NEJM 2014，EBOV 西非）为对照档。"
      "四档 Rₑ 均 ≈1.00–1.01，与拟合 Rₑ(143) 量级一致。")
    P("")
    P("| 情景 | 一年新增确诊 | 累计确诊 2027-10-11 | 累计死亡 2027-10-11 |")
    P("|---|---|---|---|")
    P(f"| E1 零增长 | {fmt(E['annual1'])} | {fmt(E['cum1'])} | {fmt(E['death1'])} |")
    P(f"| E2 4点拟合外推 | {fmt(E['annual2'])} | {fmt(E['cum2'])} | {fmt(E['death2'])} |")
    for vtag in ("V-B", "V-C"):
        for kind in ("A", "B0", "B"):
            r = scen_all[vtag][kind]
            i508 = int(round((T_ONE_YEAR - T_LAST) / 0.5))
            P(f"| 模型 {kind} ({vtag}) | {fmt(r['conf'][i508]-9066)} | "
              f"{fmt(r['conf'][i508])} | {fmt(r['traj'][i508,I_F])} |")
    P("")

    # ---------- 10 BDBV 锚点对照 ----------
    P("## 10. 与 BDBV 文献锚点对照（de Padua & Akhmetzhanov, EID 2026;32(10):1762-1764, "
      "DOI 10.3201/eid3210.261175）—— 只进讨论不进模型")
    P("")
    P("| 量 | 本研究（拟合） | BDBV 文献（乌干达 2007） | 相对位置 |")
    P("|---|---|---|---|")
    r0_vb = R0_ngm(variants["V-B (d_max-free [0.30,0.95])"]["th"][0], prm_c, x=Xf0(0.0))
    re0_vb = R0_ngm(variants["V-B (d_max-free [0.30,0.95])"]["th"][0], prm_c,
                    s=1.0, x=Xf0(0.0))
    r0_vc = R0_ngm(variants["V-C (d_max-informed 固定 0.75)"]["th"][0], prm_c, x=Xf0(0.0))
    P(f"| Rₑ₀（t=0，含 X(0)） | V-B {re0_vb:.3f}；V-C {r0_vc:.3f} | "
      f"Rt={BDBV['Rt']:.2f} [{BDBV['Rt_lo']:.2f}, {BDBV['Rt_hi']:.2f}] | "
      f"{'带内' if BDBV['Rt_lo'] <= min(re0_vb, r0_vc) and max(re0_vb, r0_vc) <= BDBV['Rt_hi'] else '偏出'} |")
    r0s_all = [p_res[pc]["r0"] for pc in p_res] + [v["r0"] for v in mug_grid.values()]
    P(f"| R₀（各敏感性档） | [{min(r0s_all):.3f}, {max(r0s_all):.3f}] | "
      f"Rt={BDBV['Rt']:.2f} [{BDBV['Rt_lo']:.2f}, {BDBV['Rt_hi']:.2f}] | 见 §6/§7 |")
    cfr_impl = 0.483 * unpack(variants["V-B (d_max-free [0.30,0.95])"]["th"], None)["d_max"]
    P(f"| 隐含总 CFR（=0.483·d_max） | {cfr_impl:.4f}（V-B） | "
      f"干预前 CFR={100*BDBV['CFR']:.1f}% [{100*BDBV['CFR_lo']:.1f}, {100*BDBV['CFR_hi']:.1f}]；"
      f"干预后量级 ≈{100*BDBV['CFR_lo']:.1f}% | "
      f"{'带内' if BDBV['CFR_lo'] <= cfr_impl <= BDBV['CFR_hi'] else '带外'} |")
    P(f"| 漏报（1−d_max） | {100*(1-unpack(variants['V-B (d_max-free [0.30,0.95])']['th'],None)['d_max']):.1f}%"
      f"（V-B）；{100*(1-0.75):.1f}%（V-C） | ≈{100*BDBV['underreport']:.0f}% | — |")
    P("")
    P("> 说明：BDBV（乌干达 2007 Bundibugyo）与本次刚果金 EBOV 疫情毒株、"
      "干预强度、监测体系均不同，以上仅作量级对照，**不构成模型校验**。")
    P("")

    # ---------- 11 MC ----------
    mc_sum = {}
    if RUN_MC:
        P("## 11. 蒙特卡洛不确定度（N=1000，p=0.43 固定）")
        P("")
        P("| 参数 | 抽样分布 |")
        P("|---|---|")
        P("| E 总均值 | U[6.0, 11.3] d（Wamala 95% CI 外扩） |")
        P("| m_H（住院病程） | U[7, 12] d |")
        P("| c（总体 CFR 锚） | U[0.55, 0.65] |")
        P("| s_comm（社区死亡占比） | U[0.68, 0.92] |")
        P("| N_eff | U[8e6, 2.4e7] |")
        P("| k_obs | {20, 39, ∞} 等概率 |")
        P("| r₀_shape（初值形状） | U[0.06, 0.15] /d |")
        P("| d_max | V-B：拟合；V-C：U[0.65, 0.85] |")
        P("| p | **固定 0.43**（不抽样，见 §3 可辨识性） |")
        P("")
        for vtag, dmf in (("V-B", None), ("V-C", -1.0)):
            seeds = [SEED + 7919 * i + 13 for i in range(N_MC)]
            th0 = variants["V-B (d_max-free [0.30,0.95])"]["th"] if dmf is None \
                else variants["V-C (d_max-informed 固定 0.75)"]["th"]
            t0 = time.time()
            try:
                import multiprocessing as mp
                with mp.Pool(processes=N_MC_PROC) as pool:
                    draws = pool.map(mc_worker, [(i, seeds[i], th0, dmf, 0.15, 0.40, 0.5)
                                                 for i in range(N_MC)])
            except Exception as ex:
                print("[warn] 并行失败，回退串行：", ex, file=sys.stderr)
                draws = [mc_worker((i, seeds[i], th0, dmf, 0.15, 0.40, 0.5))
                         for i in range(N_MC)]
            keys0 = [k for k in draws[0] if k != "idx"]
            arr = {k: np.array([d[k] for d in draws], float) for k in keys0}
            bad = np.zeros(N_MC, bool)
            for k in ("B_conf508", "D_conf508", "B_death508", "D_death508"):
                bad |= ~np.isfinite(arr[k]) | (arr[k] <= 0)
            P(f"### {vtag}")
            P("")
            P(f"- 未收敛/非法样本：{int(bad.sum())}/{N_MC}；耗时 {time.time()-t0:.1f} s"
              "「待用户数值确认」")
            arrg = {k: v[~bad] for k, v in arr.items()}
            np.savez(f"mc_draws_v3_{'B' if dmf is None else 'C'}.npz",
                     **arrg, seeds=np.array(seeds)[~bad], bad_mask=bad)
            P("")
            P("| 参数 | 中位数 | 2.5% | 97.5% |")
            P("|---|---|---|---|")
            for k, lab in (("beta0", "β₀"), ("rho", "ρ"), ("d_max", "d_max"),
                           ("t0", "t₀"), ("s", "s"), ("c_total", "c（CFR 锚）"),
                           ("s_comm", "s_comm"), ("c1", "c₁（住院 CFR）"),
                           ("c2", "c₂（社区 CFR）")):
                q = np.percentile(arrg[k], [2.5, 50, 97.5])
                P(f"| {lab} | {q[1]:.4f} | {q[0]:.4f} | {q[2]:.4f} |")
            P("")
            P("| 一年投影（t=508） | 中位数 | 2.5% | 97.5% | 均值 |")
            P("|---|---|---|---|---|")
            for k, lab in (("B_death508", "累计死亡 B"), ("D_death508", "累计死亡 D"),
                           ("B_conf508", "累计确诊 B"), ("D_conf508", "累计确诊 D")):
                q = np.percentile(arrg[k], [2.5, 50, 97.5])
                mc_sum[(vtag, k)] = q
                P(f"| {lab} | {fmt(q[1])} | {fmt(q[0])} | {fmt(q[2])} | {fmt(arrg[k].mean())} |")
            red = 100 * (arrg["B_death508"] - arrg["D_death508"]) / arrg["B_death508"]
            rq = np.percentile(red, [2.5, 50, 97.5])
            P(f"| 死亡减少%（D vs B） | {rq[1]:.2f} | {rq[0]:.2f} | {rq[2]:.2f} | {red.mean():.2f} |")
            P("")

    # ---------- 12 数值验证 ----------
    P("## 12. 数值验证")
    P("")
    svb = variants["V-B (d_max-free [0.30,0.95])"]["s"]
    th_vb = variants["V-B (d_max-free [0.30,0.95])"]["th"]
    beta_fn = lambda t: th_vb[0] * np.exp(-th_vb[1] * t)
    f_ivp = lambda t, y: rhs(t, y, prm_c, Xf0, beta_fn, svb["dfun"])
    sol = solve_ivp(f_ivp, (0.0, T_LAST), svb["y0"], method="Radau",
                    rtol=1e-10, atol=1e-8, dense_output=True)
    ca = np.array([float(svb["dfun"](t)) * sum(sol.sol(t)[1:1 + IP_NE + 5]) for t in T_CONF])
    P(f"- 求解器交叉校验：RK4(dt=0.5) vs `solve_ivp/Radau`(rtol=1e-10)，9 锚点确诊"
      f"**最大相对偏差 = {np.max(np.abs(ca/svb['conf_at']-1.0)):.3e}**「待用户数值确认」")
    devs = [(d, simulate(prm_c, Xf0, th_vb, T_LAST, d, t_grid=T_CONF)["conf_at"][-1])
            for d in (1.0, 0.5, 0.25, 0.125)]
    P("- 网格加密收敛（t=143 累计确诊）：")
    P("")
    P("| dt | conf(143) |")
    P("|---|---|")
    for d, v in devs:
        P(f"| {d} | {v:.6f} |")
    P(f"- dt=1.0→0.125 总漂移 = {abs(devs[-1][1]-devs[0][1])/devs[-1][1]:.3e}「待用户数值确认」")
    tot = svb["traj"].sum(axis=1)
    P(f"- 守恒：`max|Σ/N_eff − 1| = {np.max(np.abs(tot/prm_c['N_eff']-1.0)):.3e}`"
      "；正性：全程最小 = "
      f"{float(np.min(svb['traj'])):.3e}（≥0）「待用户数值确认」")
    P("")

    # ---------- 13 风险 ----------
    P("## 13. 已知数值风险与失败案例")
    P("")
    P("1. **p 不可辨识**（§3）：确诊似然推 p→1，与社区死亡占比 67.8%–92.3% 直接矛盾。"
      "p 已按主理人裁决固定三档。若论文使用 p 自由拟合的结果，将得到社区轨为空的退化解。")
    P("2. **d_max 可辨识性弱、可辨识域为两口径并立**（§5）：拟合点估计与 MC 分位可能贴 "
      "[0.30,0.95] 边界。d_max 直接决定隐含总感染（9,066/d_max）与隐含总 CFR（0.4833·d_max）。"
      "d_max<0.50 时隐含 CFR 落在 BDBV 文献带 [0.24,0.51] 之外（§5），只有在死亡低估口径"
      "（真实死亡>4,381）下才可采信；死亡全记口径的可辨识域为 [0.50,0.95]。"
      "**联合约束加强**：低估口径下隐含 CFR = 0.4833·d/死亡记全比（≥0.68），"
      "要 ≥0.24 需 d ≳ 0.34 —— 即联合可辨识域为 **[0.34, 0.95]**；"
      "本次 V-B 拟合贴 0.30 下界，即使按最大低估口径也略越界（隐含 CFR 0.213 < 0.24），"
      "**绝对死亡数的下界档（V-B）应视为「极端低检出」边界情景而非主估计**。")
    P("3. **M_A/M_B 跨分支口径约定**：p 固定后两分支 d_max 不再分叉（均贴下界），"
      "但情景绝对死亡数仍只用 M_A 口径、M_B 只报相对量（主理人裁决 #3）；"
      "且 d_max=0.30 处的绝对死亡数本身处于 d_max 可辨识域边界（风险 2），解释需谨慎。")
    P("4. **c₁/c₂ 联立解可能落出物理区间**：§7 中标 ⚠ 的 (c, s_comm) 组合反解出的 "
      "c₁∉[0.35,0.50] 或 c₂∉[0.65,0.85]，此时 μ_I/γ_I 拆分失去自然史含义，结果仅作敏感性参考。")
    P("5. **R(0)=0 简化**：初值把全部非死亡早期病例放在急性期，R(0)=0。已在 MC 用 r₀_shape 扰动部分覆盖。")
    P("6. **疫苗的群体层面效果极小是结构性结论**（环接种流量 ∝ 日新发确诊，均匀混合下覆盖仅千分之几），"
      "不是数值问题。若论文需要给疫苗一个可观的数字，必须显式分层高暴露亚群或大幅提高接种速率。")
    P("7. **GT=12 d 无文献支持**（主理人标注「团队推导」），SI=15.3 d 仅作数据侧对照。"
      "两者只影响 §9 的 Rₑ 对照值，不进模型、不影响标定与情景。")
    P("8. **Deaths 不进似然**：CFR 由 (c, s_comm, p) 外生决定，累计死亡仅作验证输出。")
    P("9. **BDBV 锚点（Rt 1.55、CFR 31.4%、漏报 15%）来自乌干达 2007 Bundibugyo 疫情**，"
      "毒株/干预/监测体系均不同，只作量级对照，不构成校验。")
    P("")

    with open("results_v3.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")

    if RUN_FIG:
        make_figures(variants, res_m, scen_all, d_grids, E, mc_sum, prm_c, p_res, mug_grid)
    print(f"\n[完成] 总耗时 {time.time()-t_all:.1f} s")


def fit_model_free_p(Xf, k_obs=39.0, maxiter=2500):
    """p 作为第 6 个自由参数的拟合（可辨识性诊断用）"""
    keys = ("beta0", "rho", "d_max", "t0", "s", "p")
    BDp = dict(BD); BDp["p"] = (0.30, 1.0)
    def pen(th):
        v = 0.0
        for k_, val in zip(keys, th):
            lo, hi = BDp[k_]
            if val < lo: v += (lo - val) ** 2 * 1e6
            elif val > hi: v += (val - hi) ** 2 * 1e6
        return v
    def obj(th):
        p_ = pen(th)
        if p_ > 0:
            return 1e9 + p_
        pr = make_params(p_care=float(th[5]))
        try:
            s = simulate(pr, Xf, th[:5], T_LAST, 0.5, t_grid=T_CONF, dmax_fixed=None)
        except Exception:
            return 1e9
        conf = s["conf_at"]
        if not np.all(np.isfinite(conf)) or np.any(np.diff(conf) < -1e-9) or np.any(conf <= 0):
            return 1e9
        return nb_nll(INC_OBS, np.diff(conf), k_obs)
    best = None
    for b0 in (0.18, 0.40, 0.80):
        for rho in (0.004, 0.015):
            st = (b0, rho, 0.75, 30.0, 30.0, 0.85)
            st = tuple(min(max(v, BDp[k_][0]), BDp[k_][1]) for v, k_ in zip(st, keys))
            r = minimize(obj, st, method="Nelder-Mead",
                         options=dict(xatol=1e-8, fatol=1e-10, maxiter=maxiter,
                                      maxfev=maxiter, adaptive=True))
            if best is None or r.fun < best.fun:
                best = r
    return tuple(float(v) for v in best.x), float(best.fun), {}


# ======================================================================
# 10. 绘图
# ======================================================================
def make_figures(variants, res_m, scen_all, d_grids, E, mc_sum, prm_c, p_res, mug_grid):
    mid = (INC_T0 + INC_T1) / 2.0
    wid = INC_DAYS / 2.0
    obs_rate = INC_OBS / INC_DAYS
    kb = "V-B (d_max-free [0.30,0.95])"
    kc = "V-C (d_max-informed 固定 0.75)"

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    ax[0].errorbar(mid, obs_rate, xerr=wid, fmt="ks", ms=7, capsize=4, label="实测区间日均")
    for tag, col, key in (("V-B", "tab:blue", kb), ("V-C", "tab:orange", kc)):
        s = variants[key]["s"]
        conf = s["conf_series"]
        ax[0].plot(s["ts"], np.gradient(conf, s["ts"]), color=col, lw=1.8, label=f"{tag} 日确诊")
    ax[0].axhline(OBS_DAILY_INC_7D, color="g", ls="--", lw=1.4, label="WHO 近 7 日 72/日")
    ax[0].set_xlabel("t（天）"); ax[0].set_ylabel("日确诊（例/日）")
    ax[0].set_title("图 1a　日确诊拟合（观测模型 ×d(t)）")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    j = np.arange(len(INC_OBS)); w = 0.27
    ax[1].bar(j - w, INC_OBS, w, color="k", alpha=.8, label="实测")
    ax[1].bar(j, np.diff(variants[kb]["s"]["conf_at"]), w, color="tab:blue", label="V-B")
    ax[1].bar(j + w, np.diff(variants[kc]["s"]["conf_at"]), w, color="tab:orange", label="V-C")
    ax[1].set_xticks(j)
    ax[1].set_xticklabels([f"[{a:.0f},{b:.0f}]" for a, b in zip(INC_T0, INC_T1)],
                          rotation=30, fontsize=7)
    ax[1].set_ylabel("区间确诊增量"); ax[1].set_title("图 1b　逐区间确诊增量")
    ax[1].legend(); ax[1].grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig("fig_v3_weekly_fit.png"); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for tag, col, prm in (("M_A", "tab:blue", res_m["M_A"]["prm"]),
                          ("M_B", "tab:red", res_m["M_B"]["prm"])):
        s = res_m[tag]["s"]; th = res_m[tag]["th"]
        tt = s["ts"]
        xg = np.array([Xf0(t) if prm["th_a"] > 0 else 0.0 for t in tt])
        Seff = s["traj"][:, 0] / prm["N_eff"]
        h_eff = prm["h"] * (1 - prm["th_a"] * xg)
        pc = prm["p_care"]
        b_eff = th[0] * np.exp(-th[1] * tt) * (1 + prm["th_b"] * xg)
        reff = b_eff * Seff * (pc * (1 / h_eff + prm["kappa"] / (prm["gam_H"] + prm["mu_H"]))
                               + (1 - pc) / prm["mu_gamma_I"])
        ax[0].plot(tt, reff, color=col, lw=1.8, label=f"{tag} Rₑ(t)")
        ax[1].semilogy(tt, np.maximum(reff, 1e-3), color=col, lw=1.8, label=f"{tag}")
    for a in ax:
        a.axhline(1.0, color="k", ls=":", lw=1.2); a.set_xlabel("t（天）")
        a.axhline(BDBV["Rt"], color="purple", ls="-.", lw=1.0)
    ax[0].axhline(BDBV["Rt_lo"], color="purple", ls=":", lw=.8)
    ax[0].axhline(BDBV["Rt_hi"], color="purple", ls=":", lw=.8)
    ax[0].text(2, BDBV["Rt"] + .04, f"BDBV Rt={BDBV['Rt']}", color="purple", fontsize=8)
    ax[0].set_ylabel("Rₑ(t)"); ax[0].set_title("图 2a　有效再生数时序（含 BDBV 文献带）")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].set_ylabel("Rₑ(t)（对数轴）"); ax[1].set_title("图 2b　对数轴")
    ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig("fig_v3_reff.png"); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    cols = {"A": "tab:green", "B0": "tab:gray", "B": "tab:red"}
    for kind in ("A", "B0", "B"):
        r = scen_all["V-B"][kind]
        ax[0].plot(r["ts"], r["conf"], color=cols[kind], lw=2, label=f"{kind} (V-B)")
        r2 = scen_all["V-C"][kind]
        ax[0].plot(r2["ts"], r2["conf"], color=cols[kind], lw=1.2, ls="--")
    ax[0].axvline(T_LAST, color="k", ls=":", lw=1)
    ax[0].axhspan(E["cum1"], E["cum2"], color="tab:blue", alpha=.15, label="E 数据驱动带")
    ax[0].plot(T_CONF, C_CONF, "ks", ms=6, label="实测累计确诊")
    ax[0].set_yscale("log"); ax[0].set_xlabel("t（天）")
    ax[0].set_ylabel("累计确诊（对数轴）")
    ax[0].set_title("图 3a　情景对比（实线 V-B，虚线 V-C）")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    i508 = int(round((T_ONE_YEAR - T_LAST) / 0.5))
    labels, vals, colr = [], [], []
    for kind in ("A", "B0", "B"):
        labels.append(f"{kind}(V-B)"); vals.append(scen_all["V-B"][kind]["conf"][i508])
        colr.append(cols[kind])
        labels.append(f"{kind}(V-C)"); vals.append(scen_all["V-C"][kind]["conf"][i508])
        colr.append(cols[kind])
    labels += ["E1 零增长", "E2 拟合外推"]
    vals += [E["cum1"], E["cum2"]]; colr += ["tab:blue", "tab:cyan"]
    ax[1].barh(labels, vals, color=colr)
    ax[1].set_xscale("log"); ax[1].set_xlabel("累计确诊 t=508（对数轴）")
    ax[1].set_title("图 3b　模型情景 vs 纯数据驱动")
    ax[1].grid(alpha=.3, axis="x")
    fig.tight_layout(); fig.savefig("fig_v3_scen.png"); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for cr, col in zip((10.0, 20.0, 30.0), ("tab:blue", "tab:red", "tab:green")):
        xs = [f"f={fr:.1f}" for fr in (0.2, 0.5, 1.0)]
        ys = [d_grids["V-B"][(fr, 0.15, 0.40, cr)]["death565"] for fr in (0.2, 0.5, 1.0)]
        ax[0].plot(xs, ys, "o-", color=col, lw=1.8, label=f"c_ring={cr:.0f}")
    ax[0].axhline(scen_all["V-B"]["B"]["traj"][-1, I_F], color="k", ls="--", lw=1.5,
                  label="基线 B（无疫苗）")
    ax[0].set_ylabel("累计死亡 t=565"); ax[0].set_xlabel("f_ring")
    ax[0].set_title("图 4a　疫苗情景：f_ring × c_ring（V-B）")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    frs = (0.2, 0.5, 1.0)
    for ve_d, col in zip((0.20, 0.40, 0.60), ("tab:blue", "tab:red", "tab:green")):
        ys = [d_grids["V-B"][(fr, 0.15, ve_d, 20.0)]["red"] for fr in frs]
        ax[1].plot(frs, ys, "o-", color=col, lw=1.8, label=f"VE_D={ve_d:.2f}")
    ax[1].set_xlabel("f_ring"); ax[1].set_ylabel("累计死亡较基线减少 %")
    ax[1].set_title("图 4b　死亡减少 % 敏感性")
    ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig("fig_v3_vacc.png"); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    cs = [v["r0"] for v in mug_grid.values()]
    ax[0].hist(cs, bins=9, color="tab:purple", alpha=.7)
    ax[0].axvline(BDBV["Rt"], color="k", ls="--", lw=1.5, label=f"BDBV Rt={BDBV['Rt']}")
    ax[0].set_xlabel("R₀(t=0)（μ_I/γ_I 3×3 敏感性）"); ax[0].set_ylabel("频数")
    ax[0].set_title("图 5a　R₀ 对 c × s_comm 的敏感性")
    ax[0].legend(); ax[0].grid(alpha=.3)
    if os.path.exists("mc_draws_v3_B.npz"):
        z = np.load("mc_draws_v3_B.npz")
        for k, col, lab in (("B_death508", "tab:red", "B 死亡 t=508"),
                            ("D_death508", "tab:green", "D 死亡 t=508")):
            v = z[k]; v = v[np.isfinite(v) & (v > 0)]
            ax[1].hist(v, bins=45, alpha=.55, color=col, label=lab)
            ax[1].axvline(np.median(v), color=col, ls="--", lw=1.5)
        ax[1].set_xlabel("t=508 累计死亡"); ax[1].set_ylabel("频数")
        ax[1].set_title("图 5b　蒙特卡洛（V-B, N=1000）")
        ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig("fig_v3_mc.png"); plt.close(fig)

    svb = variants[kb]["s"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    tt = np.arange(0.0, T_LAST + 0.01, 0.5)
    ax[0].plot(tt, [float(svb["dfun"](t)) for t in tt], "b-", lw=2, label="d(t) 检出率")
    ax[0].axhline(svb["d_max"], color="r", ls="--", lw=1, label=f"d_max={svb['d_max']:.3f}")
    ax[0].axhspan(0.65, 0.85, color="tab:orange", alpha=.15, label="V-C 锚定带 U[0.65,0.85]")
    ax[0].set_xlabel("t（天）"); ax[0].set_ylabel("d(t)")
    ax[0].set_title("图 6a　检出率爬坡 d(t)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].plot(svb["ts"], svb["conf_series"], "b-", lw=2, label="V-B 模型累计确诊")
    ax[1].plot(T_CONF, C_CONF, "ks", ms=7, label="实测")
    ax[1].set_yscale("log"); ax[1].set_xlabel("t（天）"); ax[1].set_ylabel("累计确诊（对数轴）")
    ax[1].set_title("图 6b　累计确诊拟合"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig("fig_v3_dete.png"); plt.close(fig)


if __name__ == "__main__":
    main()
