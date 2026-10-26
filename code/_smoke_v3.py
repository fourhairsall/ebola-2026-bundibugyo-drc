# -*- coding: utf-8 -*-
"""烟测：build_ic 修复验证 + H1'/H2' 通路 + 一次完整拟合"""
import numpy as np
import ebola_model_v3 as m

prm = m.make_params()
y0 = m.build_ic(0.0953, prm, m.Xf0)
print("state size:", y0.size, "(expect 12)")
print("E1..E5:", np.round(y0[1:6], 2))
print("IS, IC, H:", round(y0[6], 2), round(y0[7], 2), round(y0[8], 2))
print("R, F, V:", round(y0[9], 3), round(y0[10], 3), round(y0[11], 3))
print("S + rest =", round(y0[0], 2), "  sum =", round(y0.sum(), 2), " N_eff =", prm["N_eff"])
d0 = 83.0 / y0[10]  # conf0/F0 consistency: cum_inf(0)/F(0) should = 83/9
print("cum_inf(0) =", round(float(m.cum_inf_of(y0[None, :])[0]), 2), " (expect 83/0.0953=871)")
print("F(0) =", round(y0[10], 2), " (expect 9/0.0953=94.4)")
assert y0[7] > 0, "I_C still zero!"
assert y0.size == 12

# H1 ramp 路径
th5 = (0.30, 0.010, 0.70, 30.0, 25.0)
s1 = m.simulate(prm, m.Xf0, th5, m.T_LAST, 0.5, t_grid=m.T_CONF)
print("H1 conf(143) =", round(float(s1["conf_at"][-1]), 1))
# H2 无爬坡 3 参数路径
th3 = (0.30, 0.010, 0.70)
s2 = m.simulate(prm, m.Xf0, th3, m.T_LAST, 0.5, t_grid=m.T_CONF, ramp=False)
print("H2 conf(143) =", round(float(s2["conf_at"][-1]), 1))
n1 = m.neg_ll(th5, prm, m.Xf0)
n2 = m.neg_ll(th3, prm, m.Xf0, ramp=False)
print("NLL H1(start) =", round(n1, 3), "  NLL H2(start) =", round(n2, 3))

# V-C 固定 d_max 4 参数路径
th4 = (0.30, 0.010, 30.0, 25.0)
s3 = m.simulate(prm, m.Xf0, th4, m.T_LAST, 0.5, t_grid=m.T_CONF, dmax_fixed=0.75)
print("V-C conf(143) =", round(float(s3["conf_at"][-1]), 1))

# 短拟合冒烟（H2, 单起点, 少迭代）
from scipy.optimize import minimize
r = minimize(m.neg_ll, th3, args=(prm, m.Xf0, 39.0, None, False), method="Nelder-Mead",
             options=dict(maxiter=60, maxfev=60))
print("H2 短拟合 NLL:", round(float(r.fun), 3), " x =", np.round(r.x, 4))
print("SMOKE OK")
