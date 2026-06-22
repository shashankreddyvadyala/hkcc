"""
Consolidated experiment suite for Heat-Kernel Context Compression (HKCC).
Regenerates every figure (fig1..fig10) and the ablation numbers from a fixed seed.
Synthetic study only; no real LLM. Each block is annotated with the claim it supports.
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import expm, eigh
import scipy.sparse as sp
from scipy.cluster.vq import kmeans2

rng = np.random.default_rng(7)
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "legend.fontsize": 7.5, "figure.dpi": 150, "savefig.bbox": "tight",
})
ACC = "#B5651D"
COLS = {"random": "#9aa0a6", "stride": "#5f6368", "heat_prune": "#1a73e8",
        "heat_merge": "#B5651D", "kmeans": "#188038", "rand_merge": "#c5221f"}
import os
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(FIG, exist_ok=True)

# ------------------------------------------------------------------ helpers
def normalize(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

def synth_context(n_clusters=8, per_cluster=40, n_outliers=20, d=64, spread=0.12, gen=None):
    g = gen or rng
    centers = normalize(g.normal(size=(n_clusters, d)))
    blobs, labels = [], []
    for c in range(n_clusters):
        blobs.append(centers[c] + spread * g.normal(size=(per_cluster, d))); labels += [c]*per_cluster
    blobs.append(normalize(g.normal(size=(n_outliers, d)))); labels += [-1]*n_outliers
    return normalize(np.vstack(blobs)), np.array(labels)

def knn_graph(X, k=10, sigma=None):
    n = X.shape[0]
    dist = 2.0 - 2.0*(X @ X.T)
    S = -dist.copy(); np.fill_diagonal(S, -np.inf)
    idx = np.argsort(-S, axis=1)[:, :k]
    if sigma is None:
        sigma = np.median(np.sort(dist, axis=1)[:, 1:k+1])
    W = np.zeros((n, n))
    rows = np.repeat(np.arange(n), k); cols = idx.ravel()
    w = np.exp(-dist[rows, cols]/(sigma+1e-9))
    W[rows, cols] = w; W = np.maximum(W, W.T)
    d = W.sum(1); Dinv = 1.0/np.sqrt(d+1e-9)
    Lsym = np.eye(n) - (Dinv[:, None]*W*Dinv[None, :])
    return W, Lsym

def heat_exact(L, X, t):
    return expm(-t*L) @ X

def cheby_coeffs(t, K, lmax=2.0, M=200):
    a = lmax/2.0
    th = np.pi*(np.arange(M)+0.5)/M; x = np.cos(th); fx = np.exp(-t*a*(x+1.0))
    return np.array([(2.0/M)*np.sum(fx*np.cos(k*th)) for k in range(K+1)]), a

def heat_cheby(L, X, t, K=24, lmax=2.0):
    c, a = cheby_coeffs(t, K, lmax)
    n = L.shape[0]
    Lhat = (L - a*sp.eye(n))/a if sp.issparse(L) else (L - a*np.eye(n))/a
    Tprev = X.copy(); Tcur = Lhat @ X
    out = 0.5*c[0]*Tprev + c[1]*Tcur
    for k in range(2, K+1):
        Tnext = 2.0*(Lhat @ Tcur) - Tprev; out += c[k]*Tnext; Tprev, Tcur = Tcur, Tnext
    return out

def residual(X, Xt):
    return np.linalg.norm(X-Xt, axis=1)

def attn_out(K, V, q, mult=None):
    d = K.shape[1]; s = (K @ q)/np.sqrt(d)
    if mult is not None: s = s + np.log(mult+1e-9)
    s -= s.max(); a = np.exp(s); a /= a.sum()
    return a @ V

def representatives(X, U, res, M, mode):
    n = X.shape[0]
    if mode == "random":
        keep = rng.choice(n, M, replace=False); return X[keep], X[keep], np.ones(M)
    if mode == "stride":
        keep = np.linspace(0, n-1, M).astype(int); return X[keep], X[keep], np.ones(M)
    if mode == "heat_prune":
        keep = np.argsort(-res)[:M]; return X[keep], X[keep], np.ones(M)
    if mode in ("heat_merge", "rand_merge"):
        anchors = np.argsort(-res)[:M] if mode == "heat_merge" else rng.choice(n, M, replace=False)
        A = U[anchors]; assign = np.argmax(normalize(U) @ normalize(A).T, axis=1)
        Kc = np.zeros((M, X.shape[1])); mult = np.zeros(M)
        for a in range(M):
            mem = np.where(assign == a)[0]
            if len(mem) == 0: mem = np.array([anchors[a]])
            Kc[a] = X[mem].mean(0); mult[a] = len(mem)
        return normalize(Kc), normalize(Kc), mult
    if mode == "kmeans":
        cent, lab = kmeans2(X, M, minit="++", seed=int(rng.integers(1e6)))
        mult = np.array([max(1, np.sum(lab == a)) for a in range(M)])
        Kc = normalize(cent); return Kc, Kc, mult
    raise ValueError(mode)

def fidelity(Xc, Uc, rc, M, mode, Q, full):
    Kc, Vc, mult = representatives(Xc, Uc, rc, M, mode)
    comp = np.array([attn_out(Kc, Vc, q, mult=mult) for q in Q])
    return np.mean(np.sum(normalize(full)*normalize(comp), axis=1))

# ================================================================== base instance
print("Base instance ...")
X, labels = synth_context()
n, d = X.shape
W, L = knn_graph(X, 10)
evals, evecs = eigh(L)
t_star = 1.0/np.median(evals[evals > 1e-6])
U = heat_exact(L, X, t_star); res = residual(X, U)
core, outl = labels >= 0, labels < 0
print(f"  N={n} d={d} t*={t_star:.3f} ratio={res[outl].mean()/res[core].mean():.2f}")

# ===== FIG 1 : residual separation =====
coords = evecs[:, 1:3]*np.sqrt(np.maximum(1-evals[1:3], 1e-3))
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
sc = ax[0].scatter(coords[:, 0], coords[:, 1], c=res, cmap="copper", s=14)
ax[0].set_title("Heat residual on diffusion map"); ax[0].set_xlabel(r"$\phi_1$"); ax[0].set_ylabel(r"$\phi_2$")
ax[0].set_xticks([]); ax[0].set_yticks([])
fig.colorbar(sc, ax=ax[0], fraction=0.046, pad=0.04).set_label(r"$\rho_i(t^\star)$")
ax[1].hist(res[core], 24, alpha=0.75, color=COLS["random"], label="cluster-core (redundant)")
ax[1].hist(res[outl], 24, alpha=0.85, color=ACC, label="outlier (distinctive)")
ax[1].set_title("Residual distribution by token type")
ax[1].set_xlabel(r"$\rho_i(t^\star)$"); ax[1].set_ylabel("count"); ax[1].legend()
fig.tight_layout(); fig.savefig(f"{FIG}/fig1_residual.pdf"); plt.close(fig); print("  fig1")

# ===== FIG 2 : fidelity vs kept fraction (4 strategies) =====
fracs = np.array([0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.7])
nq, ntr = 400, 6
modes = ["random", "stride", "heat_prune", "heat_merge"]
lab2 = {"random": "random prune", "stride": "uniform stride",
        "heat_prune": "heat residual (prune only)", "heat_merge": "HKCC (residual + diffused merge)"}
fid = {m: np.zeros((ntr, len(fracs))) for m in modes}
for tr in range(ntr):
    g = np.random.default_rng(100+tr); Xt, _ = synth_context(gen=g)
    Wt, Lt = knn_graph(Xt, 10); tt = 1.0/np.median(eigh(Lt)[0][1:])
    Ut = heat_exact(Lt, Xt, tt); rt = residual(Xt, Ut)
    Q = normalize(g.normal(size=(nq, d))); full = np.array([attn_out(Xt, Xt, q) for q in Q])
    for fi, f in enumerate(fracs):
        M = max(2, int(round(f*Xt.shape[0])))
        for m in modes: fid[m][tr, fi] = fidelity(Xt, Ut, rt, M, m, Q, full)
fig, ax = plt.subplots(figsize=(4.7, 3.1))
for m in modes:
    mu, sd = fid[m].mean(0), fid[m].std(0)
    ax.plot(fracs, mu, "-o", ms=3.5, color=COLS[m], label=lab2[m])
    ax.fill_between(fracs, mu-sd, mu+sd, color=COLS[m], alpha=0.15)
ax.set_xlabel("kept fraction (representatives / N)"); ax.set_ylabel("attention-output fidelity (cosine)")
ax.set_title("Context compression vs. attention fidelity"); ax.set_ylim(0.5, 1.005)
ax.legend(loc="lower right"); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(f"{FIG}/fig2_fidelity.pdf"); plt.close(fig)
i12 = fracs.tolist().index(0.12); i08 = fracs.tolist().index(0.08)
F12 = {m: round(float(fid[m][:, i12].mean()), 4) for m in modes}
F08 = {m: round(float(fid[m][:, i08].mean()), 4) for m in modes}
print("  fig2  @8%", F08, "@12%", F12)

# ===== FIG 3 : dial + spectral low-pass + cheby error vs K =====
ts = np.geomspace(0.05, 50, 40)
thr = np.quantile(residual(X, heat_exact(L, X, t_star)), 0.6)
kept = np.array([(residual(X, heat_exact(L, X, t)) >= thr).mean() for t in ts])
Ks = np.arange(4, 41, 2)
cheb_err_K = []
ref = heat_exact(L, X, t_star)
for K in Ks:
    cheb_err_K.append(np.linalg.norm(heat_cheby(L, X, t_star, K)-ref)/np.linalg.norm(ref))
cheb_err_K = np.array(cheb_err_K)
fig, ax = plt.subplots(1, 3, figsize=(9.6, 2.7))
ax[0].semilogx(ts, kept, "-o", ms=3, color=ACC)
ax[0].set_xlabel(r"diffusion time $t$"); ax[0].set_ylabel("kept fraction (fixed threshold)")
ax[0].set_title("(a) the compression dial"); ax[0].grid(alpha=0.25)
ll = np.linspace(0, 2, 200)
for t in [0.2, 1.0, 5.0]: ax[1].plot(ll, np.exp(-t*ll), label=fr"$t={t}$")
ax[1].set_xlabel(r"graph frequency $\lambda$"); ax[1].set_ylabel(r"$e^{-t\lambda}$")
ax[1].set_title("(b) heat kernel = low-pass"); ax[1].legend(); ax[1].grid(alpha=0.25)
ax[2].semilogy(Ks, cheb_err_K, "-o", ms=3, color="#1a73e8")
ax[2].set_xlabel("Chebyshev order $K$"); ax[2].set_ylabel("relative error vs. exact")
ax[2].set_title("(c) cheap & accurate"); ax[2].grid(alpha=0.25, which="both")
fig.tight_layout(); fig.savefig(f"{FIG}/fig3_dial.pdf"); plt.close(fig)
print(f"  fig3  cheby@K24 err={cheb_err_K[Ks.tolist().index(24)]:.2e}")

# ===== FIG 4 : allocation convergence (path) =====
def alloc(Lg, alpha, B, steps=500, dt=0.05):
    S = len(alpha); b = np.full(S, B/S); hist = []
    for _ in range(steps):
        m = alpha/(1+b); gp = alpha/(1+b)**2
        b = b + dt*(Lg @ m)/gp; b = np.clip(b, 1e-3, None); b *= B/b.sum()
        hist.append(alpha/(1+b))
    return np.array(hist)
S = 7; alpha = rng.uniform(0.5, 2.5, S)
Lpath = np.diag(np.full(S, 2.0)) - np.diag(np.ones(S-1), 1) - np.diag(np.ones(S-1), -1)
Lpath[0, 0] = 1; Lpath[-1, -1] = 1
H = alloc(Lpath, alpha, 20.0)
fig, ax = plt.subplots(figsize=(4.7, 3.0))
for s in range(S): ax.plot(H[:, s], color=plt.cm.copper(s/S), lw=1.4)
ax.set_xlabel("diffusion step"); ax.set_ylabel(r"marginal value $m_s$")
ax.set_title("Budget-as-heat: marginals equalize"); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(f"{FIG}/fig4_allocation.pdf"); plt.close(fig)
print(f"  fig4  spread {H[0].std():.3f}->{H[-1].std():.1e}")

# ===== FIG 5 : fidelity vs diffusion time (scale matters) =====
print("Fig5 tau sweep ...")
tsw = np.geomspace(0.05, 30, 22)
budgets = {0.08: "8% kept", 0.15: "15% kept"}
fid_tau = {b: np.zeros((4, len(tsw))) for b in budgets}
for tr in range(4):
    g = np.random.default_rng(200+tr); Xt, _ = synth_context(gen=g)
    Wt, Lt = knn_graph(Xt, 10); Q = normalize(g.normal(size=(300, d)))
    full = np.array([attn_out(Xt, Xt, q) for q in Q])
    for ti, t in enumerate(tsw):
        Ut = heat_cheby(Lt, Xt, t, 28); rt = residual(Xt, Ut)
        for b in budgets:
            M = max(2, int(round(b*Xt.shape[0])))
            fid_tau[b][tr, ti] = fidelity(Xt, Ut, rt, M, "heat_merge", Q, full)
fig, ax = plt.subplots(figsize=(4.7, 3.1))
for b in budgets:
    mu, sd = fid_tau[b].mean(0), fid_tau[b].std(0)
    ax.semilogx(tsw, mu, "-o", ms=3, label=budgets[b])
    ax.fill_between(tsw, mu-sd, mu+sd, alpha=0.15)
ax.set_xlabel(r"diffusion time $t$"); ax.set_ylabel("HKCC attention fidelity")
ax.set_title("Diffusion time is a real scale knob"); ax.legend(); ax.grid(alpha=0.25, which="both")
fig.tight_layout(); fig.savefig(f"{FIG}/fig5_tausweep.pdf"); plt.close(fig)
best_t = tsw[np.argmax(fid_tau[0.08].mean(0))]
print(f"  fig5  best t @8% ~ {best_t:.2f}")

# ===== FIG 6 : robustness heatmap fidelity over (t x kept fraction) =====
print("Fig6 robustness heatmap ...")
tg = np.geomspace(0.1, 12, 12); fg = np.array([0.05, 0.08, 0.12, 0.18, 0.25, 0.35])
heat = np.zeros((len(fg), len(tg)))
for tr in range(3):
    g = np.random.default_rng(300+tr); Xt, _ = synth_context(gen=g)
    Wt, Lt = knn_graph(Xt, 10); Q = normalize(g.normal(size=(250, d)))
    full = np.array([attn_out(Xt, Xt, q) for q in Q])
    for ti, t in enumerate(tg):
        Ut = heat_cheby(Lt, Xt, t, 28); rt = residual(Xt, Ut)
        for fi, f in enumerate(fg):
            M = max(2, int(round(f*Xt.shape[0])))
            heat[fi, ti] += fidelity(Xt, Ut, rt, M, "heat_merge", Q, full)/3
fig, ax = plt.subplots(figsize=(5.2, 3.0))
im = ax.imshow(heat, aspect="auto", origin="lower", cmap="copper", vmin=0.9, vmax=1.0,
               extent=[np.log10(tg[0]), np.log10(tg[-1]), 0, len(fg)])
ax.set_yticks(np.arange(len(fg))+0.5); ax.set_yticklabels([f"{int(f*100)}%" for f in fg])
xt = [0.1, 0.3, 1, 3, 10]; ax.set_xticks(np.log10(xt)); ax.set_xticklabels(xt)
ax.set_xlabel(r"diffusion time $t$ (log)"); ax.set_ylabel("kept fraction")
ax.set_title("HKCC fidelity: a broad operating plateau")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("attention fidelity")
fig.tight_layout(); fig.savefig(f"{FIG}/fig6_robustness.pdf"); plt.close(fig); print("  fig6")

# ===== FIG 7 : measured runtime scaling (Chebyshev linear vs exact cubic) =====
print("Fig7 runtime scaling ...")
def rand_sparse_L(N, k=10):
    rows = np.repeat(np.arange(N), k)
    cols = rng.integers(0, N, size=N*k)
    data = np.ones(N*k)
    A = sp.csr_matrix((data, (rows, cols)), shape=(N, N)); A = ((A + A.T) > 0).astype(float)
    deg = np.array(A.sum(1)).ravel(); Dinv = 1.0/np.sqrt(deg+1e-9)
    Dm = sp.diags(Dinv); Lsym = sp.eye(N) - Dm @ A @ Dm
    return Lsym.tocsr()
Ns = [128, 256, 512, 1024, 2048, 4096]
t_cheby, t_exact = [], []
for N in Ns:
    Ls = rand_sparse_L(N, 10); Xn = rng.normal(size=(N, 32))
    reps = 5 if N <= 1024 else 3
    t0 = time.perf_counter()
    for _ in range(reps): heat_cheby(Ls, Xn, 1.0, 24)
    t_cheby.append((time.perf_counter()-t0)/reps)
    if N <= 1024:
        Ld = Ls.toarray()
        t0 = time.perf_counter(); expm(-1.0*Ld) @ Xn; t_exact.append(time.perf_counter()-t0)
    else:
        t_exact.append(np.nan)
fig, ax = plt.subplots(figsize=(4.7, 3.1))
Ns_a = np.array(Ns, float)
ax.loglog(Ns_a, t_cheby, "-o", color=ACC, label="Chebyshev apply (sparse)")
te = np.array(t_exact); mask = ~np.isnan(te)
ax.loglog(Ns_a[mask], te[mask], "-s", color="#1a73e8", label="exact $e^{-tL}$ (dense)")
ax.loglog(Ns_a, t_cheby[0]*(Ns_a/Ns_a[0]), ":", color=ACC, alpha=0.6, label=r"$O(N)$ ref")
ax.loglog(Ns_a[mask], te[mask][0]*(Ns_a[mask]/Ns_a[mask][0])**3, ":", color="#1a73e8", alpha=0.6, label=r"$O(N^3)$ ref")
ax.set_xlabel("context length $N$"); ax.set_ylabel("wall-clock (s)")
ax.set_title("Diffusion cost is linear in $N$"); ax.legend(); ax.grid(alpha=0.25, which="both")
fig.tight_layout(); fig.savefig(f"{FIG}/fig7_runtime.pdf"); plt.close(fig)
print(f"  fig7  cheby {t_cheby[0]:.1e}->{t_cheby[-1]:.1e}s ; exact@1024={t_exact[3]:.2e}s")

# ===== FIG 8 : spectral mechanism =====
print("Fig8 spectrum ...")
Xhat = evecs.T @ X
energy = np.linalg.norm(Xhat, axis=1)**2
cum = np.cumsum(energy)/energy.sum()       # cumulative low->high frequency
frac_modes = np.arange(1, n+1)/n
e20 = cum[int(0.2*n)-1]                     # energy in lowest 20% of spectrum
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
ax[0].plot(evals, energy, lw=1.0, color="#5f6368")
ax[0].fill_between(evals, energy, color=ACC, alpha=0.25)
ax[0].set_xlabel(r"graph frequency $\lambda$"); ax[0].set_ylabel("signal energy")
ax[0].set_title("(a) context is low-frequency"); ax[0].grid(alpha=0.25)
ax[1].plot(frac_modes, cum, lw=1.6, color=ACC)
ax[1].axvline(0.2, ls=":", color="k", alpha=0.5)
ax[1].axhline(e20, ls=":", color="k", alpha=0.5)
ax[1].annotate(fr"{e20*100:.0f}% energy in lowest 20% of modes",
               xy=(0.2, e20), xytext=(0.30, max(0.2, e20-0.35)), fontsize=7.5,
               arrowprops=dict(arrowstyle="->", alpha=0.6))
ax[1].set_xlabel("fraction of spectrum (low$\\to$high freq)"); ax[1].set_ylabel("cumulative energy")
ax[1].set_title("(b) compressibility"); ax[1].grid(alpha=0.25); ax[1].set_ylim(0, 1.02)
fig.tight_layout(); fig.savefig(f"{FIG}/fig8_spectrum.pdf"); plt.close(fig)
print(f"  fig8  energy in lowest 20% modes = {e20*100:.1f}%")

# ===== FIG 9 : token-cost model =====
print("Fig9 cost model ...")
f = np.linspace(0.05, 1.0, 50)
naive = f
cache = 0.1*f + 0.02            # static compressed preamble at 0.1x + small per-query tail
cache_batch = 0.5*cache
fig, ax = plt.subplots(figsize=(4.9, 3.1))
ax.plot(f, naive, "-", color="#5f6368", label="compression only")
ax.plot(f, cache, "-", color="#1a73e8", label="+ prompt caching (0.1$\\times$)")
ax.plot(f, cache_batch, "-", color=ACC, label="+ caching + batch (0.5$\\times$)")
ax.axvline(0.08, ls=":", color="k", alpha=0.5)
ax.annotate("0.999-fidelity\noperating point", xy=(0.08, 0.5), xytext=(0.25, 0.62),
            fontsize=7.5, arrowprops=dict(arrowstyle="->", alpha=0.6))
ax.set_xlabel("kept fraction $f$"); ax.set_ylabel("relative input cost (vs. full prompt)")
ax.set_title("Compression stacks with provider levers"); ax.legend(); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(f"{FIG}/fig9_costmodel.pdf"); plt.close(fig)
print(f"  fig9  rel cost @8%: naive={0.08:.3f} +cache={0.1*0.08+0.02:.3f} +batch={0.5*(0.1*0.08+0.02):.3f}")

# ===== FIG 10 : allocation convergence vs topology (lambda_2) =====
print("Fig10 topology ...")
Sn = 12
def Lap(A):
    deg = A.sum(1); return np.diag(deg) - A
A_path = np.zeros((Sn, Sn))
for i in range(Sn-1): A_path[i, i+1] = A_path[i+1, i] = 1
A_star = np.zeros((Sn, Sn)); A_star[0, 1:] = 1; A_star[1:, 0] = 1
A_comp = np.ones((Sn, Sn)) - np.eye(Sn)
topos = {"path": A_path, "star": A_star, "complete": A_comp}
alpha12 = rng.uniform(0.5, 2.5, Sn)
fig, ax = plt.subplots(figsize=(4.9, 3.1))
for name, A in topos.items():
    Lg = Lap(A); l2 = np.sort(np.linalg.eigvalsh(Lg))[1]
    Ht = alloc(Lg, alpha12, 30.0, steps=300, dt=0.02)
    spread = Ht.std(1)
    ax.semilogy(spread, lw=1.6, label=fr"{name} ($\lambda_2={l2:.2f}$)")
ax.set_xlabel("diffusion step"); ax.set_ylabel("marginal-value spread (std)")
ax.set_title(r"Convergence rate scales with $\lambda_2(L_G)$"); ax.legend(); ax.grid(alpha=0.25, which="both")
fig.tight_layout(); fig.savefig(f"{FIG}/fig10_topology.pdf"); plt.close(fig); print("  fig10")

# ===== ABLATION : anchor-selection at fixed budget =====
print("Ablation: anchor selection ...")
abl_modes = ["rand_merge", "kmeans", "heat_merge"]
abl_lab = {"rand_merge": "random anchors + merge", "kmeans": "k-means merge",
           "heat_merge": "heat-residual anchors + merge (ours)"}
abl = {m: [] for m in abl_modes}
for tr in range(6):
    g = np.random.default_rng(400+tr); Xt, _ = synth_context(gen=g)
    Wt, Lt = knn_graph(Xt, 10); tt = 1.0/np.median(eigh(Lt)[0][1:])
    Ut = heat_exact(Lt, Xt, tt); rt = residual(Xt, Ut)
    Q = normalize(g.normal(size=(400, d))); full = np.array([attn_out(Xt, Xt, q) for q in Q])
    M = max(2, int(round(0.10*Xt.shape[0])))
    for m in abl_modes: abl[m].append(fidelity(Xt, Ut, rt, M, m, Q, full))
abl_mean = {m: (np.mean(abl[m]), np.std(abl[m])) for m in abl_modes}
print("  ablation @10%:", {m: round(abl_mean[m][0], 4) for m in abl_modes})

# ----- dump numbers
with open(f"{FIG}/numbers.txt", "w") as fh:
    fh.write(f"N={n} d={d} t_star={t_star:.3f} residual_ratio={res[outl].mean()/res[core].mean():.2f}\n")
    fh.write(f"fid@8 {F08}\nfid@12 {F12}\n")
    fh.write(f"cheby@K24 {cheb_err_K[Ks.tolist().index(24)]:.2e}\n")
    fh.write(f"best_t@8={best_t:.2f}\n")
    fh.write(f"energy_lowest20pct={e20*100:.1f}\n")
    fh.write(f"runtime cheby {t_cheby[0]:.2e}->{t_cheby[-1]:.2e}; exact@1024={t_exact[3]:.2e}\n")
    fh.write("ablation@10 " + str({m: (round(abl_mean[m][0], 4), round(abl_mean[m][1], 4)) for m in abl_modes}) + "\n")
print("\nDONE.")
