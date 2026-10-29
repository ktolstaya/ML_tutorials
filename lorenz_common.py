"""
Общий модуль сквозного примера «Аттрактор Лоренца» для дополнительных примеров
к пособию «Нейронные сети для инженеров».

Используется блокнотами:
  ch03_Lorenz_SINDy.ipynb   — глава 3: идентификация методом разреженной регрессии (SINDy)
  KAN_Lorenz_attarctor.ipynb — глава 4: та же задача с сетью Колмогорова–Арнольда
  (планируется) глава 11: обратная задача для PINN

Модуль задаёт единые для всех глав параметры системы, данные, сиды, оценки
производных и метрики, чтобы результаты разных методов можно было сравнивать
напрямую. Требуется только NumPy; Matplotlib нужен лишь для setup_style().

Соглашение, как в главах 2–4: каждая строка массива состояний X (N x 3) — одно
наблюдение (x, y, z) в момент времени t.

Содержимое:
  SIGMA, RHO, BETA, DT, LYAP     — параметры системы, шаг дискретизации, показатель Ляпунова
  lorenz_rhs, rk4, simulate      — правые части и интегрирование методом Рунге–Кутты 4-го порядка
  ensure_data, load              — общие наборы данных (создаются при первом вызове)
  box_points                     — точки, равномерно покрывающие область (набор B главы 4)
  add_noise                      — относительный гауссов шум измерений
  central_diff, savgol_diff      — оценки производных по отсчётам
  window_integrals               — интегральная постановка: квадратуры вместо производных
  library                        — полиномиальная библиотека признаков Θ(X)
  stlsq                          — последовательная пороговая регрессия (SINDy)
  model_rhs                      — правая часть по матрице коэффициентов
  true_coefficients              — эталонная матрица коэффициентов
  lyapunov_max                   — старший показатель Ляпунова (метод Бенеттина)
  attractor_scale, valid_time    — масштаб аттрактора и время достоверного прогноза
  lorenz_map                     — последовательные максимумы z (статистика аттрактора)
  setup_style                    — единый чёрно-белый стиль рисунков пособия
"""
from itertools import combinations_with_replacement
import pathlib
import numpy as np

# ---------------------------------------------------------------------------
# Параметры, общие для всех глав
# ---------------------------------------------------------------------------
SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0   # классические параметры Лоренца
DT = 0.01                                   # шаг дискретизации наблюдений
T_TRANSIENT = 10.0                          # отбрасываемый переходный участок
LYAP = 0.9056                               # старший показатель Ляпунова (литературное значение)
SEED = 20261017                             # сид данных
NOISE_SEED = 314159                         # сид шума измерений
X0 = {"train": (-8.0, 7.0, 27.0), "val": (1.0, 1.0, 1.0), "test": (5.0, -5.0, 20.0)}
T_LEN = {"train": 50.0, "val": 20.0, "test": 30.0}
BOX = np.array([[-20.0, 20.0], [-25.0, 25.0], [0.0, 50.0]])   # область для box_points и переходов
DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"
VARS = ("x", "y", "z")


def lorenz_rhs(X, sigma=SIGMA, rho=RHO, beta=BETA):
    """Правые части системы Лоренца; X: (..., 3)."""
    x, y, z = X[..., 0], X[..., 1], X[..., 2]
    return np.stack([sigma * (y - x), x * (rho - z) - y, x * y - beta * z], axis=-1)


def rk4(f, x0, dt, n_steps):
    """Метод Рунге–Кутты 4-го порядка. x0: (3,) или (K, 3). Возвращает (n_steps+1, ..., 3)."""
    X = np.empty((n_steps + 1,) + np.shape(x0))
    X[0] = x0
    with np.errstate(all="ignore"):
        for k in range(n_steps):
            s = X[k]
            k1 = f(s); k2 = f(s + 0.5 * dt * k1); k3 = f(s + 0.5 * dt * k2); k4 = f(s + dt * k3)
            X[k + 1] = s + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            bad = ~np.all(np.isfinite(X[k + 1]) & (np.abs(X[k + 1]) < 1e6), axis=-1)
            if np.any(bad):                            # траектория модели «ушла на бесконечность»
                X[k + 1][bad] = np.nan                 # остальные траектории считаются дальше
                if np.all(bad):
                    X[k + 1:] = np.nan
                    break
    return X


def simulate(x0, t_len, dt=DT, transient=T_TRANSIENT, f=lorenz_rhs, substeps=10):
    """Траектория с отброшенным переходным участком.
    Интегрирование идёт с шагом dt/substeps, наблюдения сохраняются с шагом dt."""
    h = dt / substeps
    n_tr = int(round(transient / h))
    if n_tr:
        x0 = rk4(f, np.asarray(x0, float), h, n_tr)[-1]
    n = int(round(t_len / dt))
    X = rk4(f, x0, h, n * substeps)[::substeps]
    return np.arange(n + 1) * dt, X


# ---------------------------------------------------------------------------
# Общие данные
# ---------------------------------------------------------------------------
def _transients(n_traj=40, t_len=1.0, seed=SEED + 1):
    rng = np.random.default_rng(seed)
    x0 = BOX[:, 0] + (BOX[:, 1] - BOX[:, 0]) * rng.uniform(size=(n_traj, 3))
    t, X = simulate(x0, t_len, transient=0.0)          # (n+1, K, 3) — без отбрасывания: нужен именно переход
    rows = []
    for k in range(n_traj):
        rows.append(np.column_stack([np.full(t.size, k), t, X[:, k], lorenz_rhs(X[:, k])]))
    return np.vstack(rows)


def ensure_data(force=False):
    """Создаёт файлы data/lorenz_{train,val,test}.csv и data/lorenz_transients.csv.
    Столбцы: t, x, y, z, dx, dy, dz (точные производные — для контроля, а не для обучения).
    В lorenz_transients.csv первым идёт номер траектории."""
    DATA_DIR.mkdir(exist_ok=True)
    hdr = "t,x,y,z,dx,dy,dz"
    for name in ("train", "val", "test"):
        f = DATA_DIR / f"lorenz_{name}.csv"
        if force or not f.exists():
            t, X = simulate(X0[name], T_LEN[name])
            np.savetxt(f, np.column_stack([t, X, lorenz_rhs(X)]), delimiter=",",
                       fmt="%.17g", header=hdr, comments="")
    f = DATA_DIR / "lorenz_transients.csv"
    if force or not f.exists():
        np.savetxt(f, _transients(), delimiter=",", fmt="%.17g", header="traj," + hdr, comments="")
    return DATA_DIR


def load(name):
    """Возвращает (t, X, dX_true) для 'train', 'val', 'test'
    или (traj, t, X, dX_true) для 'transients'."""
    ensure_data()
    a = np.loadtxt(DATA_DIR / f"lorenz_{name}.csv", delimiter=",", skiprows=1)
    if name == "transients":
        return a[:, 0].astype(int), a[:, 1], a[:, 2:5], a[:, 5:8]
    return a[:, 0], a[:, 1:4], a[:, 4:7]


def box_points(n, seed=SEED + 2):
    """Точки, равномерно покрывающие область BOX, и точные скорости в них (набор B главы 4)."""
    rng = np.random.default_rng(seed)
    X = BOX[:, 0] + (BOX[:, 1] - BOX[:, 0]) * rng.uniform(size=(n, 3))
    return X, lorenz_rhs(X)


def add_noise(X, eta, seed=NOISE_SEED):
    """Аддитивный гауссов шум с СКО eta * std(X_i) по каждой координате."""
    rng = np.random.default_rng(seed)
    return X + eta * X.std(axis=0) * rng.standard_normal(X.shape)


# ---------------------------------------------------------------------------
# Производные по отсчётам
# ---------------------------------------------------------------------------
def central_diff(X, dt=DT):
    """Центральные разности второго порядка; возвращает (X[1:-1], dX)."""
    return X[1:-1], (X[2:] - X[:-2]) / (2 * dt)


def savgol_weights(half, order, dt=DT):
    """Веса фильтра Савицкого–Голея: локальный МНК-полином степени order
    на окне из 2*half+1 точек. Возвращает веса значения и первой производной в центре."""
    s = np.arange(-half, half + 1) * dt
    V = np.vander(s, order + 1, increasing=True)        # базис 1, s, s^2, ...
    Pinv = np.linalg.pinv(V)                             # коэффициенты полинома = Pinv @ y
    return Pinv[0], Pinv[1]                              # c0 = значение, c1 = производная


def savgol_diff(X, half=7, order=3, dt=DT):
    """Сглаженные состояния и производные; края (по half точек) отбрасываются."""
    w0, w1 = savgol_weights(half, order, dt)
    n = X.shape[0] - 2 * half
    idx = np.arange(n)[:, None] + np.arange(2 * half + 1)[None, :]
    win = X[idx]                                         # (n, 2h+1, 3)
    return np.einsum("k,nkd->nd", w0, win), np.einsum("k,nkd->nd", w1, win)


def window_integrals(Theta, X, m, rule="simpson", dt=DT):
    """Интегральная (квадратурная) постановка SINDy: x(t_k+m) - x(t_k) = ∫ Θ(x) dt · Ξ.
    Интеграл по окну из m шагов вычисляется квадратурой (trapezoid или simpson, m чётное).
    Возвращает (интегралы признаков, приращения состояния) по перекрывающимся окнам."""
    if rule == "trapezoid":
        w = np.ones(m + 1); w[[0, -1]] = 0.5; w *= dt
    elif rule == "simpson":
        assert m % 2 == 0, "для формулы Симпсона число шагов m должно быть чётным"
        w = np.ones(m + 1); w[1:-1:2] = 4; w[2:-1:2] = 2; w *= dt / 3
    else:
        raise ValueError(rule)
    win = np.lib.stride_tricks.sliding_window_view(Theta, m + 1, axis=0)   # (N-m, F, m+1)
    return win @ w, X[m:] - X[:-m]


# ---------------------------------------------------------------------------
# Библиотека признаков и SINDy
# ---------------------------------------------------------------------------
def library(X, degree=3):
    """Мономы от x, y, z до степени degree. Возвращает (Θ, имена в LaTeX, показатели)."""
    exps = []
    for d in range(degree + 1):
        for comb in combinations_with_replacement(range(3), d):
            e = [0, 0, 0]
            for c in comb:
                e[c] += 1
            exps.append(tuple(e))
    names = []
    for e in exps:
        s = "".join(v + (f"^{p}" if p > 1 else "") for v, p in zip(VARS, e) if p)
        names.append(s if s else "1")
    E = np.array(exps)
    Theta = np.prod(X[:, None, :] ** E[None, :, :], axis=2)
    return Theta, names, E


def ridge_lstsq(A, b, lam, free=(0,)):
    """min ||A c - b||^2 + lam * sum_{k not in free} c_k^2  (формула 3.7)."""
    if lam == 0:
        return np.linalg.lstsq(A, b, rcond=None)[0]
    D = np.sqrt(lam) * np.eye(A.shape[1])
    for k in free:
        D[k, k] = 0.0
    return np.linalg.lstsq(np.vstack([A, D]), np.vstack([b, np.zeros((A.shape[1], b.shape[1]))]),
                           rcond=None)[0]


def stlsq(Theta, dX, threshold, ridge=0.0, max_iter=25):
    """Последовательная пороговая регрессия (STLSQ, алгоритм SINDy).
    Столбцы Θ нормируются на свои среднеквадратичные значения s_k; член с относительным
    вкладом |ξ_k|·s_k / s(ẋ_i) < threshold обнуляется, остальные пересчитываются МНК.
    Возвращает Ξ в исходных единицах (n_features x 3)."""
    s = np.sqrt(np.mean(Theta ** 2, axis=0)); s[s == 0] = 1.0   # среднеквадратичные нормы столбцов
    A = Theta / s
    sy = np.sqrt(np.mean(dX ** 2, axis=0))
    Xi = ridge_lstsq(A, dX, ridge)
    active = np.ones_like(Xi, bool)
    for _ in range(max_iter):
        new = np.abs(Xi) / sy >= threshold
        if np.array_equal(new, active):
            break
        active = new
        for i in range(dX.shape[1]):
            Xi[:, i] = 0.0
            if active[:, i].any():
                Xi[active[:, i], i] = ridge_lstsq(A[:, active[:, i]], dX[:, [i]], ridge, free=())[:, 0]
    return Xi / s[:, None]


def true_coefficients(names, sigma=SIGMA, rho=RHO, beta=BETA):
    Xi = np.zeros((len(names), 3))
    ref = {("x", 0): -sigma, ("y", 0): sigma, ("x", 1): rho, ("y", 1): -1.0, ("xz", 1): -1.0,
           ("xy", 2): 1.0, ("z", 2): -beta}
    for (n, i), v in ref.items():
        Xi[names.index(n), i] = v
    return Xi


def model_rhs(Xi, degree=3):
    """Правая часть ẋ = Θ(x) Ξ для интегрирования найденной модели."""
    _, _, E = library(np.zeros((1, 3)), degree)
    def f(S):
        S2 = np.atleast_2d(S)
        Th = np.prod(S2[:, None, :] ** E[None, :, :], axis=2)
        return (Th @ Xi).reshape(np.shape(S))
    return f


# ---------------------------------------------------------------------------
# Метрики для хаотической системы
# ---------------------------------------------------------------------------
def lyapunov_max(f=lorenz_rhs, x0=X0["train"], dt=0.005, t_len=200.0, d0=1e-8, renorm=0.1):
    """Старший показатель Ляпунова методом Бенеттина: две близкие траектории,
    перенормировка расстояния через каждые renorm единиц времени."""
    _, x = simulate(x0, 0.0, dt=dt, substeps=1)
    a = x[-1]; b = a + np.array([d0, 0, 0])
    m = int(round(renorm / dt)); s = 0.0; n = int(round(t_len / renorm))
    for _ in range(n):
        a = rk4(f, a, dt, m)[-1]; b = rk4(f, b, dt, m)[-1]
        d = np.linalg.norm(b - a); s += np.log(d / d0)
        b = a + (b - a) * d0 / d
    return s / (n * renorm)


def attractor_scale(X):
    """Масштаб аттрактора D = sqrt(<||x - <x>||^2>)."""
    return np.sqrt(np.mean(np.sum((X - X.mean(axis=0)) ** 2, axis=1)))


def valid_time(X_true, X_pred, t, D, tol=0.4):
    """Время достоверного прогноза: первый момент, когда ||x̂ - x|| / D > tol."""
    e = np.linalg.norm(X_pred - X_true, axis=1) / D
    bad = np.flatnonzero(~(e <= tol))
    return t[bad[0]] if bad.size else t[-1]


def lorenz_map(z):
    """Последовательные локальные максимумы z: пары (z_n, z_{n+1})."""
    i = np.flatnonzero((z[1:-1] > z[:-2]) & (z[1:-1] >= z[2:])) + 1
    zm = z[i]
    return zm[:-1], zm[1:]


# ---------------------------------------------------------------------------
# Единый стиль рисунков пособия
# ---------------------------------------------------------------------------
def setup_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 11,
        "axes.titlesize": 11, "axes.labelsize": 11, "legend.fontsize": 9.5,
        "legend.frameon": False, "axes.prop_cycle": mpl.cycler(color=["black", "0.35", "0.6"]),
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": "0.85", "grid.linestyle": ":", "grid.linewidth": 0.7,
        "lines.linewidth": 1.3, "image.cmap": "gray", "figure.dpi": 110,
    })

    class CommaFormatter(mpl.ticker.ScalarFormatter):
        def __call__(self, v, pos=None):
            return super().__call__(v, pos).replace(".", ",")
    return CommaFormatter
