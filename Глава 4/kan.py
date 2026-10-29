"""
Реализация сети Колмогорова–Арнольда (KAN) со сплайновыми функциями на связях.
Цифровое приложение 2 к главе 4 пособия «Нейронные сети для инженеров».

Модуль импортируется в блокноты lab04.ipynb (лабораторная работа 4)
и KAN_примеры.ipynb. Требуется только NumPy.

Содержимое:
  knots, bspline         — равномерная сетка узлов и B-сплайны, формулы (4.11)–(4.15)
  silu, dsilu            — базовая функция b(x) и её производная, (4.20), (4.25)
  KANLayer, KAN          — слой и сеть, формулы (4.17)–(4.19), обратный проход (4.24)–(4.26),
                           обновление сетки (4.27), продолжение функций за границы сетки
  Adam, train            — метод Adam и обучение по MSE (4.23) с отбором по валидации
  check_gradients        — проверка градиентов конечными разностями

Соглашение, как в главе 2: каждая строка матрицы входов X — одно наблюдение.
Функции train и check_gradients работают с любой моделью, у которой есть методы
forward, backward и params (например, с перцептроном из lab04.ipynb).
"""
import numpy as np

__all__ = ['knots', 'bspline', 'silu', 'dsilu', 'KANLayer', 'KAN', 'Adam', 'train', 'check_gradients']


# =====================================================================
# 1. B-сплайны: формулы (4.11)–(4.13) и производная (4.15)
# =====================================================================
def knots(a, b, G, k):
    """Равномерная сетка из G интервалов на [a, b], дополненная k узлами с каждой стороны (4.11)."""
    h = (b - a) / G
    return a + (np.arange(G + 2 * k + 1) - k) * h


def bspline(x, t, k, deriv=False):
    """B-сплайны степени k по формуле Кокса–де Бура (4.12)–(4.13).

    x — вектор точек длины M, t — узлы (4.11).
    Возвращает матрицу B размера M × (G+k); строка — точка, столбец — базисная функция.
    При deriv=True дополнительно возвращает матрицу производных (4.15).
    На равномерной сетке все знаменатели равны p*h > 0, поэтому деления на нуль нет.
    """
    x = np.asarray(x, float)[:, None]
    B = ((x >= t[None, :-1]) & (x < t[None, 1:])).astype(float)       # степень 0
    B_prev = None
    for p in range(1, k + 1):
        B_prev = B
        left = (x - t[None, :-(p + 1)]) / (t[None, p:-1] - t[None, :-(p + 1)])
        right = (t[None, p + 1:] - x) / (t[None, p + 1:] - t[None, 1:-p])
        B = left * B_prev[:, :-1] + right * B_prev[:, 1:]
    if not deriv:
        return B
    d1 = k / (t[k:-1] - t[:-(k + 1)])
    d2 = k / (t[k + 1:] - t[1:-k])
    dB = d1[None, :] * B_prev[:, :-1] - d2[None, :] * B_prev[:, 1:]
    return B, dB


def silu(x):
    """Базовая функция b(x) = SiLU(x) = x / (1 + e^{-x}), формула (4.20)."""
    return x / (1.0 + np.exp(-x))


def dsilu(x):
    """Производная b'(x) = s(x)[1 + x(1 - s(x))], s — логистическая сигмоида, формула (4.25)."""
    s = 1.0 / (1.0 + np.exp(-x))
    return s * (1.0 + x * (1.0 - s))


# =====================================================================
# 2. Слой и сеть Колмогорова–Арнольда: формулы (4.17)–(4.19), (4.24)–(4.27)
# =====================================================================
class KANLayer:
    """Слой: x_j^(l) = sum_i phi_{l,j,i}(x_i^(l-1)),
    phi(x) = w_b * SiLU(x) + w_s * sum_m c_m B_{m,k}(x)  — формула (4.19)."""

    def __init__(self, nin, nout, G, k, lo=-1.0, hi=1.0):
        self.nin, self.nout, self.G, self.k = nin, nout, G, k
        self.t = [knots(lo, hi, G, k) for _ in range(nin)]       # своя сетка у каждого входа
        self.c = np.zeros((nout, nin, G + k))                    # коэффициенты сплайнов
        self.wb = np.zeros((nout, nin))                          # веса базовой функции
        self.ws = np.ones((nout, nin))                           # масштаб сплайновой части
        self.extend = 'zero'      # 'zero' — финитные сплайны; 'linear' — линейное продолжение

    def params(self):
        return [self.c, self.wb, self.ws]

    def n_params(self):
        return self.c.size + self.wb.size + self.ws.size        # (G+k+2) на связь, формула (4.21)

    def forward(self, X):
        """Прямой проход; X — матрица M × nin. Матрица B вычисляется один раз на вход i."""
        M = X.shape[0]
        Y = np.zeros((M, self.nout))
        self.cache = []
        for i in range(self.nin):
            B = bspline(X[:, i], self.t[i], self.k)             # не зависит от j
            S = silu(X[:, i])
            spl = B @ self.c[:, i, :].T                         # M × nout: сплайновые части
            Y += S[:, None] * self.wb[:, i] + spl * self.ws[:, i]
            self.cache.append((B, S, spl))
        self.X = X
        return Y

    def backward(self, D):
        """Обратный проход: D = dL/dY (M × nout). Возвращает градиенты параметров и dL/dX."""
        X = self.X
        gc = np.zeros_like(self.c)
        gwb = np.zeros_like(self.wb)
        gws = np.zeros_like(self.ws)
        DX = np.zeros_like(X)
        for i in range(self.nin):
            B, S, spl = self.cache[i]
            _, dB = bspline(X[:, i], self.t[i], self.k, deriv=True)
            dS = dsilu(X[:, i])
            gc[:, i, :] = (D.T @ B) * self.ws[:, i][:, None]     # (4.24): dphi/dc_m = w_s B_m
            gwb[:, i] = D.T @ S                                  # dphi/dw_b = b(x)
            gws[:, i] = np.sum(D * spl, axis=0)                  # dphi/dw_s = сплайн
            dspl = dB @ self.c[:, i, :].T                        # M × nout
            dphi = dS[:, None] * self.wb[:, i] + dspl * self.ws[:, i]   # (4.25)
            DX[:, i] = np.sum(D * dphi, axis=1)                  # (4.26)
        return [gc, gwb, gws], DX

    def edge(self, j, i, x):
        """Значения функции phi_{j,i} в точках x с учётом выбранного продолжения за сетку."""
        x = np.asarray(x, float)
        a, b = self.t[i][self.k], self.t[i][-self.k - 1]
        f = lambda z: self.wb[j, i] * silu(z) + self.ws[j, i] * (bspline(z, self.t[i], self.k) @ self.c[j, i])
        if self.extend == 'zero':
            return f(x)
        # линейное продолжение по касательной в граничных точках (п. 4.4.6)
        eps = 1e-6
        fa, fb = f(np.array([a]))[0], f(np.array([b - eps]))[0]
        da = (f(np.array([a + eps]))[0] - fa) / eps
        db = (fb - f(np.array([b - 2 * eps]))[0]) / eps
        y = f(np.clip(x, a, b - 1e-12))
        y = np.where(x > b, fb + db * (x - b), y)
        return np.where(x < a, fa + da * (x - a), y)

    def update_grid(self, Xin, G_new, margin=0.05):
        """Обновление и уточнение сетки (4.27): новая сетка по диапазону данных,
        коэффициенты — МНК так, чтобы сохранить сплайновую часть функций."""
        K_new = G_new + self.k
        c_new = np.zeros((self.nout, self.nin, K_new))
        t_new = []
        for i in range(self.nin):
            lo, hi = Xin[:, i].min(), Xin[:, i].max()
            w = hi - lo
            tn = knots(lo - margin * w, hi + margin * w, G_new, self.k)
            xi = np.linspace(lo - margin * w, hi + margin * w, 20 * K_new)   # контрольные точки
            B_old = bspline(xi, self.t[i], self.k)
            B_new = bspline(xi, tn, self.k)
            for j in range(self.nout):
                c_new[j, i] = np.linalg.lstsq(B_new, B_old @ self.c[j, i], rcond=None)[0]
            t_new.append(tn)
        self.t, self.G, self.c = t_new, G_new, c_new


class KAN:
    """Сеть как композиция слоёв (4.18). sizes — например [2, 1, 1]."""

    def __init__(self, sizes, G, k=3):
        self.sizes = sizes
        self.layers = [KANLayer(a, b, G, k) for a, b in zip(sizes[:-1], sizes[1:])]

    def forward(self, X):
        for L in self.layers:
            X = L.forward(X)
        return X

    def backward(self, D):
        grads = []
        for L in reversed(self.layers):
            g, D = L.backward(D)
            grads = g + grads
        return grads

    def params(self):
        return [p for L in self.layers for p in L.params()]

    def n_params(self):
        return sum(L.n_params() for L in self.layers)

    def update_grid(self, X, G_new):
        for L in self.layers:
            Xin = X
            X = L.forward(X)
            L.update_grid(Xin, G_new)

    def set_extension(self, mode):
        for L in self.layers:
            L.extend = mode

    def predict(self, X):
        """Прогноз с учётом способа продолжения (для 'zero' совпадает с forward)."""
        if all(L.extend == 'zero' for L in self.layers):
            return self.forward(X)
        for L in self.layers:
            Y = np.zeros((X.shape[0], L.nout))
            for j in range(L.nout):
                for i in range(L.nin):
                    Y[:, j] += L.edge(j, i, X[:, i])
            X = Y
        return X


# =====================================================================
# 3. Обучение: MSE (4.23), метод Adam, отбор состояния по валидации
# =====================================================================
class Adam:
    def __init__(self, params, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.p, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = [np.zeros_like(q) for q in params]
        self.v = [np.zeros_like(q) for q in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for q, g, m, v in zip(self.p, grads, self.m, self.v):
            m *= self.b1; m += (1 - self.b1) * g
            v *= self.b2; v += (1 - self.b2) * g * g
            q -= self.lr * (m / (1 - self.b1 ** self.t)) / (np.sqrt(v / (1 - self.b2 ** self.t)) + self.eps)


def train(model, Xtr, ytr, Xva, yva, epochs, lr, history):
    """Полный пакет; после обучения восстанавливается состояние с наименьшей валидационной MSE."""
    opt = Adam(model.params(), lr)
    best_val, best_params = np.inf, None
    for _ in range(epochs):
        y = model.forward(Xtr)
        opt.step(model.backward(2 * (y - ytr) / y.size))
        tr = np.mean((model.forward(Xtr) - ytr) ** 2)
        va = np.mean((model.forward(Xva) - yva) ** 2)
        history.append((tr, va))
        if va < best_val:
            best_val, best_params = va, [p.copy() for p in model.params()]
    for p, q in zip(model.params(), best_params):
        p[...] = q
    return best_val


def check_gradients(model, X, y, n_checks=1, h=1e-6, seed=0):
    """Сравнение градиентов с конечными разностями; возвращает наибольшее относительное расхождение."""
    rng = np.random.default_rng(seed)
    loss = lambda: np.mean((model.forward(X) - y) ** 2)
    grads = model.backward(2 * (model.forward(X) - y) / y.size)
    worst = 0.0
    for p, g in zip(model.params(), grads):
        for _ in range(n_checks):
            idx = tuple(rng.integers(0, s) for s in p.shape)
            old = p[idx]
            p[idx] = old + h; l1 = loss()
            p[idx] = old - h; l2 = loss()
            p[idx] = old
            worst = max(worst, abs((l1 - l2) / (2 * h) - g[idx]) / (abs(g[idx]) + 1e-8))
    return worst
