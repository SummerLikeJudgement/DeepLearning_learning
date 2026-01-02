import random
from math import gcd


def modp(x, p):
    return x % p

def poly_trim(a):
    # 去除高位多余0
    while len(a) > 1 and a[-1] == 0:
        a.pop()
    return a

def poly_add(a, b, p):
    # 多项式加法 (系数 mod p)
    n = max(len(a), len(b))
    res = [(0) for _ in range(n)]
    for i in range(n):
        ai = a[i] if i < len(a) else 0
        bi = b[i] if i < len(b) else 0
        res[i] = (ai + bi) % p
    return poly_trim(res)

def poly_sub(a, b, p):
    n = max(len(a), len(b))
    res = [(0) for _ in range(n)]
    for i in range(n):
        ai = a[i] if i < len(a) else 0
        bi = b[i] if i < len(b) else 0
        res[i] = (ai - bi) % p
    return poly_trim(res)

def poly_mul(a, b, p):
    res = [0]*(len(a)+len(b)-1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            res[i+j] = (res[i+j] + ai * bj) % p
    return poly_trim(res)

def poly_divmod(a, b, p):
    # 返回 q, r 使得 a = q*b + r, deg(r) < deg(b)
    a = a[:]
    b = poly_trim(b[:])
    if b == [0]:
        raise ZeroDivisionError("polynomial division by zero")
    # 预计算 b 的首项逆元
    deg_b = len(b) - 1
    inv_lead = pow(b[-1], -1, p)  # 乘法逆元
    q = [0]*(max(0, len(a) - len(b) + 1))
    while len(a) >= len(b) and a != [0]:
        deg_a = len(a) - 1
        coeff = (a[-1] * inv_lead) % p
        shift = deg_a - deg_b
        q[shift] = coeff
        # a -= coeff * x^shift * b
        for i in range(len(b)):
            a[shift + i] = (a[shift + i] - coeff * b[i]) % p
        poly_trim(a)
    return poly_trim(q), poly_trim(a)

def poly_mod(a, m, p):
    _, r = poly_divmod(a, m, p)
    return r

def poly_gcd(a, b, p):
    a = poly_trim(a[:])
    b = poly_trim(b[:])
    while b != [0]:
        _, r = poly_divmod(a, b, p)
        a, b = b, r
    # 归一化，使首项为1
    if a != [0]:
        inv = pow(a[-1], -1, p)
        a = [(c * inv) % p for c in a]
    return a

def poly_pow_mod(base, exp, mod_poly, p):
    # 快速幂，结果在 F_p[x]/(mod_poly)
    result = [1]  # 多项式1
    b = base[:]
    e = exp
    while e > 0:
        if e & 1:
            result = poly_mod(poly_mul(result, b, p), mod_poly, p)
        b = poly_mod(poly_mul(b, b, p), mod_poly, p)
        e >>= 1
    return result

def poly_equal(a, b):
    return poly_trim(a[:]) == poly_trim(b[:])

def poly_deg(a):
    a = poly_trim(a[:])
    return len(a) - 1

def is_irreducible_monic(poly, p):
    # 检查在 F_p 上不可约（并要求首项为1）
    poly = poly_trim(poly[:])
    n = len(poly) - 1
    if n <= 0:
        return False
    if poly[-1] % p != 1:
        return False
    # 使用 Rabin 判别
    x = [0, 1]  # 多项式 x
    # 检查 gcd 条件
    pow_poly = x[:]  # x^{p^i} mod f
    for i in range(1, n // 2 + 1):
        # 计算 x^{p^i} mod f: 对于特征 p，Frobenius: (∑ a_k x^k)^{p} = ∑ a_k x^{k p}
        pow_poly = frobenius_power(pow_poly, p, poly, p)  # 一次 p 次幂
        g = poly_gcd(poly_sub(pow_poly, x, p), poly, p)
        if not poly_equal(g, [1]):
            return False
    # 检查 f | (x^{p^n} - x)
    xpown = x[:]
    for _ in range(n):
        xpown = frobenius_power(xpown, p, poly, p)
    # 检查 x^{p^n} - x ≡ 0 (mod f)
    if not poly_equal(poly_sub(xpown, x, p), [0]):
        return False
    return True

def frobenius_power(a, p_char, mod_poly, p):
    res = [0]
    for k, ak in enumerate(a):
        if ak % p == 0:
            continue
        # x^{k * p_char} * ak
        term = [0]*(k * p_char) + [ak % p]
        res = poly_add(res, term, p)
    res = poly_mod(res, mod_poly, p)
    return res

def random_monic_irreducible(p, n, max_trials=100000):
    # 枚举/随机搜索一个 n 次首一不可约多项式
    # 形式：a0 + a1 x + ... + a_{n-1} x^{n-1} + x^n
    for _ in range(max_trials):
        coeffs = [random.randrange(p) for _ in range(n)]
        poly = coeffs + [1]  # monic
        if is_irreducible_monic(poly, p):
            return poly
    raise RuntimeError("未能在给定次数内找到不可约多项式")


class FPn:
    def __init__(self, coeffs, p, mod_poly):
        # coeffs: 低到高
        self.p = p
        self.mod_poly = mod_poly
        self.n = len(mod_poly) - 1
        self.coeffs = poly_mod(coeffs, mod_poly, p)
        # 填充到固定长度 n 便于展示
        self.coeffs += [0] * (self.n - len(self.coeffs))

    def __add__(self, other):
        return FPn(poly_add(self.coeffs, other.coeffs, self.p), self.p, self.mod_poly)

    def __sub__(self, other):
        return FPn(poly_sub(self.coeffs, other.coeffs, self.p), self.p, self.mod_poly)

    def __mul__(self, other):
        prod = poly_mul(self.coeffs, other.coeffs, self.p)
        prod = poly_mod(prod, self.mod_poly, self.p)
        return FPn(prod, self.p, self.mod_poly)

    def pow(self, e):
        res = [1]
        base = self.coeffs[:]
        modp = self.mod_poly
        p = self.p
        ee = e
        while ee > 0:
            if ee & 1:
                res = poly_mod(poly_mul(res, base, p), modp, p)
            base = poly_mod(poly_mul(base, base, p), modp, p)
            ee >>= 1
        return FPn(res, p, modp)

    def is_zero(self):
        return all(c % self.p == 0 for c in self.coeffs)

    def __eq__(self, other):
        return all((a - b) % self.p == 0 for a, b in zip(self.coeffs, other.coeffs))

    def __repr__(self):
        return f"FPn({self.coeffs})"

    def to_poly_str(self):
        # 人类可读的多项式
        terms = []
        for i, c in enumerate(self.coeffs):
            c = c % self.p
            if c == 0:
                continue
            if i == 0:
                terms.append(f"{c}")
            elif i == 1:
                if c == 1:
                    terms.append("x")
                else:
                    terms.append(f"{c}*x")
            else:
                if c == 1:
                    terms.append(f"x^{i}")
                else:
                    terms.append(f"{c}*x^{i}")
        if not terms:
            return "0"
        return " + ".join(terms)


def prime_factors(n):
    i = 2
    fac = set()
    x = n
    while i * i <= x:
        while x % i == 0:
            fac.add(i)
            x //= i
        i += 1
    if x > 1:
        fac.add(x)
    return sorted(fac)

def is_generator(elem, p, n):
    # 检查 elem 是否为 F_{p^n}^× 的生成元
    order = p**n - 1
    factors = prime_factors(order)
    for q in factors:
        # 若 elem^{order/q} == 1 则不是生成元
        if elem.pow(order // q).coeffs == [1] + [0]*(n-1):
            return False
    return True

def find_generator(p, mod_poly):
    n = len(mod_poly) - 1
    # 从非零元素中随机找一个生成元
    attempts = 0
    while True:
        attempts += 1
        # 随机系数，非零
        coeffs = [random.randrange(p) for _ in range(n)]
        if all(c % p == 0 for c in coeffs):
            continue
        a = FPn(coeffs, p, mod_poly)
        if is_generator(a, p, n):
            return a, attempts


def build_power_table(alpha, p, mod_poly):
    n = len(mod_poly) - 1
    order = p**n - 1
    table = []
    cur = FPn([1], p, mod_poly)  # α^0 = 1
    for k in range(order):
        table.append((k, cur.coeffs[:], cur.to_poly_str()))
        cur = cur * alpha
    return table

def poly_to_str(poly, var='x'):
    # 多项式列表低到高，打印
    poly = poly_trim(poly[:])
    terms = []
    for i, c in enumerate(poly):
        c = c % p_global  # for pretty; will set global
        if c == 0:
            continue
        if i == 0:
            terms.append(f"{c}")
        elif i == 1:
            terms.append(f"{'' if c==1 else str(c)+'*'}{var}")
        else:
            terms.append(f"{'' if c==1 else str(c)+'*'}{var}^{i}")
    return " + ".join(terms) if terms else "0"


def main():
    global p_global
    random.seed()

    # 随机取 10 以内素数 p（>2 以避免过小），和 3..10 的 n
    primes_under_10 = [2, 3, 5, 7]
    p = random.choice(primes_under_10)
    n = random.randint(3, 10)

    print(f"随机选择：p = {p}, n = {n}")

    # 搜索 F_p 上的 n 次不可约（极小）多项式 p(x)
    print("正在搜索 F_p 上的 n 次不可约（极小）多项式 p(x)...")
    px = random_monic_irreducible(p, n)
    p_global = p  # for pretty printer
    print("找到的不可约多项式 p(x) 为（系数从常数到 x^n）：")
    print(px)
    # 同时以可读形式打印
    print("p(x) = " + poly_to_str(px, var='x'))

    # 构造域 F_{p^n}
    print(f"构造扩域 F_{{{p}^{n}}} = F_{p}[x]/(p(x))")

    # 寻找一个本原元 α
    print("正在寻找乘法群的生成元 α（本原元）...")
    alpha, attempts = find_generator(p, px)
    print(f"找到生成元 α（尝试 {attempts} 次）：")
    print(f"α = {alpha.coeffs}  即  {alpha.to_poly_str()}")

    # 构造 α 的幂到多项式的映射表
    print("构造 α 的幂与多项式对应表（所有非零元，共 p^n - 1 个）...")
    table = build_power_table(alpha, p, px)

    # 输出
    print(f"p(x): {px}")
    print(f"p(x) 即 {poly_to_str(px, 'x')}")
    print(f"α: {alpha.coeffs} 即 {alpha.to_poly_str()}")
    print(f"F_{{p^{n}}}^× 的阶 = {p**n - 1}")

    print("\nα^k 与多项式表示（系数在 F_p，按 [常数, x, x^2, ..., x^{n-1}]）：")
    for k, coeffs, poly_str in table:
        print(f"k={k:>4}: α^{k:<4} -> {coeffs})")

if __name__ == "__main__":
    main()