import random

# 计算(a/n)
def jacobi_symbol(a, n):
    if n <= 0 or n % 2 == 0:
        return 0
    a = a % n
    result = 1
    while a != 0:
        while a % 2 == 0:
            a = a // 2
            mod = n % 8
            if mod == 3 or mod == 5:
                result = -result
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3:
            result = -result
        a = a % n
    if n == 1:
        return result
    else:
        return 0

# 主要算法
def solovay_strassen(n, t):
    if n == 2:
        return True
    if n % 2 == 0 or n == 1:
        return False

    for round_num in range(1, t + 1):
        b = random.randint(2, n - 1)
        print(f"轮次{round_num}: b = {b}")

        # 计算b^((n-1)/2) (mod n)
        exponent = (n - 1) // 2
        b_power = pow(b, exponent, n)
        print(f"b^((n-1)/2) (mod n) = {b_power}")

        # (i)
        if b_power != 1 and b_power != n - 1:
            print(f"(i)失败: {b_power} != ±1 mod {n}")
            print(f"总轮数: {round_num}, 在(i)退出")
            return False

        jacobi = jacobi_symbol(b, n)
        print(f"(b/n) = {jacobi}")

        if jacobi == -1:
            jacobi_mod = n - 1
        else:
            jacobi_mod = jacobi

        # (ii)
        if b_power != jacobi_mod:
            print(f"(ii)失败: {b_power} != {jacobi_mod} (mod {n})")
            print(f"总轮数: {round_num}, 在(ii)退出")
            return False

    print(f"{t}轮都通过")
    print(f"总轮数: {t}, 在(iii)退出")
    return True


n = 506951
t = 10
print(f"测试n = {n},t = {t}")
result = solovay_strassen(n, t)
if result:
    print(f"\n{n}是素数")
else:
    print(f"\n{n} is 合数")