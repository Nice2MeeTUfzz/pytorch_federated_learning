import random
import time
import numpy as np
from sympy import symbols, Poly, mod_inverse
from scipy.interpolate import lagrange
from random import getrandbits, seed
from decimal import Decimal
from secrets import randbelow

"""
大素数p
"""
p = 2 ** 512 - 1


def shamir_secret_sharing(secret, n, k, gen_seed, p=p):
    """
    秘密共享函数。
    :param secret: 需要共享的秘密
    :param n: 分享的总数
    :param k: 恢复秘密所需的最小份额数
    :return: 一个包含n个份额的列表
    """
    seed(gen_seed)
    x = symbols('x')
    # coeffs = [getrandbits(32) for _ in range(k - 1)]
    coeffs = [randbelow(p) for _ in range(k - 1)]
    coeffs.append(secret)
    # poly = Poly(coeffs, x, domain='ZZ')
    poly = Poly(coeffs, x, domain=f'GF({p})')
    print(poly)
    shares = []
    for i in range(1, n + 1):
        share = poly.subs(x, i) % p
        shares.append((i, share))
    return shares


def lagrange_interpolation(x, y, t, p=p):
    """
    计算拉格朗日插值多项式的常数项。
    :param x: 份额的 x 值列表
    :param y: 份额的 y 值列表
    :param t: 插值的变量
    :param p: 有限域的模数
    :return: 多项式在 t 点的值
    """
    assert len(x) == len(y), "x 和 y 的长度必须相同"
    n = len(x)
    L = 0
    for i in range(n):
        # 计算第 i 个基多项式
        Li = 1
        for j in range(n):
            if i != j:
                if x[i] == x[j]:
                    raise ValueError('x值需不相同')
                Li *= ((t - x[j]) * mod_inverse(x[i] - x[j], p)) % p
        L += (y[i] * Li) % p
    return L


def recover_secret(shares):
    """
    从份额中恢复秘密。
    :param shares: 份额列表
    :return: 恢复的秘密
    """
    x = []
    y = []
    for index, value in shares:
        x.append(index)
        y.append(value)
    t = symbols('t')
    # a = lagrange(x, y)
    a = lagrange_interpolation(x, y, t)
    # recovered_secret = round(a(0))
    recovered_secret = a.subs(t, 0) % p
    # return recovered_secret
    return int(recovered_secret)


if __name__ == '__main__':
    model_gradient = 12345678901234567890
    n = 10
    k = 5
    shares = shamir_secret_sharing(model_gradient, n, k, gen_seed=123)
    print(shares)
    recovered_gradient = recover_secret(random.sample(shares, k))
    print(f"Recovered model gradient: {recovered_gradient}")
