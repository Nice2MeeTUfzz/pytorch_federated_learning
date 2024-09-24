import random

import mod
from mod import Mod
from os import urandom
from Crypto.Util.number import inverse
from functools import reduce
from operator import mul

P = 2 ** 521 - 1


def split_large_number(secret_number, client_number, data_seed):
    """
    返回一个列表
    """
    if secret_number < client_number:
        raise ValueError("Total must be at least as large as n")

    # 设置随机数生成器的种子
    random.seed(data_seed)

    # 初始化分割结果列表
    parts = []

    # 创建n-1个随机分割点
    split_points = set()
    while len(split_points) < client_number - 1:
        split_points.add(random.randint(1, secret_number - (client_number - len(split_points)) - 1))

    # 将分割点排序
    split_points = sorted(list(split_points))

    # 根据分割点创建分割
    current_sum = 0
    for point in split_points:
        parts.append(point - current_sum)
        current_sum = point

    # 最后一个部分是剩余的部分
    parts.append(secret_number - current_sum)

    return parts


def int_from_bytes(s):
    acc = 0
    for b in s:
        acc = acc * 256
        acc += b
    return acc


def create_shares(secret, re_client_number):
    if secret > P:
        raise ValueError("The secret must be smaller than P")
    secret = mod.Mod(secret, P)
    polynomial = [secret]
    for i in range(re_client_number - 1):
        polynomial.append(Mod(int_from_bytes(urandom(16)), P))
    return polynomial


def evaluate(coefficients, x):
    acc = 0
    power = 1
    for c in coefficients:
        acc += c * power
        power *= x
    return acc


def distribute_shares(secret, n, t):
    shards = {}
    polynomial = create_shares(secret=secret, re_client_number=t)
    for i in range(n):
        x = Mod(int_from_bytes(urandom(16)), P)
        y = evaluate(polynomial, x)
        shards[i] = (x, y)
    return shards


def retrieve_original(secrets):
    x_s = [s[0] for s in secrets]
    acc = Mod(0, P)
    for i in range(len(secrets)):
        others = list(x_s)
        cur = others.pop(i)
        factor = Mod(1, P)
        for el in others:
            print(f"el={el}")
            print(f"cur={cur}")
            print(f"jian={el - cur}")
            d = int(el-cur)
            print(f"inverse={inverse(d, P)}")
            factor *= inverse((el * (el - cur)), P)
            acc += factor * secrets[i][1]
    return acc


if __name__ == '__main__':
    # result = split_large_number(secret_number=1000000, client_number=10, seed=42)
    # print(result)
    # create_shares(10209540320490692034234, 4)
    shards = distribute_shares(10, 5, 3)
    del shards[2]
    del shards[3]
    retrieved = list(shards.values())
    re_secret = retrieve_original(retrieved)
    print(re_secret)
