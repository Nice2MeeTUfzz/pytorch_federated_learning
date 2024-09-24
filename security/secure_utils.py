import random

import gmpy2
from phe import paillier
from gmpy2 import mpz, powmod, f_div, invert, is_prime, random_state, mpz_urandomb, rint_round, log2, gcd

global_pub_key, global_privy_key = paillier.generate_paillier_keypair()
public_key = global_pub_key
privy_key = global_privy_key


def key_split(sk, lambda_val, n):
    while True:
        pk1, sk1 = paillier.generate_paillier_keypair()
        sk2 = sk - sk1
        if ((sk1 + sk2) % n == 1) and ((sk1 + sk2) % lambda_val == 0):
            return sk1, sk2


if __name__ == '__main__':
    sk1, sk2 = key_split(privy_key, gmpy2.lcm(privy_key.p - 1, privy_key.q - 1), public_key.n)
    print(f"sk1:{sk1}")
    print(f"sk2:{sk2}")
    # key_split(privy_key, )
