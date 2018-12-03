__author__ = 'Vitaly Butoma'

from scipy.stats import norm
import math
from math import factorial as fact
from functools import partial


def bits_dist(mu=128, sigma=127):
    return partial(norm.pdf, loc=mu, scale=sigma)


def C(n, m):
    p = fact(m)
    q = fact(n) * fact(m - n)
    return p // q


if __name__ == "__main__":
    p = [C(i, 255) / 2**255 for i in range(256)]
    print(sum(p))
    m_x = 0
    d_x = 0
    for i in range(256):
        m_x += p[i] * i
    for i in range(256):
        d_x += p[i] * (i - m_x)**2
    print(m_x)
    print(d_x)
    f = bits_dist(mu=m_x, sigma=math.sqrt(d_x))
