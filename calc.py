from math import log, sqrt, exp
import numpy as np
from scipy.stats import norm

def greeks(s, k, sigma, r, t, option_type):
    sqrt_t = sqrt(t)
    d1 = (log(s / k) + (r + pow(sigma, 2) / 2.0) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    tmp = exp(-pow(d1, 2) / 2.0)
    tmp2 = sqrt(2.0 * np.pi * t)
    tmp3 = r * k * exp(-r * t)
    gamma = tmp / (s * sigma * tmp2)
    theta_call = -(s * sigma * tmp) / (2.0 * tmp2) - tmp3 * norm.cdf(d2)
    vega = s * sqrt_t * tmp / sqrt(2.0 * np.pi)
    if option_type == 'C':
        delta = norm.cdf(d1)
        theta = theta_call
    else:
        delta = norm.cdf(d1) - 1.0
        theta = theta_call + tmp3
    return delta, gamma, theta, vega

def bs_call(s, k, sigma, r, t):
    tmp = sqrt(t)
    d1 = (log(s / k) + (r + pow(sigma, 2) / 2.0) * t) / (sigma * tmp)
    d2 = d1 - sigma * tmp
    return s * norm.cdf(d1) - k * exp(-r * t) * norm.cdf(d2)


def bs_put(s, k, sigma, r, t):
    tmp = sqrt(t)
    d1 = (log(s / k) + (r + pow(sigma, 2) / 2.0) * t) / (sigma * tmp)
    d2 = d1 - sigma * tmp
    return k * exp(-r * t) * norm.cdf(-d2) - s * norm.cdf(-d1)


def call_iv(c, s, k, t, r=0.03, sigma_min=0.01, sigma_max=1.0, e=0.00001):
    sigma_mid = (sigma_min + sigma_max) / 2.0
    call_min = bs_call(s, k, sigma_min, r, t)
    call_max = bs_call(s, k, sigma_max, r, t)
    call_mid = bs_call(s, k, sigma_mid, r, t)
    diff = c - call_mid
    if c <= call_min:
        return sigma_min
    elif c >= call_max:
        return sigma_max
    while abs(diff) > e:
        if c > call_mid:
            sigma_min = sigma_mid
        else:
            sigma_max = sigma_mid
        sigma_mid = (sigma_min + sigma_max) / 2.0
        call_mid = bs_call(s, k, sigma_mid, r, t)
        diff = c - call_mid
    # print(sigma_mid)
    return sigma_mid

def put_iv(c, s, k, t, r=0.03, sigma_min=0.01, sigma_max=1.0, e=0.00001):
    sigma_mid = (sigma_min + sigma_max) / 2.0
    put_min = bs_put(s, k, sigma_min, r, t)
    put_max = bs_put(s, k, sigma_max, r, t)
    put_mid = bs_put(s, k, sigma_mid, r, t)
    diff = c - put_mid
    if c <= put_min:
        return sigma_min
    elif c >= put_max:
        return sigma_max
    while abs(diff) > e:
        if c > put_mid:
            sigma_min = sigma_mid
        else:
            sigma_max = sigma_mid
        sigma_mid = (sigma_min + sigma_max) / 2.0
        put_mid = bs_put(s, k, sigma_mid, r, t)
        diff = c - put_mid
    return sigma_mid