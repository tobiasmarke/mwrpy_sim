"""Module for atmospheric functions."""

import os

import numpy as np
import pandas as pd

import mwrpy_sim.constants as con


def spec_heat(T: np.ndarray) -> np.ndarray:
    """Specific heat for evaporation (J/kg)."""
    return con.LATENT_HEAT - 2420.0 * (T - con.T0)


def vap_pres(q: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Water vapor pressure (Pa)."""
    return q * con.RW * T


def abs_hum(T: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """Absolute humidity (kg/m^3)."""
    es = calc_saturation_vapor_pressure(T)
    return (rh * es) / (con.RW * T)


def calc_rho(T: np.ndarray, rh: np.ndarray):
    """Density of moist air (g/m^3) from temperature and relative humidity.
    Input:
        T: Temperature (K).
        rh: Relative humidity (0-1).
    Output:
        Density of moist air (g/m^3).
    """
    es = calc_saturation_vapor_pressure(T) / 100.0

    return es * rh


def calc_p_baro(
    T: np.ndarray, a: np.ndarray, p: np.ndarray, z: np.ndarray
) -> np.ndarray:
    """Calculate pressure (Pa) in each level using barometric height formula."""
    e = vap_pres(a, T)
    q = con.MW_RATIO * e / (np.broadcast_to(p, e.T.shape).T - 0.378 * e)
    Tv = T * (1 + 0.608 * q)
    Tv_half = (Tv[:-1] + Tv[1:]) / 2
    dz = np.diff(z)
    dp = np.ma.exp(-con.g0 * dz / (con.RS * Tv_half))
    tmp = np.insert(dp, 0, p, axis=0)
    p_baro = np.cumprod(tmp, axis=0)
    return p_baro


def calc_saturation_vapor_pressure(temperature: np.ndarray) -> np.ndarray:
    """Calculate saturation vapor pressure (Pa) over water.
    Source: Smithsonian Tables 1984, after Goff and Gratch 1946
    Input:
        temperature: Temperature (K).
    Output:
        Saturation vapor pressure (Pa).
    """
    t = 373.16 / temperature
    es = (
        -7.90298 * (t - 1.0)
        + 5.02808 * np.log10(t)
        - 1.3816e-07 * (10 ** (11.344 * (1.0 - (1.0 / t))) - 1.0)
        + 0.0081328 * (10 ** (-3.49149 * (t - 1.0)) - 1.0)
        + np.log10(1013.246)
    )

    return 10.0**es * 100.0


def q2rh(q, T, p):
    """Calculate relative humidity from specific humidity.

    Input:
        q is specific humidity [kg/kg]
        T is temperature [K]
        p is pressure [Pa]
    Output:
        rh is relative humidity [0-1]

    """
    esat = calc_saturation_vapor_pressure(T)

    eps = 0.621970585
    e = np.multiply(p, q) / (np.dot(1000.0, eps) + q)
    rh = np.dot(100.0, e) / esat

    return rh / 100.0


def moist_rho_rh(p, T, rh):
    """Density of moist air from pressure, temperature and relative humidity.

    Input:
        p is in Pa
        T is in K
        rh is in Pa/Pa
    Output:
        density of moist air [kg/m^3]
    """
    eStar = calc_saturation_vapor_pressure(T)
    e = rh * eStar
    q = con.MW_RATIO * e / (p - (1 - con.MW_RATIO) * e)

    return p / (con.RS * T * (1 + (con.RW / con.RS - 1) * q))


def t_dew_rh(T: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """Dew point temperature (K) from relative humidity ()."""
    es = calc_saturation_vapor_pressure(T)
    e = rh * es
    return con.T0 + 243.5 * np.ma.log(e / con.e0) / (17.67 - np.ma.log(e / con.e0))


def mixr(T: np.ndarray, q: np.ndarray, p: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Mixing ratio (kg/kg)."""
    e = vap_pres(q, T)
    p_baro = calc_p_baro(T, q, p, z)
    return con.MW_RATIO * e / (p_baro - e)


def pot_tem(T: np.ndarray, q: np.ndarray, p: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Potential temperature (K)."""
    p_baro = calc_p_baro(T, q, p, z)
    return T * (100000.0 / p_baro) ** (con.RS / con.SPECIFIC_HEAT)


def eq_pot_tem(
    T: np.ndarray, q: np.ndarray, p: np.ndarray, z: np.ndarray
) -> np.ndarray:
    """Equivalent potential temperature (K)."""
    Theta = pot_tem(T, q, p, z)
    return Theta + (spec_heat(T) * mixr(T, q, p, z) / con.SPECIFIC_HEAT) * Theta / T


def hum_to_iwv(ahum, height):
    """Calculate the integrated water vapour.

    Input:
        ahum is in kg/m^3
        height is in m
    Output:
        iwv in kg/m^2
    """
    iwv = np.sum((ahum[1:] + ahum[:-1]) / 2.0 * np.diff(height))

    return iwv


def era5_geopot(level, ps, gpot, temp, hum) -> tuple[np.ndarray, np.ndarray]:
    """Compute geopotential height and pressure from ERA5 model levels.
    Adapted from compute_geopotential_on_ml.py, Copyright 2023 ECMWF.

    Input:
        level: Model level
        ps: Surface Pressure
        gpot: Geopotential Height at Surface
        temp: Temperature
        hum: Humidity
    Output:
        z_f: Geopotential height on model levels
        pres: Pressure on model levels
    """
    file_mh = (
        os.path.dirname(os.path.realpath(__file__))
        + "/rad_trans/coeff/era5_model_levels_137.csv"
    )
    mod_lvl = pd.read_csv(file_mh)
    a_cf = mod_lvl["a [Pa]"].values[:].astype("float")
    b_cf = mod_lvl["b"].values[:].astype("float")
    z_f = np.empty(len(level), np.float32)

    p_h = a_cf + b_cf * ps
    pres = (p_h + np.roll(p_h, 1, axis=0))[1:] / 2

    for lev in sorted(level.astype(int), reverse=True):
        i_z = np.where(level == lev)[0]
        p_l = a_cf[lev - 1] + (b_cf[lev - 1] * ps)
        p_lp = a_cf[lev] + (b_cf[lev] * ps)

        if lev == 1:
            dlog_p = np.log(p_lp / 0.1)
            alpha = np.log(2)
        else:
            dlog_p = np.log(p_lp / p_l)
            alpha = 1.0 - ((p_l / (p_lp - p_l)) * dlog_p)

        temp[i_z] = (temp[i_z] * (1.0 + 0.609133 * hum[i_z])) * con.RS
        z_f[i_z] = gpot + (temp[i_z] * alpha)
        gpot += temp[i_z] * dlog_p

    return np.flip(z_f), np.flip(pres)
