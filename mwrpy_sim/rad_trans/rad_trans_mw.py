import multiprocessing
from functools import partial

import numpy as np

import mwrpy_sim.constants as con
from mwrpy_sim.rad_trans import calc_absorption
from mwrpy_sim.utils import GAUSS, exponential_integration, read_config


def calc_mw_rt(
    z_final: np.ndarray,
    T_final: np.ndarray,
    p_final: np.ndarray,
    q_final: np.ndarray,
    LWC: np.ndarray,
    f: np.ndarray,
    coeff_bdw: dict,
    mp: bool,
    ape_ang: np.ndarray,
    theta,
) -> np.ndarray:
    """Non-scattering microwave radiative transfer.

    Args:
        z_final: Height profile (km).
        T_final: Temperature profile (K).
        p_final: Pressure profile (hPa).
        q_final: Water vapor density profile (g/m^3).
        LWC: Liquid water content profile (g/m^3).
        f: Frequency vector (GHz).
        coeff_bdw: Coefficients for bandwidth/beamwidth correction.
        mp: Multiprocessing flag (True/False).
        ape_ang: Array of antenna aperture angles (degrees).
        theta: Zenith angle of observation (degrees).

    Returns:
        TB: Brightness temperatures for each frequency (K).
    """
    config = read_config(None, "global_specs")
    if (
        not config["clouds"]
        and np.sum((LWC[1:] + LWC[:-1]) / 2.0 * np.diff(z_final)) > 0.001
        or np.any(np.ma.array(T_final).mask)
    ):
        return np.ones(len(f), np.float32) * -999.0

    if not config["corrections"]:
        # No bandwidth/beamwidth correction
        if mp:
            pool = multiprocessing.Pool()
            func = partial(
                TAU_CALC,
                z_final,
                T_final,
                p_final,
                q_final,
                LWC,
                config,
                theta,
            )
            tau = np.squeeze(np.array([pool.map(func, f)], np.float32))
            pool.close()
            pool.join()
        else:
            tau = np.array(
                [
                    TAU_CALC(
                        z_final,
                        T_final,
                        p_final,
                        q_final,
                        LWC,
                        config,
                        theta,
                        freq,
                    )
                    for _, freq in enumerate(f)
                ],
                np.float32,
            )
        TB = TB_CALC(f, T_final, tau)

    else:
        # With bandwidth/beamwidth correction
        tau_k = np.array(
            [
                TAU_CALC(
                    z_final,
                    T_final,
                    p_final,
                    q_final,
                    LWC,
                    config,
                    theta,
                    freq,
                )
                for ind, freq in enumerate(f[:7])
            ],
            np.float32,
        )
        tau_v = np.array(
            [
                TAU_CALC(
                    z_final,
                    T_final,
                    p_final,
                    q_final,
                    LWC,
                    config,
                    theta,
                    freq,
                )
                for ind, freq in enumerate(coeff_bdw["f_all"])
            ],
            np.float32,
        )

        ape_wgh = GAUSS(ape_ang + theta, theta)
        ape_wgh = ape_wgh / np.sum(ape_wgh)
        ape_0 = ape_wgh[0] if "beamwidth" in config["corrections"] else 1.0
        TB = np.empty(len(f), np.float32)
        TB_k = TB_CALC(f[:7], T_final, tau_k) * ape_0
        for ff in range(7):
            fr_wgh = coeff_bdw["bdw_wgh"][
                ff, coeff_bdw["bdw_wgh"][ff, :] > 0.0
            ] / np.sum(coeff_bdw["bdw_wgh"][ff, coeff_bdw["bdw_wgh"][ff, :] > 0.0])
            TB_v = (
                np.sum(
                    TB_CALC(
                        coeff_bdw["f_all"][
                            coeff_bdw["ind1"][ff] : coeff_bdw["ind1"][ff + 1]
                        ],
                        T_final,
                        tau_v[coeff_bdw["ind1"][ff] : coeff_bdw["ind1"][ff + 1], :],
                    )
                    * fr_wgh
                )
                * ape_0
            )
            if "beamwidth" in config["corrections"]:
                for ia, _ in enumerate(ape_ang[1:]):
                    if ff == 0:
                        # K-band calculations
                        TB_k = np.vstack(
                            (
                                TB_k,
                                TB_CALC(
                                    f[:7],
                                    T_final,
                                    tau_k,
                                )
                                * ape_wgh[ia + 1],
                            )
                        )
                    # V-band calculations
                    TB_v = np.hstack(
                        (
                            TB_v,
                            np.sum(
                                TB_CALC(
                                    coeff_bdw["f_all"][
                                        coeff_bdw["ind1"][ff] : coeff_bdw["ind1"][
                                            ff + 1
                                        ]
                                    ],
                                    T_final,
                                    tau_v[
                                        coeff_bdw["ind1"][ff] : coeff_bdw["ind1"][
                                            ff + 1
                                        ],
                                        :,
                                    ],
                                )
                                * fr_wgh
                            )
                            * ape_wgh[ia + 1],
                        )
                    )
            TB[ff + 7] = np.sum(TB_v)
        TB[:7] = np.sum(TB_k, axis=0) if "beamwidth" in config["corrections"] else TB_k

    return TB


def TAU_CALC(
    z: np.ndarray,
    T: np.ndarray,
    p: np.ndarray,
    rhow: np.ndarray,
    LWC: np.ndarray,
    config: dict,
    theta: float,
    f: float,
) -> np.ndarray:
    """Calculate optical thickness tau.

    Args:
        z: Height profile (km above observation height).
        T: Temperature profile (K).
        p: Pressure profile (hPa).
        rhow: Water vapor density profile (kg/m^3).
        LWC: Liquid water content profile (kg/m^3).
        f: Frequency in GHz.
        config: Configuration dictionary.
        theta: Zenith angle of observation in degrees.

    Returns:
        tau: Optical thickness profile (tau).
    """
    model = config["mw_model"]
    if model not in ["R22", "R24"]:
        raise ValueError(
            f"Microwave model '{model}' is not supported. Use 'R22' or 'R24'."
        )
    abs_wv = np.array(
        [
            eval(f"calc_absorption.ABWV_{model}")(rhow[ii], TT, p[ii], f) / 1000.0
            for ii, TT in enumerate(T)
        ],
        np.float32,
    )
    abs_o2 = np.array(
        [
            eval(f"calc_absorption.ABO2_{model}")(TT, p[ii], rhow[ii], f) / 1000.0
            for ii, TT in enumerate(T)
        ],
        np.float32,
    )
    abs_n2 = np.array(
        [
            calc_absorption.ABN2_R(TT, p[ii] - rhow[ii], f) / 1000.0
            for ii, TT in enumerate(T)
        ],
        np.float32,
    )
    abs_liq = np.array(
        [
            calc_absorption.ABLIQ_R(LWC[ii] * 1000.0, f, TT) / 1000.0
            for ii, TT in enumerate(T)
        ],
        np.float32,
    )

    if not np.isclose(theta, 0.0, 1.0):
        method = config["refractivity"]
        if method not in ["Rueeger2002", "Thayer1974"]:
            raise ValueError(
                f"Refractivity method '{method}' is not supported. "
                "Use 'Rueeger2002' or 'Thayer1974'."
            )
        mu = eval(f"refractivity_{method}")(p, T, rhow)
        deltaz = ray_tracing(z, mu, 90.0 - theta, float(z[0]))
    else:
        deltaz = np.hstack([0.0, np.diff(z)])

    _, tau_wv = exponential_integration(True, abs_wv, deltaz, 1, len(T), 1)
    _, tau_dry = exponential_integration(True, abs_o2 + abs_n2, deltaz, 1, len(T), 1)
    _, tau_liq = exponential_integration(False, abs_liq, deltaz, 1, len(T), 1)

    return tau_wv + tau_dry + tau_liq


def refractivity_Rueeger2002(p: np.ndarray, t: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Computes profile of refractive index.
    Refractivity equations were taken from Rueeger 2002.

    These equations were intended for frequencies under 20 GHz

    Args:
        p: Pressure profile (hPa).
        t: Temperature profile (K).
        e: Vapor pressure profile (hPa).

    Returns:
        refindx: Refractive index profile
    """
    coeff = [77.695, 71.97, 3.75406]
    N = (
        1.0
        + (
            coeff[0] * (((p / 100.0) - e) / t)
            + coeff[1] * (e / t)
            + coeff[2] * (e / (t**2.0))
        )
        * 1e-6
    )
    return N


def refractivity_Thayer1974(p: np.ndarray, t: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Computes profile of refractive index.

    Adapted from pyrtlib:
    Larosa, S., Cimini, D., Gallucci, D., Nilo, S. T., & Romano, F. (2024).
    PyRTlib: a python package for non-scattering line-by-line microwave Radiative
    Transfer simulations. https://doi.org/10.5281/zenodo.8219145
    Refractivity equations were taken from [Thayer-1974]_.

    These equations were intended for frequencies under 20 GHz

    Args:
        p: Pressure profile (hPa).
        t: Temperature profile (K).
        e: Vapor pressure profile (hPa).

    Returns:
        refindx: Refractive index profile
    """
    pa = p - e
    tc = t - 273.16
    rza = 1.0 + np.dot(
        pa, (np.dot(5.79e-07, (1.0 + 0.52 / t)) - np.dot(0.00094611, tc) / t**2)
    )
    rzw = 1.0 + np.dot(
        np.dot(1650.0, (e / (np.dot(t, t**2)))),
        (
            1.0
            - np.dot(0.01317, tc)
            + np.dot(0.000175, tc**2)
            + np.dot(1.44e-06, (np.dot(tc**2, tc)))
        ),
    )
    wetn = np.dot((np.dot(64.79, (e / t)) + np.dot(377600.0, (e / t**2))), rzw)
    dryn = np.dot(np.dot(77.6036, (pa / t)), rza)

    return 1.0 + np.dot((dryn + wetn), 1e-06)


def ray_tracing(
    z: np.ndarray, refindx: np.ndarray, angle: float, z0: float
) -> np.ndarray:
    """Ray-tracing algorithm of Dutton, Thayer, and Westwater.

    Adapted from pyrtlib:
    Larosa, S., Cimini, D., Gallucci, D., Nilo, S. T., & Romano, F. (2024).
    PyRTlib: a python package for non-scattering line-by-line microwave Radiative
    Transfer simulations. https://doi.org/10.5281/zenodo.8219145

    Args:
        z: Height profile (km above observation height, z0).
        refindx: Refractive index profile.
        angle: Elevation angle (degrees).
        z0: Observation height (km msl).

    Returns:
        ds: Array containing slant path length profiles (km)

    note::
        The algorithm assumes that x decays exponentially over each layer.
    """
    deg2rad = np.pi / 180
    re = 6370949.0
    ds = np.zeros(z.shape)
    nl = len(z)

    # Convert angle degrees to radians.  Initialize constant values.
    theta0 = np.dot(angle, deg2rad)
    rs = re + z[0] + z0
    costh0 = np.cos(theta0)
    sina = np.sin(np.dot(theta0, 0.5))
    a0 = np.dot(2.0, (sina**2))
    # Initialize lower boundary values for 1st layer.
    ds[0] = 0.0
    phil = 0.0
    taul = 0.0
    rl = re + z[0] + z0
    tanthl = np.tan(theta0)
    # Construct the slant path length profile.
    for i in range(1, nl):
        r = re + z[i] + z0
        if refindx[i] == refindx[i - 1] or refindx[i] == 1.0 or refindx[i - 1] == 1.0:
            refbar = np.dot((refindx[i] + refindx[i - 1]), 0.5)
        else:
            refbar = 1.0 + (refindx[i - 1] - refindx[i]) / (
                np.log((refindx[i - 1] - 1.0) / (refindx[i] - 1.0))
            )
        argdth = z[i] / rs - (np.dot((refindx[0] - refindx[i]), costh0) / refindx[i])
        argth = np.dot(0.5, (a0 + argdth)) / r
        if argth <= 0:
            return ds
        # Compute d-theta for this layer.
        sint = np.sqrt(np.dot(r, argth))
        theta = np.dot(2.0, np.arcsin(sint))
        if (theta - np.dot(2.0, theta0)) <= 0.0:
            dendth = np.dot(
                np.dot(2.0, (sint + sina)), np.cos(np.dot((theta + theta0), 0.25))
            )
            sind4 = (np.dot(0.5, argdth) - np.dot(z[i], argth)) / dendth
            dtheta = np.dot(4.0, np.arcsin(sind4))
            theta = theta0 + dtheta
        else:
            dtheta = theta - theta0
        # Compute d-tau for this layer (eq.3.71) and add to integral, tau.
        tanth = np.tan(theta)
        cthbar = np.dot(((1.0 / tanth) + (1.0 / tanthl)), 0.5)
        dtau = np.dot(cthbar, (refindx[i - 1] - refindx[i])) / refbar
        tau = taul + dtau
        phi = dtheta + tau
        ds[i] = np.sqrt(
            (z[i] - z[i - 1]) ** 2
            + np.dot(
                np.dot(np.dot(4.0, r), rl), ((np.sin(np.dot((phi - phil), 0.5))) ** 2)
            )
        )
        if dtau != 0.0:
            dtaua = np.abs(tau - taul)
            ds[i] = np.dot(ds[i], (dtaua / (np.dot(2.0, np.sin(np.dot(dtaua, 0.5))))))
        # Make upper boundary into lower boundary for next layer.
        phil = float(np.copy(phi))
        taul = float(np.copy(tau))
        rl = np.copy(r)
        tanthl = np.copy(tanth)

    return np.asarray(ds)


def TB_CALC(frq: np.ndarray, t: np.ndarray, taulay: np.ndarray) -> np.ndarray:
    """Calculate brightness temperatures without scattering.

    Adapted from pyrtlib:
    Larosa, S., Cimini, D., Gallucci, D., Nilo, S. T., & Romano, F. (2024).
    PyRTlib: a python package for non-scattering line-by-line microwave Radiative
    Transfer simulations. https://doi.org/10.5281/zenodo.8219145

    Args:
        frq (numpy.ndarray): Frequency vector in GHz.
        t (numpy.ndarray): Temperature profile in K.
        taulay (numpy.ndarray): Optical thickness profile.

    Returns:
        numpy.ndarray: Brightness temperatures in K.
    """
    hvk = np.dot(frq * 1e9, con.h) / con.kB
    tauprof = np.cumsum(taulay, axis=1)
    boft = 1.0 / (
        np.exp(np.broadcast_to(np.expand_dims(hvk, axis=1), taulay.shape) / t) - 1.0
    )
    boftlay = (boft[:, :-1] + boft[:, 1:] * np.exp(-taulay[:, 1:])) / (
        1.0 + np.exp(-taulay[:, 1:])
    )
    batmlay = boftlay * np.exp(-tauprof[:, :-1]) * (1.0 - np.exp(-taulay[:, 1:]))
    boftatm = np.hstack(
        (np.zeros((len(frq), 1), np.float32), np.cumsum(batmlay, axis=1))
    )

    nl = len(t)
    i_g = np.where(tauprof[:, nl - 1] >= 125.0)[0]
    i_l = np.where(tauprof[:, nl - 1] < 125.0)[0]
    boftotl = np.zeros(len(frq))
    if len(i_l) > 0:
        boftbg = 1.0 / (np.exp(hvk[i_l] / con.Tc) - 1.0)
        boftotl[i_l] = boftbg * np.exp(-tauprof[i_l, nl - 1]) + boftatm[i_l, nl - 1]
    if len(i_g) > 0:
        boftotl[i_g] = boftatm[i_g, nl - 1]

    return hvk / np.log(1.0 + (1.0 / boftotl))
