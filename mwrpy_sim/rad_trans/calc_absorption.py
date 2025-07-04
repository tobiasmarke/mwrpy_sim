"""
This class contains the absorption models used in mwrpy_sim. Adapted from:
Rosenkranz, P.W.: Line-by-line microwave radiative transfer (non-scattering)
[2022] (version 2022/08/25),
[2024] (version 2024/07/03),
http://cetemps.aquila.infn.it/mwrnet/lblmrt_ns.html
"""

import os

import numpy as np

from mwrpy_sim.utils import dcerror, loadCoeffsJSON


def ABWV_R22(
    RHO,  # abs. humidity in gm-3
    T,  # temp. in K
    P,  # pressure in hPa
    F,  # freqeuncy in GHz
):
    """
    ABSORPTION COEF IN ATMOSPHERE DUE TO WATER VAPOR
      T       KELVIN    I   TEMPERATURE
      P       MILLIBAR  I   PRESSURE              .1 TO 1000
      RHO     G/M**3    I   WATER VAPOR DENSITY
      F       GHZ       I   FREQUENCY             0 TO 800
      ALPHA   NEPERS/KM O   ABSORPTION COEFFICIENT
    """

    if RHO <= 0.0:
        return 0.0

    path = os.path.dirname(os.path.realpath(__file__)) + "/coeff/r22/h2o_list.json"
    CF = loadCoeffsJSON(path)
    CF["W0"] = CF["Wair"] / 1000.0
    CF["W0S"] = CF["Wself"] / 1000.0
    CF["SH"] = CF["Sair"] / 1000.0
    CF["SHS"] = CF["Sself"] / 1000.0
    CF["W2"] = CF["W2air"] / 1000.0
    CF["W2S"] = CF["W2self"] / 1000.0
    CF["D2"] = CF["D2air"] / 1000.0
    CF["D2S"] = CF["D2self"] / 1000.0

    RVAP = (0.01 * 8.31451) / 18.01528
    RHO = RHO / (RVAP * T)
    PVAP = RHO * T * 4.6152e-3
    PDA = P - PVAP
    TI = CF["Trefcon"] / T
    CON = (
        (CF["Cf"] * PDA * TI ** CF["Xcf"] + CF["Cs"] * PVAP * TI ** CF["Xcs"])
        * PVAP
        * F**2.0
    )
    REFTLINE = 296.0
    TI = REFTLINE / T
    TILN = np.log(TI)
    TI2 = np.exp(2.5 * TILN)

    # ****ADD RESONANCES
    SUMM = 0.0
    for I, FL in enumerate(CF["FL"]):
        WIDTH0 = (
            CF["W0"][I] * PDA * TI ** CF["X"][I]
            + CF["W0S"][I] * PVAP * TI ** CF["XS"][I]
        )
        if CF["W2"][I] > 0.0:
            WIDTH2 = (
                CF["W2"][I] * PDA * TI ** CF["XW2"][I]
                + CF["W2S"][I] * PVAP * TI ** CF["XW2S"][I]
            )
        else:
            WIDTH2 = 0.0

        DELTA2 = CF["D2"][I] * PDA + CF["D2S"][I] * PVAP
        SHIFTF = CF["SH"][I] * PDA * (1.0 - CF["AAIR"][I] * TILN) * TI ** CF["XH"][I]
        SHIFTS = (
            CF["SHS"][I] * PVAP * (1.0 - CF["ASELF"][I] * TILN) * TI ** CF["XHS"][I]
        )
        SHIFT = SHIFTF + SHIFTS
        S = CF["S1"][I] * TI2 * np.exp(CF["B2"][I] * (1.0 - TI))

        DF = np.zeros(2, np.float32)
        DF[0] = F - FL - SHIFT
        DF[1] = F + FL + SHIFT
        WSQ = WIDTH0**2.0
        BASE = WIDTH0 / (562500.0 + WSQ)
        RES = 0.0
        for J in range(0, 2):
            if WIDTH2 > 0 and J == 0 and np.abs(DF[J]) < (10 * WIDTH0):
                XC = complex((WIDTH0 - 1.5 * WIDTH2), DF[J] + 1.5 * DELTA2) / complex(
                    WIDTH2, -DELTA2
                )
                XRT = np.sqrt(XC)
                PXW = 1.77245385090551603 * XRT * dcerror(-np.imag(XRT), np.real(XRT))
                SD = 2.0 * (1.0 - PXW) / complex(WIDTH2, -DELTA2)
                RES += np.real(SD) - BASE
            elif np.abs(DF[J]) < 750.0:
                RES += WIDTH0 / (DF[J] ** 2 + WSQ) - BASE

        SUMM += S * RES * (F / FL) ** 2.0

    return 1.0e-10 * (RHO / 2.9915075e-23) * SUMM / np.pi + CON


def ABWV_R24(
    RHO,  # abs. humidity in gm-3
    T,  # temp. in K
    P,  # pressure in hPa
    F,  # freqeuncy in GHz
):
    """
    ABSORPTION COEF IN ATMOSPHERE DUE TO WATER VAPOR
      T       KELVIN    I   TEMPERATURE
      P       MILLIBAR  I   PRESSURE              .1 TO 1000
      RHO     G/M**3    I   WATER VAPOR DENSITY
      F       GHZ       I   FREQUENCY             0 TO 800
      ALPHA   NEPERS/KM O   ABSORPTION COEFFICIENT
    """

    if RHO <= 0.0:
        return 0.0

    path = os.path.dirname(os.path.realpath(__file__)) + "/coeff/r24/h2o_list.json"
    CF = loadCoeffsJSON(path)

    RVAP = (0.01 * 8.31451) / 18.01528
    RHO = RHO / (RVAP * T)
    PVAP = RHO * T * 4.6152e-3
    PDA = P - PVAP
    TI = CF["Trefcon"] / T
    TILN = np.log(TI)
    TI2 = np.exp(2.5 * TILN)

    SUM = 0.0
    for i, FL in enumerate(CF["FL"]):
        WIDTH0 = (
            CF["W0"][i] / 1000.0 * PDA * TI ** CF["X"][i]
            + CF["W0S"][i] / 1000.0 * PVAP * TI ** CF["XS"][i]
        )
        WIDTH2 = (
            (
                CF["W2"][i] / 1000.0 * PDA * TI ** CF["XW2"][i]
                + CF["W2S"][i] / 1000.0 * PVAP * TI ** CF["XW2S"][i]
            )
            if CF["W2"][i] > 0.0
            else 0.0
        )
        DELTA2 = CF["D2"][i] / 1000.0 * PDA + CF["D2S"][i] / 1000.0 * PVAP
        SHIFT = (
            CF["SH"][i]
            / 1000.0
            * PDA
            * (1.0 - CF["AAIR"][i] * TILN)
            * TI ** CF["XH"][i]
        ) + (
            CF["SHS"][i]
            / 1000.0
            * PVAP
            * (1.0 - CF["ASELF"][i] * TILN)
            * TI ** CF["XHS"][i]
        )
        S = CF["S1"][i] * TI2 * np.exp(CF["B2"][i] * (1.0 - TI))

        DF = np.array([F - FL - SHIFT, F + FL + SHIFT], np.float32)
        WSQ = WIDTH0**2.0
        BASE = WIDTH0 / (562500.0 + WSQ)
        RES = 0.0
        for J in range(2):
            if WIDTH2 > 0 and J == 0 and np.abs(DF[J]) < (10 * WIDTH0):
                XC = complex((WIDTH0 - 1.5 * WIDTH2), DF[J] + 1.5 * DELTA2) / complex(
                    WIDTH2, -DELTA2
                )
                XRT = np.sqrt(XC)
                PXW = 1.77245385090551603 * XRT * dcerror(-np.imag(XRT), np.real(XRT))
                SD = 2.0 * (1.0 - PXW) / complex(WIDTH2, -DELTA2)
                RES += np.real(SD) - BASE
            elif np.abs(DF[J]) < 750.0:
                RES += WIDTH0 / (DF[J] ** 2 + WSQ) - BASE

        SUM += S * RES * (F / FL) ** 2.0

    ALPHA = H2OCON(F, T)
    CON = (CF["Cf"] * TI ** CF["Xcf"] * PDA + ALPHA * PVAP) * PVAP * F**2.0

    return 1.0e-10 * (RHO / 2.9915075e-23) * SUM / np.pi + CON


def H2OCON(F, T):
    """Compute the self-continuum absorption of water vapor."""
    NF = 6
    SELFCON = [2.877e-21, 2.855e-21, 2.731e-21, 2.49e-21, 2.178e-21, 1.863e-21]
    SELFTEXP = [6.413, 6.414, 6.275, 6.049, 5.789, 5.557]

    THETA = 296.0 / T
    A = np.array(
        [6.532e12 * SELFCON[j] * THETA ** (SELFTEXP[j] + 3.0) for j in range(NF)],
        np.float32,
    )
    A = np.insert(A, 0, A[1])

    FJ = F / 299.792458
    J = np.minimum(np.int32(FJ), NF - 2)
    P = FJ - J
    C = (3.0 - 2.0 * P) * P**2.0
    B = 0.5 * P * (1.0 - P)
    B1, B2 = B * (1.0 - P), B * P

    return -A[J] * B1 + A[J + 1] * (1.0 - C + B2) + A[J + 2] * (C + B1) - A[J + 3] * B2


def ABO2_R22(TEMP, PRES, VAPDEN, FREQ):
    """
    ABSORPTION COEFFICIENT DUE TO OXYGEN IN AIR
    TEMP    KELVIN   TEMPERATURE        UNCERTAIN, but believed to be
                                         valid for atmosphere
    PRES   MILLIBARS PRESSURE           3 TO 1000
    VAPDEN  G/M^3    WV DENSITY         (ENTERS LINEWIDTH CALCULATION
                     DUE TO GREATER BROADENING EFFICIENCY OF H2O)
    FREQ    GHZ      FREQUENCY          0 TO 900
    """

    path = os.path.dirname(os.path.realpath(__file__)) + "/coeff/r22/o2_list.json"
    CF = loadCoeffsJSON(path)

    TH = 300.0 / TEMP
    TH1 = TH - 1.0
    B = TH ** CF["X"]
    RVAP = (0.01 * 8.314510) / 18.01528
    VAPDEN = VAPDEN / (RVAP * TEMP)
    PRESWV = VAPDEN * TEMP / 216.68
    PRESDA = PRES - PRESWV
    DEN = 0.001 * (PRESDA * B + 1.2 * PRESWV * TH)
    DFNR = CF["WB300"] * DEN
    PE2 = DEN**2.0

    SUM = 1.584e-17 * FREQ**2.0 * DFNR / (TH * (FREQ**2.0 + DFNR**2.0))

    for K, FL in enumerate(CF["F"]):
        Y = DEN * (CF["Y0"][K] + CF["Y1"][K] * TH1)
        DNU = PE2 * (CF["DNU0"][K] + CF["DNU1"][K] * TH1)
        GFAC = 1.0 + PE2 * (CF["G0"][K] + CF["G1"][K] * TH1)
        DF = CF["W300"][K] * DEN
        STR = CF["S300"][K] * np.exp(-CF["BE"][K] * TH1)
        DEL1 = FREQ - FL - DNU
        DEL2 = FREQ + FL + DNU
        D1 = DEL1**2.0 + DF**2.0
        D2 = DEL2**2.0 + DF**2.0
        SF1 = (DF * GFAC + DEL1 * Y) / D1
        SF2 = (DF * GFAC - DEL2 * Y) / D2

        SUM += STR * (SF1 + SF2) * (FREQ / FL) ** 2.0

    ALPHA = 1.6097e11 * SUM * PRESDA * TH**3.0

    return np.maximum(ALPHA, 0.0) * 1.004


def ABO2_R24(TEMP, PRES, VAPDEN, FREQ):
    """Calculate absorption coefficient due to oxygen in air."""
    path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "coeff/r24/o2_list.json"
    )
    CF = loadCoeffsJSON(path)

    RVAP = (0.01 * 8.314510) / 18.01528
    VAPDEN /= RVAP * TEMP
    TH = 300.0 / TEMP
    TH1 = TH - 1.0
    PRESWV = 4.615228e-3 * VAPDEN * TEMP
    PRESDA = PRES - PRESWV
    PE1 = 0.001 * (PRESDA * TH ** CF["X"] + 1.2 * PRESWV * TH)
    PE2 = PE1**2
    WNR = CF["WB300"] * PE1
    SUMM = 1.584e-17 * WNR / (FREQ**2.0 + WNR**2.0)

    A, Y, G = (
        np.zeros(len(CF["F"]), np.float32),
        np.zeros(len(CF["F"]), np.float32),
        np.zeros(len(CF["F"]), np.float32),
    )
    for k, f in enumerate(CF["F"]):
        A[k] = CF["S300"][k] * np.exp(-CF["BE"][k] * TH1) * TH / f**2
        Y[k] = 0.99 * (CF["Y300"][k] + CF["Y1"][k] * TH1)
        G[k] = CF["G0"][k] + CF["G1"][k] * TH1

    SUMY = 1.584e-17 * CF["WB300"] + 2.0 * np.sum(
        A[:38] * (CF["W300"][:38] + Y[:38] * CF["F"][:38])
    )
    SUMG = np.sum(A[1:37] * G[1:37])
    ASQ = np.sum(A[1:37] ** 2)
    ANORM = np.sum(A[1:37])
    SUMY2 = SUMY / (2.0 * ANORM)
    RATIO = SUMG / ASQ
    Y[1:37] -= SUMY2 / CF["F"][1:37]
    G[1:37] -= A[1:37] * RATIO

    for k, F in enumerate(CF["F"]):
        WIDTH = CF["W300"][k] * PE1
        YK = PE1 * Y[k]
        GFAC = 1.0 + PE2 * G[k] if k in range(1, 38) else 1.0
        FCEN = F + PE2 * (CF["DNU0"][k] + CF["DNU1"][k] * TH1)

        if k == 0 and np.abs(FREQ - FCEN) < 10.0 * WIDTH:
            width2 = 0.076 * WIDTH
            xc = complex(WIDTH - 1.5 * width2, (FREQ - FCEN)) / width2
            xrt = np.sqrt(xc)
            pxw = 1.77245385090551603 * xrt * dcerror(-np.imag(xrt), np.real(xrt))
            asd = complex(1.0, YK) * 2.0 * (1.0 - pxw) / width2
            SF1 = np.real(asd)
        else:
            SF1 = (WIDTH * GFAC + (FREQ - FCEN) * YK) / (
                (FREQ - FCEN) ** 2 + WIDTH**2
            )

        SF2 = (WIDTH * GFAC - (FREQ + FCEN) * YK) / (
            (FREQ + FCEN) ** 2.0 + WIDTH**2.0
        )
        SUMM += A[k] * (SF1 + SF2)

    return 1.6097e11 * SUMM * PRESDA * (FREQ * TH) ** 2


def ABN2_R(T, P, F):
    """
    ABSORPTION COEFFICIENT DUE TO NITROGEN IN AIR
    T = TEMPERATURE (K)
    P = PRESSURE (MB)
    F = FREQUENCY (GHZ)
    """
    TH = 300.0 / T
    FDEPEN = 0.5 + 0.5 / (1.0 + (F / 450.0) ** 2.0)

    return 9.95e-14 * FDEPEN * P**2.0 * F**2.0 * TH**3.22


def ABLIQ_R(LWC, F, T):
    """POWER ABSORPTION COEFFICIENT
    BY SUSPENDED CLOUD LIQUID WATER DROPLETS."""

    if LWC <= 0.0 or T < 233.0:
        return 0.0

    Tc = T - 273.15
    theta = 300.0 / T
    z = complex(0.0, F)

    kappa = (
        -43.7527 * theta**0.05
        + 299.504 * theta**1.47
        - 399.364 * theta**2.11
        + 221.327 * theta**2.31
    )

    delta = 80.69715 * np.exp(-Tc / 226.45)
    sd = 1164.023 * np.exp(-651.4728 / (Tc + 133.07))
    kappa -= delta * z / (sd + z)
    delta = 4.008724 * np.exp(-Tc / 103.05)
    hdelta = delta / 2.0
    f1 = (
        10.46012
        + (0.1454962 * Tc)
        + (0.063267156 * Tc**2)
        + (0.00093786645 * Tc**3)
    )
    z1 = complex(-0.75, 1.0) * f1
    z2 = complex(-4500.0, 2000.0)
    cnorm = np.log(z2 / z1)
    chip = (hdelta * np.log((z - z2) / (z - z1))) / cnorm
    chij = (hdelta * np.log((z - np.conj(z2)) / (z - np.conj(z1)))) / np.conj(cnorm)
    dchi = chip + chij - delta
    kappa += dchi

    RE = (kappa - 1.0) / (kappa + 2.0)

    return -0.06286 * np.imag(RE) * F * LWC
