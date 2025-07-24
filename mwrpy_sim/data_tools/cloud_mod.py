import numpy as np

from mwrpy_sim import constants as con
from mwrpy_sim.atmos import abs_hum, moist_rho_rh, spec_heat


def detect_cloud_mod(z, lwc):
    """Detect liquid cloud boundaries from model data.

    Input:
        z: height grid
        lwc: liquid water content on z
    Output:
        z_top: array of cloud tops
        z_base: array of cloud bases
    """
    i_cloud, i_top, i_base = (
        np.array(np.where(lwc > 0.0)[0], dtype=np.int32),
        np.empty(0, np.int32),
        np.empty(0, np.int32),
    )
    if len(i_cloud) > 1:
        i_base = np.unique(
            np.hstack((i_cloud[0], i_cloud[np.diff(np.hstack((0, i_cloud))) > 1]))
        )
        i_top = np.hstack(
            (i_cloud[np.diff(np.hstack((i_cloud, i_cloud[-1]))) > 1], i_cloud[-1])
        )

        if len(i_top) != len(i_base):
            print("something wrong, number of bases NE number of cloud tops!")
            return [], []

    return z[i_top], z[i_base]


def detect_liq_cloud(z, t, rh, p_rs):
    """Detect liquid water cloud boundaries from relative humidity and temperature.

    Adapted from PAMTRA:
    Mech, M., Maahn, M., Kneifel, S., Ori, D., Orlandi, E., Kollias, P., Schemann, V.,
    and Crewell, S.: PAMTRA 1.0: the Passive and Active Microwave radiative TRAnsfer
    tool for simulating radiometer and radar measurements of the cloudy atmosphere,
    Geosci. Model Dev., 13, 4229–4251, https://doi.org/10.5194/gmd-13-4229-2020, 2020

    Input:
        z: height grid
        T: temperature on z
        rh: relative humidty on z
        rh_thres: relative humidity threshold for the detection on liquid clouds on z
        T_thres: do not detect liquid water clouds below this value (scalar)
    Output:
        z_top: array of cloud tops
        z_base: array of cloud bases
    """
    alpha = 0.59
    beta = 1.37
    sigma = p_rs / p_rs[0]
    rh_thres = 1.0 - alpha * sigma * (1.0 - sigma) * (1.0 + beta * (sigma - 0.5))
    # rh_thres = 0.95  # 1
    t_thres = 253.15  # K
    # ***determine cloud boundaries
    # --> layers where mean rh GT rh_thres

    i_cloud, i_top, i_base = (
        np.array(np.where((rh > rh_thres) & (t > t_thres))[0], dtype=np.int32),
        np.empty(0, np.int32),
        np.empty(0, np.int32),
    )
    if len(i_cloud) > 1:
        i_base = np.unique(
            np.hstack((i_cloud[0], i_cloud[np.diff(np.hstack((0, i_cloud))) > 1]))
        )
        i_top = np.hstack(
            (i_cloud[np.diff(np.hstack((i_cloud, 0))) > 1] - 1, i_cloud[-1])
        )

        if len(i_top) != len(i_base):
            print("something wrong, number of bases NE number of cloud tops!")
            return [], []

    return z[i_top], z[i_base]


def adiab(i, T, P, z):
    """Adiabatic liquid water content assuming pseudo-adiabatic lapse rate
    throughout the whole cloud layer. Thus, the assumed temperature
    profile is different from the measured one.

    Adapted from PAMTRA:
    Mech, M., Maahn, M., Kneifel, S., Ori, D., Orlandi, E., Kollias, P., Schemann, V.,
    and Crewell, S.: PAMTRA 1.0: the Passive and Active Microwave radiative TRAnsfer
    tool for simulating radiometer and radar measurements of the cloudy atmosphere,
    Geosci. Model Dev., 13, 4229–4251, https://doi.org/10.5194/gmd-13-4229-2020, 2020

    Input:
        i no of levels
        T is in K
        p is in Pa
        z is in m
    Output:
        LWC is in kg/m^3
    """
    #   Set actual cloud base temperature to the measured one
    #   Initialize Liquid Water Content (LWC)
    #   Compute adiabatic LWC by integration from cloud base to level I

    TCL = T[0]
    LWC = 0.0

    for j in range(1, i + 1):
        deltaz = z[j] - z[j - 1]

        #   Compute actual cloud temperature

        #   1. Compute air density
        #   2. Compute water vapor density of saturated air
        #   3. Compute mixing ratio of saturated air
        #   4. Compute pseudoadiabatic lapse rate
        #   5. Compute actual cloud temperature

        R = moist_rho_rh(P[j], T[j], 1.0)
        RWV = abs_hum(T[j], np.ones(1, np.float32))
        WS = RWV / (R - RWV)
        DTPS = pseudoAdiabLapseRate(T[j], WS)
        TCL -= DTPS * deltaz

        #   Compute adiabatic LWC

        #   1. Compute air density
        #   2. Compute water vapor density of saturated air
        #   3. Compute mixing ratio of saturated air
        #   4. Compute specific heat of vaporisation
        #   5. Compute adiabatic LWC

        R = moist_rho_rh(P[j], TCL, 1.0)
        RWV = abs_hum(TCL, np.ones(1, np.float32))
        WS = RWV / (R - RWV)
        L = spec_heat(TCL)

        LWC += (
            R
            * con.SPECIFIC_HEAT
            / L
            * ((con.g0 / con.SPECIFIC_HEAT) - pseudoAdiabLapseRate(TCL, WS))
            * deltaz
        )

    return LWC


def mod_ad(T_cloud, p_cloud, z_cloud):
    """Modified adiabatic liquid water content assuming pseudo-adiabatic lapse rate.

    Adapted from PAMTRA:
    Mech, M., Maahn, M., Kneifel, S., Ori, D., Orlandi, E., Kollias, P., Schemann, V.,
    and Crewell, S.: PAMTRA 1.0: the Passive and Active Microwave radiative TRAnsfer
    tool for simulating radiometer and radar measurements of the cloudy atmosphere,
    Geosci. Model Dev., 13, 4229–4251, https://doi.org/10.5194/gmd-13-4229-2020, 2020

    Input:
        T_cloud: Temperature [K]
        p_cloud: Pressure [Pa]
        z_cloud: Height [m]
    Output:
        lwc: Liquid water content [kg/m^3]
        cloud_new: New cloud height [m]
    """
    n_level = len(T_cloud)
    lwc = np.zeros(n_level - 1)
    cloud_new = np.zeros(n_level - 1)

    thick = 0.0
    for jj in range(n_level - 1):
        deltaz = z_cloud[jj + 1] - z_cloud[jj]
        thick = deltaz + thick
        lwc[jj] = adiab(jj + 1, T_cloud, p_cloud, z_cloud) * (
            -0.144779 * np.log(thick) + 1.239387
        )
        cloud_new[jj] = z_cloud[jj] + deltaz / 2.0
    return lwc, cloud_new


def pseudoAdiabLapseRate(T, Ws):
    """Pseudoadiabatic lapse rate.

    Adapted from PAMTRA:
    Mech, M., Maahn, M., Kneifel, S., Ori, D., Orlandi, E., Kollias, P., Schemann, V.,
    and Crewell, S.: PAMTRA 1.0: the Passive and Active Microwave radiative TRAnsfer
    tool for simulating radiometer and radar measurements of the cloudy atmosphere,
    Geosci. Model Dev., 13, 4229–4251, https://doi.org/10.5194/gmd-13-4229-2020, 2020

    Input:
        T [K]: thermodynamic temperature
        Ws [1]: mixing ratio of saturation
    Output:
        PSEUDO [K/m]: pseudoadiabatic lapse rate
    Constants:
        Grav [m/s2]: constant of acceleration
        CP [J/(kg K)]: specific heat cap. at const. press
        Rair [J/(kg K)]: gas constant of dry air
        Rvapor [J/(kg K)]: gas constant of water vapor
    Source: Rogers and Yau, 1989: A Short Course in Cloud Physics
    (III.Ed.), Pergamon Press, 293p. Page 32
    """
    # Compute specific humidity of vaporisation
    L = spec_heat(T)

    # Compute pseudo-adiabatic temperature gradient
    pseudo = (
        (con.g0 / con.SPECIFIC_HEAT)
        * (1 + (L * Ws / con.RS / T))
        / (1 + (Ws * L**2 / con.SPECIFIC_HEAT / con.RW / T**2))
    )

    return pseudo


def get_cloud_prop(
    base: np.ndarray, top: np.ndarray, input_dat: dict, method: str
) -> tuple[np.ndarray, float]:
    """Calculate cloud properties."""
    if method == "prognostic":
        lwc_new, height_new = input_dat["lwc_in"][:], input_dat["height"][:]
        lwp = np.sum(
            (input_dat["lwc_in"][1:] + input_dat["lwc_in"][:-1])
            / 2.0
            * np.diff(input_dat["height"][:])
        )
    else:
        lwc, cloud_new = np.empty(0, np.float64), np.empty(0, np.float64)
        lwp, height_new = 0.0, np.empty(0, np.float64)
        for icl, _ in enumerate(top):
            xcl = np.where(
                (input_dat["height"][:] >= base[icl] - 0.01)
                & (input_dat["height"][:] <= top[icl] + 0.01)
            )[0]
            if len(xcl) > 1:
                lwcx, cloudx = mod_ad(
                    input_dat["air_temperature"][xcl],
                    input_dat["air_pressure"][xcl],
                    input_dat["height"][xcl],
                )
                lwp += np.sum(lwcx * np.diff(input_dat["height"][xcl]))
                cloud_new = np.hstack((cloud_new, cloudx))
                lwc = np.hstack((lwc, lwcx))
                if len(height_new) == 0:
                    height_new = input_dat["height"][input_dat["height"][:] < base[0]]
                else:
                    height_new = np.hstack(
                        (
                            height_new,
                            input_dat["height"][
                                (input_dat["height"][:] > top[icl - 1])
                                & (input_dat["height"][:] < base[icl])
                            ],
                        )
                    )
        height_new = np.sort(
            np.hstack(
                (
                    height_new,
                    cloud_new,
                    input_dat["height"][input_dat["height"][:] > top[-1]],
                )
            )
        )
        lwc_new = np.zeros(len(height_new), np.float32)
        if len(lwc) > 0:
            _, xx, yy = np.intersect1d(
                height_new, cloud_new, assume_unique=False, return_indices=True
            )
            lwc_new[xx] = lwc[yy]

    lwc_in = np.interp(input_dat["height"][:], height_new, lwc_new)
    return lwc_in, lwp
