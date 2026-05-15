"""Prepare input data for MWRPy sim."""

import datetime
import os

import atmoslib
import netCDF4 as nc
import numpy as np
import pandas as pd
from atmoslib import constants as con
from scipy.interpolate import CubicSpline

from mwrpy_sim.utils import seconds_since_epoch


def prepare_cn(
    cn_data: nc.Dataset,
    date_i: list,
    add_2m: bool = False,
) -> dict:
    """Prepare input data from ECMWF's IFS or ERA5 model (Cloudnet format)."""
    hasl = atmoslib.geometric_height(cn_data["sfc_geopotential"][:] / con.G)
    input_cn = {
        "height": cn_data["height"][:-1, :]
        + np.resize(hasl, cn_data["height"][:-1, :].shape),
        "air_temperature": cn_data["temperature"][:-1, :],
        "air_pressure": cn_data["pressure"][:-1, :],
        "relative_humidity": cn_data["rh"][:-1, :],
    }
    vp = atmoslib.vapor_pressure(input_cn["air_pressure"], cn_data["q"][:-1, :])
    mr = atmoslib.mixing_ratio(vp, input_cn["air_pressure"])
    input_cn["lwc_in"] = cn_data["ql"][:-1, :] * atmoslib.air_density(
        input_cn["air_temperature"], input_cn["air_pressure"], mr
    )
    if add_2m:
        input_cn = _add_2m_fields(cn_data, input_cn, hasl)
    input_cn["time"] = np.array(
        [seconds_since_epoch(d_str) for d_str in date_i], dtype=np.int64
    )
    check_len = 138 if add_2m else 137
    n_h = (
        len(input_cn["height"])
        if input_cn["height"].ndim == 1
        else input_cn["height"].shape[1]
    )
    return _add_std_atm(input_cn, h_val=90.0) if n_h == check_len else {}


def _add_2m_fields(cn_data: nc.Dataset, input_cn: dict, hasl: float) -> dict:
    """Add 2m fields to input."""
    if "sfc_dewpoint_temp_2m" in cn_data.variables:
        nn = np.exp(
            (17.625 * atmoslib.k2c(cn_data["sfc_dewpoint_temp_2m"][:]))
            / (243.04 + atmoslib.k2c(cn_data["sfc_dewpoint_temp_2m"][:]))
        )
        dn = np.exp(
            (17.625 * atmoslib.k2c(cn_data["sfc_temp_2m"][:]))
            / (243.04 + atmoslib.k2c(cn_data["sfc_temp_2m"][:]))
        )
        rh_sfc = nn / dn
    else:
        rh_sfc = atmoslib.relative_humidity(
            cn_data["sfc_temp_2m"][:],
            cn_data["sfc_pressure"][:],
            cn_data["sfc_q_2m"][:],
        )

    ax = 0 if input_cn["height"].ndim == 1 else 1
    lwc_0 = (
        input_cn["lwc_in"][0]
        if input_cn["height"].ndim == 1
        else input_cn["lwc_in"][:-1, 0]
    )
    input_cn = {
        "height": np.insert(input_cn["height"], 0, 2.0 + hasl, axis=ax),
        "air_temperature": np.insert(
            input_cn["air_temperature"],
            0,
            cn_data["sfc_temp_2m"][:-1],
            axis=ax,
        ),
        "air_pressure": np.insert(
            input_cn["air_pressure"],
            0,
            cn_data["sfc_pressure"][:-1],
            axis=ax,
        ),
        "relative_humidity": np.insert(
            input_cn["relative_humidity"],
            0,
            rh_sfc[:-1],
            axis=ax,
        ),
        "lwc_in": np.insert(
            input_cn["lwc_in"],
            0,
            lwc_0,
            axis=ax,
        ),
    }
    return input_cn


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
        + "/data_tools/era5_download/era5_model_levels_137.csv"
    )
    mod_lvl = pd.read_csv(file_mh)
    a_cf = mod_lvl["a [Pa]"].values[:].astype("float")
    b_cf = mod_lvl["b"].values[:].astype("float")
    z_f = np.empty(len(level), np.float32)

    p_h = a_cf + b_cf * ps
    pres = (p_h + np.roll(p_h, 1, axis=0))[1:] / 2

    level_i = np.array(sorted(level.astype(int), reverse=True))
    for lev in level_i:
        i_z = np.where(level == lev)[0]
        p_l = a_cf[lev - 1] + (b_cf[lev - 1] * ps)
        p_lp = a_cf[lev] + (b_cf[lev] * ps)

        if lev == 1:
            dlog_p = np.log(p_lp / 0.1)
            alpha = np.log(2)
        else:
            dlog_p = np.log(p_lp / p_l)
            alpha = 1.0 - ((p_l / (p_lp - p_l)) * dlog_p)

        temp[i_z] = (temp[i_z] * (1.0 + 0.609133 * hum[i_z])) * 287.058
        z_f[i_z] = gpot + (temp[i_z] * alpha)
        gpot += temp[i_z] * dlog_p

    return np.flip(z_f), np.flip(pres)


def prepare_era5_mod(
    mod_data_sfc: nc.Dataset, mod_data_pro: nc.Dataset, index: int, date_i: str
) -> dict:
    """Prepare input data from ERA5 on model levels."""
    input_era5: dict = {}
    geopotential, input_era5["air_pressure"] = era5_geopot(
        mod_data_pro["model_level"][:],
        np.mean(np.exp(mod_data_sfc["lnsp"][index, 0, :, :]), axis=(0, 1)),
        np.mean(mod_data_sfc["z"][index, 0, :, :], axis=(0, 1)),
        np.mean(mod_data_pro["t"][index, :, :, :], axis=(1, 2)),
        np.mean(mod_data_pro["q"][index, :, :, :], axis=(1, 2)),
    )
    input_era5["height"] = atmoslib.geometric_height(geopotential[:] / con.G)
    input_era5["air_temperature"] = np.flip(
        np.mean(mod_data_pro["t"][index, :, :, :], axis=(1, 2))
    )[:]
    q = np.flip(np.mean(mod_data_pro["q"][index, :, :, :], axis=(1, 2)))[:]
    input_era5["relative_humidity"] = atmoslib.relative_humidity(
        input_era5["air_temperature"],
        input_era5["air_pressure"],
        q,
    )
    clwc = np.flip(np.mean(mod_data_pro["clwc"][index, :, :, :], axis=(1, 2)))[:]
    vp = atmoslib.vapor_pressure(input_era5["air_pressure"], q)
    mxr = atmoslib.mixing_ratio(vp, input_era5["air_pressure"])
    input_era5["lwc_in"] = clwc * atmoslib.air_density(
        input_era5["air_temperature"], input_era5["air_pressure"], mxr
    )
    input_era5["time"] = np.array(seconds_since_epoch(date_i), dtype=np.int64)

    return _add_std_atm(input_era5)


def prepare_era5_pres(mod_data: nc.Dataset, index: int, date_i: str) -> dict:
    """Prepare input data from ERA5 on pressure levels."""
    input_era5: dict = {}
    geopotential = np.mean(mod_data["z"][index, :, :, :], axis=(1, 2))[:]
    input_era5["height"] = atmoslib.geometric_height(geopotential[:] / con.G)
    input_era5["air_pressure"] = mod_data["pressure_level"][:] * 100.0
    input_era5["air_temperature"] = np.mean(mod_data["t"][index, :, :, :], axis=(1, 2))[
        :
    ]
    input_era5["relative_humidity"] = (
        np.mean(mod_data["r"][index, :, :, :], axis=(1, 2))[:] / 100.0
    )
    clwc = np.mean(mod_data["clwc"][index, :, :, :], axis=(1, 2))[:]
    q = np.flip(np.mean(mod_data["q"][index, :, :, :], axis=(1, 2)))[:]
    vp = atmoslib.vapor_pressure(input_era5["air_pressure"], q)
    mxr = atmoslib.mixing_ratio(vp, input_era5["air_pressure"])
    input_era5["lwc_in"] = clwc * atmoslib.air_density(
        input_era5["air_temperature"], input_era5["air_pressure"], mxr
    )
    input_era5["time"] = np.array(seconds_since_epoch(date_i), dtype=np.int64)

    return _add_std_atm(input_era5)


def prepare_gruan(rs_data: nc.Dataset) -> dict:
    """Prepare input data from RS41-GDP radiosonde measurements."""
    height, ind_s = np.sort(rs_data["alt_amsl"]), np.argsort(rs_data["alt_amsl"])
    _, ind = np.unique(height, return_index=True, equal_nan=True)
    ind = ind_s[ind]
    ind = ind[~np.isnan(rs_data["temp"][ind])]
    if rs_data["alt_amsl"][ind[-2]] < 10000.0:
        return {}
    time = datetime.datetime.strptime(
        rs_data.__dict__["g.Measurement.StartTime"], "%Y-%m-%dT%H:%M:%S.%fZ"
    )
    input_rs = {
        "height": rs_data["alt_amsl"][ind[::10]],
        "air_temperature": rs_data["temp"][ind[::10]],
        "relative_humidity": rs_data["rh"][ind[::10]] / 100.0,
        "air_pressure": rs_data["press"][ind[::10]] * 100.0,
        "time": np.array(
            [seconds_since_epoch(datetime.datetime.strftime(time, "%Y%m%d%H%M"))],
            dtype=np.int64,
        ),
    }
    return _add_std_atm(input_rs, np.max(input_rs["height"] / 1000.0) + 1)


def prepare_vaisala(vs_data: nc.Dataset, altitude: float = 0.0) -> dict:
    """Prepare input data from JOYCE Vaisala radiosonde measurements."""
    ind = np.where(np.diff(vs_data["Height"][:]) < 0)[0]
    ind_h = len(vs_data["Height"][:]) - 1 if len(ind) == 0 else ind[0]
    input_vs: dict = {
        "height": np.append(
            vs_data["Height"][:ind_h][:100], vs_data["Height"][:ind_h][100:][0::10]
        )
        + altitude,
        "air_temperature": np.append(
            vs_data["Temperature"][:ind_h][:100],
            vs_data["Temperature"][:ind_h][100:][0::10],
        ),
        "air_pressure": np.append(
            vs_data["Pressure"][:ind_h][:100], vs_data["Pressure"][:ind_h][100:][0::10]
        )
        * 100.0,
        "relative_humidity": np.append(
            vs_data["Humidity"][:ind_h][:100], vs_data["Humidity"][:ind_h][100:][0::10]
        )
        / 100,
        "time": np.array(
            seconds_since_epoch(
                datetime.datetime.strptime(
                    vs_data["Time"][0][:19], "%Y-%m-%dT%H:%M:%S"
                ).strftime("%Y%m%d%H%M")
            ),
            dtype=np.int64,
        ),
    }

    return _add_std_atm(input_vs, 50.0)


def prepare_standard_atmosphere(prof: int = 5) -> dict:
    """Prepare standard atmosphere data for MWRPy sim.
    Parameters
        prof : int, optional
        Profile index to select from the standard atmosphere data, by default 5:
        0: midlatitude summer,
        1: midlatitude winter,
        2: subarctic summer,
        3: subarctic winter,
        4: tropic,
        5: U.S. standard
    Returns
        dict
            Dictionary containing standard atmosphere data.
    """
    sa_data = nc.Dataset(
        os.path.abspath(os.curdir) + "/tests/data/standard_atmospheres.nc"
    )
    input_sa = {
        "height": sa_data.variables["height"][:].astype(np.float64) * 1000.0,
        "air_temperature": sa_data.variables["t_atmo"][:, prof].astype(np.float64),
        "air_pressure": sa_data.variables["p_atmo"][:, prof].astype(np.float64) * 100.0,
        "lwc_in": np.zeros(len(sa_data.variables["height"][:]), dtype=np.float64),
    }
    input_sa["relative_humidity"] = atmoslib.relative_humidity(
        input_sa["air_temperature"],
        input_sa["air_pressure"],
        sa_data.variables["q_atmo"][:, prof].astype(np.float64),
    )
    input_sa["time"] = np.array([0], dtype=np.int64)

    return _add_vars(input_sa)


def _add_std_atm(input_dat: dict, h_val: float = 100.0, prof: int = 5) -> dict:
    """Add standard atmosphere data to input data dictionary.

    Parameters
    input_dat : dict
        Input data dictionary to which the standard atmosphere data will be added.
    h_val : float, optional
        Height from which the standard atmosphere data is added, by default 100.0 km.
    prof : int, optional
        Profile index to select from the standard atmosphere data, by default 5:
        0: midlatitude summer,
        1: midlatitude winter,
        2: subarctic summer,
        3: subarctic winter,
        4: tropic,
        5: U.S. standard
    Returns
    dict
        Updated input data dictionary with standard atmosphere data added.
    """
    sa = prepare_standard_atmosphere(prof)
    ind_sa = np.where(sa["height"][:] >= h_val * 1000.0)[0]
    var_names = [
        "height",
        "air_temperature",
        "air_pressure",
        "relative_humidity",
        "lwc_in",
    ]
    h_top = (
        float(input_dat["height"][-1])
        if input_dat["height"].ndim == 1
        else np.max(input_dat["height"][:, -1])
    )
    if len(ind_sa) > 0 and h_top < h_val * 1000.0:
        for var in var_names:
            if var == "lwc_in" and var in input_dat:
                if input_dat[var].ndim == 1:
                    input_dat["lwc_in"] = np.append(
                        input_dat["lwc_in"], np.zeros(len(ind_sa), dtype=np.float64)
                    )
                else:
                    input_dat["lwc_in"] = np.hstack(
                        (
                            input_dat["lwc_in"],
                            np.zeros(
                                (len(input_dat["lwc_in"]), len(sa[var][ind_sa])),
                                dtype=np.float64,
                            ),
                        )
                    )
            elif var != "lwc_in" and var in input_dat:
                sa_add = (
                    sa[var][ind_sa]
                    if input_dat[var].ndim == 1
                    else np.resize(
                        sa[var][ind_sa], (len(input_dat[var]), len(sa[var][ind_sa]))
                    )
                )
                input_dat[var] = np.hstack((input_dat[var], sa_add))

    return _add_vars(input_dat)


def _add_vars(input_dat) -> dict:
    """Add variables to input data dictionary."""
    input_dat["relative_humidity"][input_dat["relative_humidity"] < 0.0] = 0.0
    q = atmoslib.specific_humidity(
        input_dat["air_temperature"][:],
        input_dat["air_pressure"][:],
        input_dat["relative_humidity"][:],
    )
    vp = atmoslib.vapor_pressure(input_dat["air_pressure"][:], q)
    input_dat["absolute_humidity"] = atmoslib.absolute_humidity(
        input_dat["air_temperature"][:], vp
    )
    for key in input_dat.keys():
        input_dat[key] = np.ma.asarray(input_dat[key])
    if "iwv" not in input_dat:
        input_dat["iwv"] = (
            np.ma.array(
                [
                    hum_to_iwv(
                        input_dat["absolute_humidity"],
                        input_dat["height"],
                    )
                ],
                dtype=np.float64,
            )
            if not np.any(input_dat["absolute_humidity"].mask)
            else np.ma.array(np.ones(len(input_dat["time"])) * np.nan)
        )

    return input_dat


def check_height(input_dict: dict, altitude: float, tolerance: float = 5.0) -> dict:
    """Compare input data height with altitude and cut/extrapolate if necessary.
    In addition, calculate integrated water vapor (iwv) from absolute humidity.

    Parameters
    input_dict : dict
        Input data dictionary containing height and other atmospheric variables.
    altitude : float
        Altitude to compare with the input data height.
    tolerance : float, optional
        Tolerance for the height comparison, by default 5 m.

    Returns:
    dict
        Updated input data dictionary with height adjusted to the specified altitude.
    """
    delta_z = input_dict["height"][0] - altitude
    if delta_z < -150.0:
        # Cut off the first part of the profile
        for key in input_dict:
            if key not in ("height", "time") and len(input_dict[key]) == len(
                input_dict["height"]
            ):
                input_dict[key] = input_dict[key][input_dict["height"][:] >= altitude]
        input_dict["height"] = input_dict["height"][input_dict["height"][:] >= altitude]

    elif np.abs(delta_z) > tolerance and delta_z > -150.0:
        # Extrapolate the first part of the profile (< 1000 m)
        ind_h = input_dict["height"] - input_dict["height"][0] <= 1000.0
        height_new = (
            input_dict["height"][ind_h]
            - delta_z
            + np.linspace(0, 1, len(input_dict["height"][ind_h])) * delta_z
        )
        for key in input_dict:
            if key not in ("height", "time") and len(input_dict[key]) == len(
                input_dict["height"]
            ):
                f = CubicSpline(
                    input_dict["height"][ind_h],
                    input_dict[key][ind_h],
                    bc_type="natural",
                )
                input_dict[key] = np.ma.asarray(
                    np.concatenate([f(height_new), input_dict[key][~ind_h].data])
                )
        if "lwc_in" in input_dict:
            input_dict["lwc_in"][input_dict["lwc_in"] < 1e-8] = 0.0
        input_dict["height"] = np.concatenate(
            [height_new, input_dict["height"][~ind_h]]
        )

    # Calculate integrated water vapor (iwv) from absolute humidity
    input_dict["iwv"] = np.array(
        [
            hum_to_iwv(
                input_dict["absolute_humidity"][:],
                input_dict["height"][:],
            )
            if not np.any(input_dict["absolute_humidity"].mask)
            else -999.0
        ],
        dtype=np.float64,
    )

    return input_dict


def check_height_day(input_dict: dict, altitude: float, tolerance: float = 5.0) -> dict:
    """Compare input data height with altitude and cut/extrapolate if necessary.
    In addition, calculate integrated water vapor (iwv) from absolute humidity.

    Parameters
    input_dict : dict
        Input data dictionary containing height and other atmospheric variables.
    altitude : float
        Altitude to compare with the input data height.
    tolerance : float, optional
        Tolerance for the height comparison, by default 5 m.

    Returns:
    dict
        Updated input data dictionary with height adjusted to the specified altitude.
    """
    delta_z = input_dict["height"][:, 0] - altitude
    height_new = np.mean(input_dict["height"], axis=0)
    if np.all(np.abs(delta_z) > tolerance) and np.all(delta_z < -150.0):
        # Cut off the first part of the profile
        for key in input_dict:
            if input_dict[key].ndim == 2:
                input_dict[key] = input_dict[key][
                    :, np.all(input_dict["height"] >= altitude, axis=0)
                ]
        height_new = np.mean(input_dict["height"], axis=0)

    elif np.all(np.abs(delta_z) > tolerance) and np.all(delta_z >= -150.0):
        # Extrapolate the first part of the profile (< 1000 m)
        ind_h = np.all(input_dict["height"] < 1000.0, axis=0)
        height_1000 = (
            height_new[ind_h]
            - np.mean(delta_z)
            + np.linspace(0, 1, len(height_new[ind_h])) * np.mean(delta_z)
        )
        height_new = np.concatenate([height_1000, height_new[~ind_h]])
        for itx, _ in enumerate(input_dict["time"]):
            for key in input_dict:
                if (
                    input_dict[key].ndim == 2
                    and input_dict[key].shape[1] == len(height_new)
                    and key != "height"
                ):
                    f = CubicSpline(
                        input_dict["height"][itx, ind_h],
                        input_dict[key][itx, ind_h],
                        bc_type="natural",
                    )
                    input_dict[key][itx, :] = np.ma.concatenate(
                        [f(height_new[ind_h]), input_dict[key][itx, ~ind_h].data]
                    )

    input_dict["height"] = height_new
    input_dict["lwc_in"][input_dict["lwc_in"] < 1e-8] = 0.0

    input_dict["iwv"] = (
        hum_to_iwv(
            input_dict["absolute_humidity"],
            input_dict["height"],
        )
        if not np.any(input_dict["absolute_humidity"].mask)
        else np.ma.array(np.ones(len(input_dict["time"])) * np.nan)
    )

    return input_dict


def hum_to_iwv(ahum, height):
    """Calculate the integrated water vapour.

    Input:
        ahum is in kg/m^3
        height is in m
    Output:
        iwv in kg/m^2
    """
    iwv = (
        np.sum((ahum[1:] + ahum[:-1]) / 2.0 * np.diff(height))
        if ahum.ndim == 1
        else np.sum((ahum[:, 1:] + ahum[:, :-1]) / 2.0 * np.diff(height), axis=1)
    )

    return iwv
