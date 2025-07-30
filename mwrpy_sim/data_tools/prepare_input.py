"""Prepare input data for MWRPy sim."""

import os

import metpy.calc
import netCDF4 as nc
import numpy as np
from metpy.units import units
from scipy.interpolate import make_interp_spline

import mwrpy_sim.constants as con
from mwrpy_sim.atmos import (
    abs_hum,
    calc_rho,
    era5_geopot,
    hum_to_iwv,
    mixr,
    moist_rho_rh,
    q2rh,
)
from mwrpy_sim.utils import seconds_since_epoch


def prepare_ifs(ifs_data: nc.Dataset, index: int, date_i: str) -> dict:
    """Prepare input data from ECMWF's IFS model (Cloudnet format)."""
    input_ifs = {
        "height": ifs_data["height"][index, :],
        "air_temperature": ifs_data["temperature"][index, :],
        "air_pressure": ifs_data["pressure"][index, :],
        "relative_humidity": ifs_data["rh"][index, :],
    }
    input_ifs["lwc_in"] = ifs_data["ql"][index, :] * moist_rho_rh(
        input_ifs["air_pressure"],
        input_ifs["air_temperature"],
        input_ifs["relative_humidity"],
    )
    input_ifs["time"] = np.array(seconds_since_epoch(date_i), dtype=np.int64)

    return (
        _add_std_atm(input_ifs, h_val=90.0) if len(input_ifs["height"]) == 137 else {}
    )


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
    geopotential = units.Quantity(geopotential, "m^2/s^2")
    input_era5["height"] = metpy.calc.geopotential_to_height(geopotential[:]).magnitude
    input_era5["air_temperature"] = np.flip(
        np.mean(mod_data_pro["t"][index, :, :, :], axis=(1, 2))
    )[:]
    input_era5["relative_humidity"] = (
        metpy.calc.relative_humidity_from_specific_humidity(
            input_era5["air_pressure"] * units.Pa,
            units.Quantity(input_era5["air_temperature"], "K"),
            np.flip(np.mean(mod_data_pro["q"][index, :, :, :], axis=(1, 2)))[:],
        ).magnitude
    )
    clwc = np.flip(np.mean(mod_data_pro["clwc"][index, :, :, :], axis=(1, 2)))[:]
    mxr = metpy.calc.mixing_ratio_from_relative_humidity(
        units.Quantity(input_era5["air_pressure"], "Pa"),
        units.Quantity(input_era5["air_temperature"], "K"),
        units.Quantity(input_era5["relative_humidity"], "dimensionless"),
    )
    rho = metpy.calc.density(
        units.Quantity(input_era5["air_pressure"], "Pa"),
        units.Quantity(input_era5["air_temperature"], "K"),
        mxr,
    )
    input_era5["lwc_in"] = clwc * rho.magnitude
    input_era5["time"] = seconds_since_epoch(date_i)

    return _add_std_atm(input_era5)


def prepare_era5_pres(mod_data: nc.Dataset, index: int, date_i: str) -> dict:
    """Prepare input data from ERA5 on pressure levels."""
    input_era5: dict = {}
    geopotential = np.mean(mod_data["z"][index, :, :, :], axis=(1, 2))[:]
    input_era5["height"] = metpy.calc.geopotential_to_height(
        units.Quantity(geopotential, "m^2/s^2")
    ).magnitude
    input_era5["air_pressure"] = mod_data["pressure_level"][:] * 100.0
    input_era5["air_temperature"] = np.mean(mod_data["t"][index, :, :, :], axis=(1, 2))[
        :
    ]
    input_era5["relative_humidity"] = (
        np.mean(mod_data["r"][index, :, :, :], axis=(1, 2))[:] / 100.0
    )
    clwc = np.mean(mod_data["clwc"][index, :, :, :], axis=(1, 2))[:]
    mxr = metpy.calc.mixing_ratio_from_relative_humidity(
        units.Quantity(input_era5["air_pressure"], "Pa"),
        units.Quantity(input_era5["air_temperature"], "K"),
        units.Quantity(input_era5["relative_humidity"], "dimensionless"),
    )
    rho = metpy.calc.density(
        units.Quantity(input_era5["air_pressure"], "Pa"),
        units.Quantity(input_era5["air_temperature"], "K"),
        mxr,
    )
    input_era5["lwc_in"] = clwc * rho.magnitude
    input_era5["time"] = seconds_since_epoch(date_i)

    return _add_std_atm(input_era5)


def prepare_radiosonde(rs_data: nc.Dataset) -> dict:
    """Prepare input data from RS41-GDP radiosonde measurements."""
    input_rs: dict = {}
    geopotential = units.Quantity(rs_data["geopotential_height"][:] * con.g0, "m^2/s^2")
    input_rs["height"] = metpy.calc.geopotential_to_height(geopotential[:]).magnitude
    input_rs["height"] = input_rs["height"][:]
    input_rs["air_temperature"] = rs_data["air_temperature"][:] + con.T0
    input_rs["relative_humidity"] = rs_data["relative_humidity"][:] / 100.0
    input_rs["air_pressure"] = rs_data["air_pressure"][:] * 100.0
    input_rs["time"] = seconds_since_epoch(rs_data.BEZUGSDATUM_SYNOP)

    return _add_std_atm(input_rs)


def prepare_vaisala(vs_data: nc.Dataset, altitude: float) -> dict:
    """Prepare input data from JOYCE Vaisala radiosonde measurements."""
    input_vs: dict = {}
    geopotential = units.Quantity(vs_data["alt"][:] * con.g0, "m^2/s^2")
    input_vs["height"] = metpy.calc.geopotential_to_height(geopotential[:]).magnitude
    input_vs["height"] = input_vs["height"][0, :]
    input_vs["air_temperature"] = vs_data.variables["ta"][0, :]
    input_vs["air_pressure"] = vs_data.variables["p"][0, :]
    input_vs["relative_humidity"] = vs_data["rh"][0, :] / 100.0
    input_vs["time"] = seconds_since_epoch(
        vs_data.date_YYYYMMDDTHHMM[0:8] + vs_data.date_YYYYMMDDTHHMM[9:11]
    )

    return _add_std_atm(input_vs)


def prepare_icon(icon_data: nc.Dataset, index: int, date_i: str) -> dict:
    """Prepare input data from ICON model (METEOGRAM)."""
    input_icon = {
        "height": np.flip(icon_data["height_2"][:]) - icon_data["height_2"][-1],
        "air_temperature": np.flip(icon_data["T"][index, :]),
        "air_pressure": np.flip(icon_data["P"][index, :]),
        "relative_humidity": np.flip(icon_data["REL_HUM"][index, :]) / 100.0,
    }
    input_icon["lwc_in"] = np.flip(icon_data["QC"][index, :]) * moist_rho_rh(
        input_icon["air_pressure"],
        input_icon["air_temperature"],
        input_icon["relative_humidity"],
    )
    input_icon["time"] = seconds_since_epoch(date_i)
    input_icon["iwv"] = icon_data["TQV"][index]

    return _add_std_atm(input_icon)


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
    }
    input_sa["relative_humidity"] = q2rh(
        sa_data.variables["q_atmo"][:, prof].astype(np.float64) * 1000.0,
        input_sa["air_temperature"],
        input_sa["air_pressure"],
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
    var_names = ["height", "air_temperature", "air_pressure", "relative_humidity"]
    if len(ind_sa) > 0 and input_dat["height"][-1] < h_val * 1000.0:
        for var in var_names:
            input_dat[var] = np.append(input_dat[var], sa[var][ind_sa])
        if "lwc_in" in input_dat:
            input_dat["lwc_in"] = np.append(
                input_dat["lwc_in"], np.zeros(len(ind_sa), dtype=np.float64)
            )

    return _add_vars(input_dat)


def _add_vars(input_dat) -> dict:
    """Add variables to input data dictionary."""
    input_dat["absolute_humidity"] = abs_hum(
        input_dat["air_temperature"][:], input_dat["relative_humidity"][:]
    )
    input_dat["e"] = calc_rho(
        input_dat["air_temperature"][:], input_dat["relative_humidity"][:]
    )
    input_dat["mixr"] = mixr(
        input_dat["air_temperature"][:],
        input_dat["absolute_humidity"][:],
        input_dat["air_pressure"][0],
        input_dat["height"][:],
    )
    if "iwv" not in input_dat:
        input_dat["iwv"] = (
            hum_to_iwv(
                input_dat["absolute_humidity"][:],
                input_dat["height"][:],
            )
            if not np.any(input_dat["absolute_humidity"].mask)
            else -999.0
        )

    return input_dat


def check_height(input_dict: dict, altitude: float, tolerance: float = 100.0) -> dict:
    """Compare input data height with altitude and cut/extrapolate if necessary.
    In addition, calculate integrated water vapor (iwv) from absolute humidity.

    Parameters
    input_dict : dict
        Input data dictionary containing height and other atmospheric variables.
    altitude : float
        Altitude to compare with the input data height.
    tolerance : float, optional
        Tolerance for the height comparison, by default 100 m.

    Returns:
    dict
        Updated input data dictionary with height adjusted to the specified altitude.
    """
    if input_dict["height"][0] - altitude < -tolerance:
        # Cut off the first part of the profile
        for key in input_dict:
            if key not in ("height", "time") and len(input_dict[key]) == len(
                input_dict["height"]
            ):
                input_dict[key] = input_dict[key][input_dict["height"][:] >= altitude]
        input_dict["height"] = input_dict["height"][input_dict["height"][:] >= altitude]

    elif input_dict["height"][0] - altitude > tolerance:
        # Extrapolate the first part of the profile
        for key in input_dict:
            if key != "height" and len(input_dict[key]) == len(input_dict["height"]):
                input_dict[key] = np.insert(
                    input_dict[key][:],
                    0,
                    make_interp_spline(input_dict["height"][:4], input_dict[key][:4])(
                        np.array([altitude])
                    ),
                )
        input_dict["height"] = np.insert(input_dict["height"], 0, altitude)

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
