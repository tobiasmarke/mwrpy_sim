"""Prepare input data for MWRPy sim."""

import os

import metpy.calc
import netCDF4 as nc
import numpy as np
from metpy.units import units

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
    input_ifs = {
        "height": ifs_data["height"][index, :],
        "air_temperature": ifs_data["temperature"][index, :],
        "air_pressure": ifs_data["pressure"][index, :],
        "relative_humidity": ifs_data["rh"][index, :],
    }
    input_ifs["lwc"] = ifs_data["ql"][index, :] * moist_rho_rh(
        input_ifs["air_pressure"],
        input_ifs["air_temperature"],
        input_ifs["relative_humidity"],
    )
    input_ifs["time"] = seconds_since_epoch(date_i)

    return _add_vars(input_ifs)


def prepare_standard_atmosphere(sa_data: nc.Dataset) -> dict:
    input_sa = {
        "height": sa_data.variables["height"][:].astype(np.float64) * 1000.0,
        "air_temperature": sa_data.variables["t_atmo"][:, 0].astype(np.float64),
        "air_pressure": sa_data.variables["p_atmo"][:, 0].astype(np.float64) * 100.0,
        "absolute_humidity1": sa_data.variables["a_atmo"][:, 0].astype(np.float64),
    }
    input_sa["relative_humidity"] = q2rh(
        sa_data.variables["q_atmo"][:, 0].astype(np.float64) * 1000.0,
        input_sa["air_temperature"],
        input_sa["air_pressure"],
    )
    input_sa["time"] = 0

    return _add_vars(input_sa)


def prepare_radiosonde(rs_data: nc.Dataset) -> dict:
    input_rs: dict = {}
    geopotential = units.Quantity(
        rs_data.variables["geopotential_height"][:] * con.g0, "m^2/s^2"
    )
    input_rs["height"] = metpy.calc.geopotential_to_height(geopotential[:]).magnitude
    input_rs["height"] = input_rs["height"][:]
    input_rs["air_temperature"] = rs_data.variables["air_temperature"][:] + con.T0
    input_rs["relative_humidity"] = rs_data.variables["relative_humidity"][:] / 100.0
    input_rs["air_pressure"] = rs_data.variables["air_pressure"][:] * 100.0
    input_rs["time"] = seconds_since_epoch(rs_data.BEZUGSDATUM_SYNOP)
    # input_rs["time"] = seconds_since_epoch(
    #     str(rs_data.variables["BEZUGSDATUM_SYNOP"][-1].data)
    # )

    return _add_vars(input_rs)


def prepare_vaisala(vs_data: nc.Dataset, altitude: float) -> dict:
    input_vs: dict = {}
    sa = nc.Dataset(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        + "/tests/data/standard_atmospheres.nc"
    )
    geopotential = units.Quantity(vs_data.variables["alt"][:] * con.g0, "m^2/s^2")
    input_vs["height"] = (
        metpy.calc.geopotential_to_height(geopotential[:]).magnitude - 20.0
    )
    ind_sa = np.where(sa.variables["height"][:] * 1000.0 > 15000.0)[0]
    ind_vs = int(np.where(input_vs["height"][0, :] > altitude)[0][0])
    if input_vs["height"][0, ind_vs] - altitude > 100.0:
        raise ValueError(f"Radiosonde data too high for altitude {altitude}. ")
    input_vs["height"] = np.append(
        input_vs["height"][0, ind_vs:2800], sa.variables["height"][ind_sa] * 1000.0
    )
    input_vs["air_temperature"] = np.append(
        vs_data.variables["ta"][0, ind_vs:2800], sa.variables["t_atmo"][ind_sa, 0]
    )
    input_vs["air_pressure"] = np.append(
        vs_data.variables["p"][0, ind_vs:2800],
        sa.variables["p_atmo"][ind_sa, 0] * 100.0,
    )
    rh = q2rh(
        sa.variables["q_atmo"][ind_sa, 0] * 1000.0,
        sa.variables["t_atmo"][ind_sa, 0],
        sa.variables["p_atmo"][ind_sa, 0] * 100.0,
    )
    input_vs["relative_humidity"] = np.append(
        vs_data.variables["rh"][0, ind_vs:2800] / 100.0, rh
    )
    input_vs["time"] = seconds_since_epoch(
        vs_data.date_YYYYMMDDTHHMM[0:8] + vs_data.date_YYYYMMDDTHHMM[9:11]
    )

    return _add_vars(input_vs)


def prepare_icon(icon_data: nc.Dataset, index: int, date_i: str) -> dict:
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

    return _add_vars(input_icon)


def prepare_era5_mod(
    mod_data_sfc: nc.Dataset, mod_data_pro: nc.Dataset, index: int, date_i: str
) -> dict:
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
    input_era5["lwc"] = clwc * rho.magnitude
    input_era5["time"] = seconds_since_epoch(date_i)

    return _add_vars(input_era5)


def prepare_era5_pres(mod_data: nc.Dataset, index: int, date_i: str) -> dict:
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
    input_era5["lwc"] = clwc * rho.magnitude
    input_era5["time"] = seconds_since_epoch(date_i)

    return _add_vars(input_era5)


def _add_vars(input_dat) -> dict:
    """Add variables to input data dictionary."""
    if "absolute_humidity" not in input_dat:
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
