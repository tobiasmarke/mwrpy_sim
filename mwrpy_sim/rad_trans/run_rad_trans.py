import multiprocessing
from functools import partial

import numpy as np

from mwrpy_sim.data_tools.cloud_mod import (
    detect_cloud_mod,
    detect_liq_cloud,
    get_cloud_prop,
)
from mwrpy_sim.data_tools.stability_indices import calc_stability_indices
from mwrpy_sim.rad_trans import calc_ir_rt, calc_mw_rt
from mwrpy_sim.utils import read_config


def rad_trans(
    input_dat: dict,
    params: dict,
    coeff_bdw: dict,
    ape_ang: np.ndarray,
    mp: bool,
) -> dict:
    """Run radiative transfer calculations for one atmospheric profile."""
    FillValue = -999.0
    config = read_config(None, "global_specs")
    theta = 90.0 - np.array(params["elevation_angle"])
    tb, tb_pro, tb_clr = (
        np.ones((1, len(params["frequency"]), len(theta)), np.float32) * FillValue
        for _ in range(3)
    )
    irt, irt_pro, irt_clr = (
        np.ones((1, len(params["wavelength"])), np.float32) * FillValue
        for _ in range(3)
    )
    lwp, lwp_pro = (FillValue for _ in range(2))
    base, base_pro, top, top_pro = (
        np.ones(15, np.float32) * FillValue for _ in range(4)
    )

    # Cloud geometry [m] / cloud water content (LWC, LWP)
    cloud_methods = (
        ("prognostic", "detected", "clear")
        if "lwc_in" in input_dat
        else ("detected", "clear")
    )
    for method in cloud_methods:
        if method == "clear":
            lwc_tmp, lwp_tmp, base_tmp, top_tmp = (
                np.zeros(len(input_dat["height"][:]), np.float32),
                0.0,
                np.zeros(15, np.int32),
                np.zeros(15, np.int32),
            )
        else:
            top_tmp, base_tmp = (
                detect_cloud_mod(input_dat["height"][:], input_dat["lwc_in"][:])
                if method == "prognostic"
                else (
                    detect_liq_cloud(
                        input_dat["height"][:],
                        input_dat["air_temperature"][:],
                        input_dat["relative_humidity"][:],
                    )
                )
            )
            lwc_tmp, lwp_tmp = (
                get_cloud_prop(base_tmp, top_tmp, input_dat, method)
                if len(top_tmp) in np.linspace(1, 15, 15)
                else (np.zeros(len(input_dat["height"][:]), np.float32), 0.0)
            )
            top_tmp, base_tmp = (
                (np.zeros(15, np.int32), np.zeros(15, np.int32))
                if len(top_tmp) not in np.linspace(1, 15, 15)
                else (top_tmp, base_tmp)
            )

        # Avoid extra "clear" RT calculation for cases without liquid water
        if method == "clear" and lwp == 0.0:
            tb_clr, irt_clr = np.copy(tb), np.copy(irt)
        elif method == "clear" and lwp_pro == 0.0:
            tb_clr, irt_clr = np.copy(tb_pro), np.copy(irt_pro)
        else:
            # MW radiative transport
            if len(theta) > 1:
                pool = multiprocessing.Pool()
                func = partial(
                    calc_mw_rt,
                    input_dat["height"][:],
                    input_dat["air_temperature"][:],
                    input_dat["air_pressure"][:] / 100.0,
                    input_dat["e"],
                    lwc_tmp,
                    np.array(params["frequency"]),
                    coeff_bdw,
                    False,
                    ape_ang,
                )
                tb_tmp = np.squeeze(np.array([pool.map(func, theta)], np.float32).T)
                pool.close()
                pool.join()
            else:
                tb_tmp = np.array(
                    [
                        calc_mw_rt(
                            input_dat["height"][:],
                            input_dat["air_temperature"][:],
                            input_dat["air_pressure"][:] / 100.0,
                            input_dat["e"],
                            lwc_tmp,
                            np.array(params["frequency"]),
                            coeff_bdw,
                            mp,
                            ape_ang,
                            ang,
                        )
                        for _, ang in enumerate(theta)
                    ],
                    np.float32,
                ).T

            # IR radiative transport (for zenith only)
            irt_tmp = (
                calc_ir_rt(input_dat, lwc_tmp, base_tmp, top_tmp, params)
                if config["calc_ir"]
                else np.ones((len(params["wavelength"])), np.float32) * FillValue
            )

            if method == "prognostic":
                (
                    lwp_pro,
                    tb_pro,
                    input_dat["lwc_pro"],
                    irt_pro,
                    base_pro[0 : len(base_tmp)],
                    top_pro[0 : len(top_tmp)],
                ) = (
                    lwp_tmp,
                    tb_tmp,
                    lwc_tmp,
                    irt_tmp,
                    base_tmp,
                    top_tmp,
                )
            elif method == "detected":
                (
                    lwp,
                    tb,
                    input_dat["lwc"],
                    irt,
                    base[0 : len(base_tmp)],
                    top[0 : len(top_tmp)],
                ) = (
                    lwp_tmp,
                    tb_tmp,
                    lwc_tmp,
                    irt_tmp,
                    base_tmp,
                    top_tmp,
                )
            else:
                tb_clr, irt_clr = tb_tmp, irt_tmp

    # Make output dictionary and interpolate to final grid
    output = {
        "time": np.asarray([input_dat["time"]]),
        "tb": np.expand_dims(tb, 0),
        "tb_pro": np.expand_dims(tb_pro, 0),
        "tb_clr": np.expand_dims(tb_clr, 0),
        "irt": np.expand_dims(irt, 0),
        "irt_pro": np.expand_dims(irt_pro, 0),
        "irt_clr": np.expand_dims(irt_clr, 0),
        "lwp": np.asarray([lwp]),
        "lwp_pro": np.asarray([lwp_pro]),
        "iwv": input_dat["iwv"],
        "cbh": np.expand_dims(base, 0),
        "cbh_pro": np.expand_dims(base_pro, 0),
        "cth": np.expand_dims(top, 0),
        "cth_pro": np.expand_dims(top_pro, 0),
    }
    var_names = [
        "air_pressure",
        "air_temperature",
        "absolute_humidity",
        "relative_humidity",
        "lwc",
        "lwc_pro",
    ]
    for var in var_names:
        if var in input_dat:
            output[var] = np.expand_dims(
                np.interp(
                    params["height"],
                    input_dat["height"][:] - input_dat["height"][0],
                    input_dat[var][:],
                ),
                0,
            )
        else:
            output[var] = np.full((1, len(params["height"])), FillValue, np.float32)

    # Calculate stability indices
    calc_stability_indices(output, params["height"][:])

    # Convert all fields in output to masked arrays
    for key in output:
        if isinstance(output[key], np.ndarray):
            output[key] = np.ma.masked_invalid(output[key])
            output[key] = np.ma.masked_equal(output[key], FillValue)

    return output
