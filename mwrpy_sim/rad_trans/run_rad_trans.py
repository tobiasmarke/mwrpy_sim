import numpy as np

from mwrpy_sim.data_tools.cloud_mod import (
    detect_cloud_mod,
    detect_liq_cloud,
    get_cloud_prop,
)
from mwrpy_sim.data_tools.stability_indices import calc_stability_indices
from mwrpy_sim.rad_trans import calc_mw_rt


def rad_trans(
    input_dat: dict,
    params: dict,
    coeff_bdw: dict,
    ape_ang: np.ndarray,
) -> dict:
    """Run radiative transfer calculations for one atmospheric profile."""
    FillValue = -999.0
    theta = 90.0 - np.array(params["elevation_angle"])
    tb = np.ones((1, len(params["frequency"]), len(theta)), np.float32) * FillValue
    tb_pro = np.ones((1, len(params["frequency"]), len(theta)), np.float32) * FillValue
    tb_clr = np.ones((1, len(params["frequency"]), len(theta)), np.float32) * FillValue
    lwp, lwp_pro = FillValue, FillValue

    # Cloud geometry [m] / cloud water content (LWC, LWP)
    cloud_methods = (
        ("prognostic", "detected", "clear")
        if "lwc_in" in input_dat
        else ("detected", "clear")
    )
    for method in cloud_methods:
        if method == "clear":
            lwc_tmp, lwp_tmp = np.zeros(len(input_dat["height"][:]), np.float32), 0.0
        else:
            top, base = (
                detect_cloud_mod(input_dat["height"][:], input_dat["lwc_in"][:])
                if method == "prognostic"
                else (
                    detect_liq_cloud(
                        input_dat["height"][:],
                        input_dat["air_temperature"][:],
                        input_dat["relative_humidity"][:],
                        input_dat["air_pressure"][:],
                    )
                )
            )
            lwc_tmp, lwp_tmp = (
                get_cloud_prop(base, top, input_dat, method)
                if len(top) in np.linspace(1, 15, 15)
                else (np.zeros(len(input_dat["height"][:]), np.float32), 0.0)
            )

        # Avoid extra "clear" RT calculation for cases without liquid water
        if method == "clear" and lwp == 0.0:
            tb_clr = np.copy(tb)
        elif method == "clear" and lwp_pro == 0.0:
            tb_clr = np.copy(tb_pro)
        else:
            # Radiative transport
            tb_tmp = np.array(
                [
                    calc_mw_rt(
                        input_dat["height"][:],
                        input_dat["air_temperature"][:],
                        input_dat["air_pressure"][:] / 100.0,
                        input_dat["e"],
                        lwc_tmp,
                        ang,
                        np.array(params["frequency"]),
                        coeff_bdw,
                        ape_ang,
                    )
                    for _, ang in enumerate(theta)
                ],
                np.float32,
            ).T
            if method == "prognostic":
                lwp_pro, tb_pro, input_dat["lwc_pro"] = lwp_tmp, tb_tmp, lwc_tmp
            elif method == "detected":
                lwp, tb, input_dat["lwc"] = lwp_tmp, tb_tmp, lwc_tmp
            else:
                tb_clr = tb_tmp

    # Make output dictionary and interpolate to final grid
    output = {
        "time": np.asarray([input_dat["time"]]),
        "tb": np.expand_dims(tb, 0),
        "tb_pro": np.expand_dims(tb_pro, 0),
        "tb_clr": np.expand_dims(tb_clr, 0),
        "lwp": np.asarray([lwp]),
        "lwp_pro": np.asarray([lwp_pro]),
        "iwv": np.asarray([input_dat["iwv"]]),
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
