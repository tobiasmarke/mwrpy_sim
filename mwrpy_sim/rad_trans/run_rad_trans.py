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
    lwc, lwc_pro = (
        np.zeros(len(input_dat["height"][:]), np.float32),
        np.zeros(len(input_dat["height"][:]), np.float32),
    )

    # Cloud geometry [m] / cloud water content (LWC, LWP)
    cloud_methods = (
        ("prognostic", "detected", "clear")
        if "lwc" in input_dat
        else ("detected", "clear")
    )
    for method in cloud_methods:
        if method == "clear":
            lwc_tmp, lwp_tmp = np.zeros(len(input_dat["height"][:]), np.float32), 0.0
        else:
            top, base = (
                detect_cloud_mod(input_dat["height"][:], input_dat["lwc"][:])
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
                lwp_pro, tb_pro, lwc_pro = lwp_tmp, tb_tmp, lwc_tmp
            elif method == "detected":
                lwp, tb, lwc = lwp_tmp, tb_tmp, lwc_tmp
            else:
                tb_clr = tb_tmp

    # Interpolate to final grid
    pressure_int = np.interp(
        params["height"],
        input_dat["height"][:] - input_dat["height"][0],
        input_dat["air_pressure"][:],
    )
    temperature_int = np.interp(
        params["height"],
        input_dat["height"][:] - input_dat["height"][0],
        input_dat["air_temperature"][:],
    )
    abshum_int = np.interp(
        params["height"],
        input_dat["height"][:] - input_dat["height"][0],
        input_dat["absolute_humidity"][:],
    )
    relhum_int = np.interp(
        params["height"],
        input_dat["height"][:] - input_dat["height"][0],
        input_dat["relative_humidity"][:],
    )
    lwc_int = np.interp(
        params["height"],
        input_dat["height"][:] - input_dat["height"][0],
        lwc,
    )
    lwc_pro_int = np.interp(
        params["height"],
        input_dat["height"][:] - input_dat["height"][0],
        lwc_pro,
    )

    output = {
        "time": np.asarray([input_dat["time"]]),
        "tb": np.expand_dims(tb, 0),
        "tb_pro": np.expand_dims(tb_pro, 0),
        "tb_clr": np.expand_dims(tb_clr, 0),
        "air_temperature": np.expand_dims(temperature_int, 0),
        "air_pressure": np.expand_dims(pressure_int, 0),
        "absolute_humidity": np.expand_dims(abshum_int, 0),
        "relative_humidity": np.expand_dims(relhum_int, 0),
        "lwc": np.expand_dims(lwc_int, 0),
        "lwc_pro": np.expand_dims(lwc_pro_int, 0),
        "lwp": np.asarray([lwp]),
        "lwp_pro": np.asarray([lwp_pro]),
        "iwv": np.asarray([input_dat["iwv"]]),
        # Save input data for further processing:
        "height_in": np.expand_dims(input_dat["height"][:], 0),
        "air_temperature_in": np.expand_dims(input_dat["air_temperature"][:], 0),
        "air_pressure_in": np.expand_dims(input_dat["air_pressure"][:], 0),
        "absolute_humidity_in": np.expand_dims(input_dat["absolute_humidity"][:], 0),
        "relative_humidity_in": np.expand_dims(input_dat["relative_humidity"][:], 0),
        "lwc_in": np.expand_dims(lwc, 0),
        "lwc_pro_in": np.expand_dims(lwc_pro, 0),
    }

    # Calculate stability indices
    calc_stability_indices(output, params["height"][:])

    # Convert all fields in output to masked arrays
    for key in output:
        if isinstance(output[key], np.ndarray):
            output[key] = np.ma.masked_invalid(output[key])
            output[key] = np.ma.masked_equal(output[key], FillValue)

    return output
