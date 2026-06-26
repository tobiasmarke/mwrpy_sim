import numpy as np
import xarray as xr
from openMWR.cloud import CloudColumn, CloudModelConfig
from openMWR.run_RT import _get_cloud_top_base
from pyrtlib.weighting_functions import WeightingFunctions
from torchMWRT import AtmProfile, RTModel

from mwrpy_sim.data_tools.stability_indices import calc_stability_indices
from mwrpy_sim.rad_trans.rad_trans_ir import run_rad_trans_ir
from mwrpy_sim.utils import interpolate_2d_nearest, read_config


def rad_trans(
    input_dat: dict,
    params: dict,
) -> dict:
    """Run radiative transfer calculations for one atmospheric profile."""
    FillValue = np.nan
    config = read_config(None, "global_specs")
    tb, tb_pro, tb_clr = (
        np.ones(
            (len(params["frequency"]), len(np.array(params["elevation_angle"]))),
            np.float32,
        )
        * FillValue
        for _ in range(3)
    )
    irt, irt_pro, irt_clr = (
        np.ones(len(params["wavelength"]), np.float32) * FillValue for _ in range(3)
    )
    lwp, lwp_pro = (FillValue for _ in range(2))
    base, base_pro, top, top_pro = (
        np.ones(15, np.float32) * FillValue for _ in range(4)
    )
    lwc_tmp, lwp_tmp, base_tmp, top_tmp = (
        np.zeros(len(input_dat["height"][:]), np.float32),
        0.0,
        np.ones(15, np.int32) * FillValue,
        np.ones(15, np.int32) * FillValue,
    )

    # Cloud geometry [m] / cloud water content (LWC, LWP)
    cloud_methods = (
        ("prognostic", "detected", "clear")
        if "lwc_in" in input_dat
        else ("detected", "clear")
    )
    for method in cloud_methods:
        if method in ("prognostic", "detected"):
            cloud_column = CloudColumn(
                input_dat["height"][:],
                input_dat["air_pressure"][:],
                input_dat["air_temperature"][:],
                input_dat["relative_humidity"][:],
                CloudModelConfig(
                    karstens=True,
                ),
            )
            if method == "prognostic":
                lwc_tmp[:] = input_dat["lwc_in"][:]
                cloud_column.lwc = lwc_tmp[:] * 1000.0
            elif method == "detected":
                lwc_tmp[:] = cloud_column.calculate_lwc() / 1000.0
            lwp_tmp = float(cloud_column.calculate_lwp() / 1000.0)
            c_top, c_base = _get_cloud_top_base(lwc_tmp[:])
            if len(c_top) in np.arange(15) + 1:
                top_tmp[0 : len(c_top)], base_tmp[0 : len(c_base)] = (
                    input_dat["height"][c_top],
                    input_dat["height"][c_base],
                )
            elif len(top_tmp) > 15:
                continue

        # Avoid extra "clear" RT calculation for cases without liquid water
        if method == "clear" and lwp == 0.0:
            tb_clr, irt_clr = np.copy(tb), np.copy(irt)
        elif method == "clear" and lwp_pro == 0.0:
            tb_clr, irt_clr = np.copy(tb_pro), np.copy(irt_pro)
        else:
            # MW radiative transport
            atm_profile = AtmProfile(
                temperature=input_dat["air_temperature"][:],
                height=input_dat["height"][:],
                pressure=input_dat["air_pressure"][:] / 100.0,
                rh=input_dat["relative_humidity"][:],
                lwc=lwc_tmp * 1000.0,
            )
            rtmodel = RTModel(
                freqs=np.array(params["frequency"]),
                angles=np.array(params["elevation_angle"]),
                absmdl=config["mw_model"],
            )
            ds = rtmodel.execute(atm_profile, return_ds=True)
            tb_tmp = ds["tbtotal"].values

            if config["calc_ir"]:
                ds = xr.Dataset(
                    data_vars=dict(
                        T=(["height"], input_dat["air_temperature"]),
                        p=(["height"], input_dat["air_pressure"] / 100.0),
                        rh=(["height"], input_dat["relative_humidity"] * 100.0),
                        lwc=(["height"], lwc_tmp * 1000.0),
                    ),
                    coords=dict(
                        height=("height", input_dat["height"][:]),
                    ),
                )
                irt_tmp = run_rad_trans_ir(ds, params).to_numpy()
            else:
                irt_tmp = np.ones((len(params["wavelength"])), np.float32) * FillValue

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
        "time": np.ma.masked_invalid(np.asarray([input_dat["time"]])),
        "tb": np.ma.masked_invalid(np.expand_dims(tb, 0)),
        "tb_pro": np.ma.masked_invalid(np.expand_dims(tb_pro, 0)),
        "tb_clr": np.ma.masked_invalid(np.expand_dims(tb_clr, 0)),
        "irt": np.ma.masked_invalid(np.expand_dims(irt, 0)),
        "irt_pro": np.ma.masked_invalid(np.expand_dims(irt_pro, 0)),
        "irt_clr": np.ma.masked_invalid(np.expand_dims(irt_clr, 0)),
        "lwp": np.ma.masked_invalid(np.asarray([lwp])),
        "lwp_pro": np.ma.masked_invalid(np.asarray([lwp_pro])),
        "iwv": np.ma.masked_invalid(input_dat["iwv"]),
        "cbh": np.ma.masked_invalid(np.expand_dims(base, 0)),
        "cbh_pro": np.ma.masked_invalid(np.expand_dims(base_pro, 0)),
        "cth": np.ma.masked_invalid(np.expand_dims(top, 0)),
        "cth_pro": np.ma.masked_invalid(np.expand_dims(top_pro, 0)),
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
            output[var] = np.ma.masked_invalid(
                np.expand_dims(
                    np.interp(
                        params["height"],
                        input_dat["height"][:] - input_dat["height"][0],
                        input_dat[var][:],
                    ),
                    0,
                )
            )
        else:
            output[var] = np.ma.masked_all((1, len(params["height"])), np.float32)

    # Compute weighting functions
    wf = WeightingFunctions(
        input_dat["height"][:] / 1000.0,
        input_dat["air_pressure"][:] / 100.0,
        input_dat["air_temperature"][:],
        input_dat["relative_humidity"][:],
    )
    wf.frequencies = np.array(params["frequency"])
    wf.satellite = False
    wgf = np.ma.array(wf.generate_wf() / 1000.0)
    output["weighting_function"] = interpolate_2d_nearest(
        params["frequency"][:],
        np.arange(0.0, input_dat["height"][-1] + 1000.0, 1000),
        wgf,
        params["frequency"][:],
        params["height"][:],
    )

    # Calculate stability indices
    stab_dict = calc_stability_indices(output, params["height"][:])

    return dict(output, **stab_dict)
