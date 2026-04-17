import numpy as np
import xarray as xr
from openMWR.cloud import CloudColumn, CloudModelConfig
from openMWR.parallel import run_pool
from openMWR.run_RT import _get_cloud_top_base
from torchMWRT import AtmProfile, RTModel

from mwrpy_sim.data_tools.stability_indices import calc_stability_indices
from mwrpy_sim.rad_trans.rad_trans_ir import run_rad_trans_ir
from mwrpy_sim.utils import interpolate_2d_nearest, read_config


def rad_trans_day(
    input_dat: dict,
    params: dict,
) -> dict:
    """Run radiative transfer calculations for one day of atmospheric profiles."""
    FillValue = np.nan
    config = read_config(None, "global_specs")
    tb, tb_pro, tb_clr = (
        np.ones(
            (
                len(input_dat["time"][:]),
                len(params["frequency"]),
                len(np.array(params["elevation_angle"])),
            ),
            np.float32,
        )
        * FillValue
        for _ in range(3)
    )
    irt, irt_pro, irt_clr = (
        np.ones((len(input_dat["time"][:]), len(params["wavelength"])), np.float32)
        * FillValue
        for _ in range(3)
    )
    lwp, lwp_pro = (
        np.ones(len(input_dat["time"][:]), np.float32) * FillValue for _ in range(2)
    )
    base, base_pro, top, top_pro = (
        np.ones((len(input_dat["time"][:]), 15), np.float32) * FillValue
        for _ in range(4)
    )

    # Cloud geometry [m] / cloud water content (LWC, LWP)
    cloud_methods = (
        ("prognostic", "detected", "clear")
        if "lwc_in" in input_dat
        else ("detected", "clear")
    )
    for method in cloud_methods:
        lwc_tmp, lwp_tmp, base_tmp, top_tmp = (
            np.zeros(input_dat["air_temperature"][:].shape, np.float32),
            np.zeros(len(input_dat["time"][:]), np.float32),
            np.ones((len(input_dat["time"][:]), 15), np.int32) * FillValue,
            np.ones((len(input_dat["time"][:]), 15), np.int32) * FillValue,
        )
        if method in ("prognostic", "detected"):
            for itx, _ in enumerate(input_dat["time"]):
                cloud_column = CloudColumn(
                    input_dat["height"][:],
                    input_dat["air_pressure"][itx, :],
                    input_dat["air_temperature"][itx, :],
                    input_dat["relative_humidity"][itx, :],
                    CloudModelConfig(
                        karstens=True,
                    ),
                )
                if method == "prognostic":
                    lwc_tmp[itx, :] = input_dat["lwc_in"][itx, :]
                    cloud_column.lwc = lwc_tmp[itx, :] * 1000.0
                elif method == "detected":
                    lwc_tmp[itx, :] = cloud_column.calculate_lwc() / 1000.0
                lwp_tmp[itx] = float(cloud_column.calculate_lwp() / 1000.0)
                c_top, c_base = _get_cloud_top_base(lwc_tmp[itx, :])
                if len(c_top) in np.arange(15) + 1:
                    top_tmp[itx, 0 : len(c_top)], base_tmp[itx, 0 : len(c_base)] = (
                        input_dat["height"][c_top],
                        input_dat["height"][c_base],
                    )
                elif len(c_top) > 15:
                    continue

        # MW radiative transport
        atm_profile = AtmProfile(
            temperature=input_dat["air_temperature"],
            height=input_dat["height"][:],
            pressure=input_dat["air_pressure"] / 100.0,
            rh=input_dat["relative_humidity"],
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
                    T=(["time", "height"], input_dat["air_temperature"]),
                    p=(["time", "height"], input_dat["air_pressure"] / 100.0),
                    rh=(["time", "height"], input_dat["relative_humidity"] * 100.0),
                    lwc=(["time", "height"], lwc_tmp * 1000.0),
                ),
                coords=dict(
                    time=(
                        "time",
                        np.array(input_dat["time"][:], dtype="datetime64[s]"),
                    ),
                    height=("height", input_dat["height"][:]),
                ),
            )
            irt_tmp = run_pool(ds, 24, run_rad_trans_ir, params)
        else:
            irt_tmp = np.ones((len(params["wavelength"])), np.float32) * FillValue

        if method == "prognostic":
            (
                lwp_pro,
                tb_pro,
                input_dat["lwc_pro"],
                irt_pro,
                base_pro,
                top_pro,
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
                base,
                top,
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
        "time": input_dat["time"],
        "tb": np.ma.masked_invalid(tb),
        "tb_pro": np.ma.masked_invalid(tb_pro),
        "tb_clr": np.ma.masked_invalid(tb_clr),
        "irt": np.ma.masked_invalid(irt),
        "irt_pro": np.ma.masked_invalid(irt_pro),
        "irt_clr": np.ma.masked_invalid(irt_clr),
        "lwp": np.ma.masked_invalid(lwp),
        "lwp_pro": np.ma.masked_invalid(lwp_pro),
        "iwv": np.ma.masked_invalid(input_dat["iwv"]),
        "cbh": np.ma.masked_invalid(base),
        "cbh_pro": np.ma.masked_invalid(base_pro),
        "cth": np.ma.masked_invalid(top),
        "cth_pro": np.ma.masked_invalid(top_pro),
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
            output[var] = interpolate_2d_nearest(
                input_dat["time"],
                input_dat["height"][:] - input_dat["height"][0],
                input_dat[var][:, :],
                input_dat["time"],
                params["height"][:],
            )
        else:
            output[var] = np.ma.masked_all(
                (len(input_dat["time"]), len(params["height"])), np.float32
            )

    # Calculate stability indices
    stab_dict = calc_stability_indices(output, params["height"][:])

    return dict(output, **stab_dict)
