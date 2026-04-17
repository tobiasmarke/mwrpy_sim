import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from openMWR.libRadtran import libRadtran
from openMWR.paths import libRadtran_dir
from openMWR.run_RT import _effective_droplet_radius_mum

from mwrpy_sim.utils import loadCoeffsJSON


def run_rad_trans_ir(ds_date: xr.Dataset, params: dict) -> np.ndarray:
    """Module for IR radiative transfer calculations."""
    z_m = ds_date.height.values
    z_km = z_m / 1000
    T_K = ds_date.T.values
    rh_100 = ds_date.rh.values
    p_hPa = ds_date.p.values
    lwc_gpm3 = ds_date.lwc.values

    effective_droplet_radius_mum = _effective_droplet_radius_mum(lwc_gpm3)

    TB_IR = calculate_IR_band_TB(
        z_km, T_K, rh_100, p_hPa, lwc_gpm3, effective_droplet_radius_mum, params
    )
    return TB_IR


def calculate_IR_band_TB(
    z_km, T_K, rh_100, p_hPa, lwc_gpm3, effective_droplet_radius_mum, params
) -> np.ndarray:
    """Calculate IRT with spectral response function."""
    zeitstempel_ns = time.time_ns()
    data_dir = "./mwrpy_sim/rad_trans/openMWR/"
    libRadtran_data_dir = Path(libRadtran_dir(data_dir)).resolve()
    os.makedirs(libRadtran_data_dir, exist_ok=True)

    radio_file = libRadtran_data_dir / f"radio_{zeitstempel_ns}.dat"

    wc_file = libRadtran_data_dir / f"wc_{zeitstempel_ns}.dat"

    df = pd.DataFrame(
        {
            "z_km": z_km,
            "T_K": T_K,
            "rh_100": rh_100,
            "p_hPa": p_hPa,
            "lwc_gpm3": lwc_gpm3,
            "effective_droplet_radius_mum": effective_droplet_radius_mum,
        }
    )

    df = df.sort_values(by="p_hPa")
    df = df.drop_duplicates(subset=["p_hPa"])

    df.to_csv(
        radio_file,
        index=False,
        columns=["p_hPa", "T_K", "rh_100"],
        sep=" ",
        header=False,
    )

    df = df.sort_values(by="z_km", ascending=False)
    df = df[df["z_km"] < 15]

    df.to_csv(
        wc_file,
        index=False,
        columns=["z_km", "lwc_gpm3", "effective_droplet_radius_mum"],
        sep=" ",
        header=False,
    )

    umu = np.cos(np.radians(90 + np.array([90])))

    irt_tmp = np.ones((len(params["wavelength"])), np.float32) * -999.0
    for ind, wvl in enumerate(params["wavelength"]):
        if wvl == 10.5:
            wavelengths = "9300 11900"
        elif wvl == 11.1:
            wavelengths = "9000 14400"
        elif wvl == 12.0:
            wavelengths = "9900 14400"
        try:
            df_rad, _ = libRadtran(
                atmosphere_file="../data/atmmod/afglus.dat",
                source="thermal",
                mol_abs_param="reptran medium",
                wavelength=wavelengths,
                radiosonde=f"{str(radio_file)} h2o RH",
                radiosonde_levels_only=True,
                wc_file=f"1D {str(wc_file)}",
                umu=umu[::-1],  # -0.5 np.array([-1, -0.5]
                output_quantity="brightness",
                output_user="wavenumber uu",
                quiet=True,
            )
        except Exception as e:
            raise e

        ir_spec = _planck(
            np.squeeze(df_rad["wavenumber"].values), np.squeeze(df_rad["uu"].values)
        )
        irt_tmp[ind] = _irspectrum2irt(wvl, ir_spec, df_rad["wavenumber"].values)

    for f in [radio_file, wc_file]:
        if f.exists():
            os.remove(f)

    return irt_tmp


def _irspectrum2irt(
    wvl: float,
    ir_spectrum: np.ndarray,
    wavenumber: np.ndarray,
) -> float:
    """Convert IR spectrum to broadband IRT."""
    # Load spectral response function
    path = os.path.dirname(os.path.realpath(__file__)) + "/irt_srf.json"
    srf = loadCoeffsJSON(path)

    # Calculate IRT
    srf_wnum = np.pad(
        np.flip(10000.0 / srf[f"wnum_{wvl}"]),
        (1, 1),
        "constant",
        constant_values=(0.0, 3000.0),
    )
    srf_resp = np.pad(
        np.flip(srf[f"srf_{wvl}"]),
        (1, 1),
        "constant",
        constant_values=(0.0, 0.0),
    )

    weight = np.interp(wavenumber, srf_wnum, srf_resp, left=0.0, right=0.0)
    mwnum = np.divide(np.sum(wavenumber * weight), np.sum(weight))
    mrad = np.divide(np.sum(ir_spectrum * weight), np.sum(weight))

    return _invplanck(mwnum, mrad)


def _invplanck(wavenumber: float, radiance: float) -> float:
    """Calculate brightness temperature from wavenumber and radiance."""
    c1 = 1.191042722e-12
    c2 = 1.4387752
    c1 = c1 * 1e7

    return c2 * wavenumber / (np.log(1.0 + (c1 * wavenumber**3 / radiance)))


def _planck(wavenumber: np.ndarray, bt: np.ndarray) -> np.ndarray:
    """Calculate radiance from wavenumber and brightness temperature."""
    c1 = 1.191042722e-12
    c2 = 1.4387752
    c1 = c1 * 1e7

    return (c1 * wavenumber**3) / (np.exp(c2 * wavenumber / bt) - 1.0)
