import contextlib
import datetime
import os
import subprocess

import numpy as np
import suncalc
from pylblrtm.tape5_writer import makeFile

from mwrpy_sim.utils import loadCoeffsJSON


def calc_ir_rt(
    input_dat: dict,
    lwc: np.ndarray,
    base: np.ndarray,
    top: np.ndarray,
    params: dict,
) -> np.ndarray:
    """Calculate IR radiative transfer.
    LNFL, LBLRTM, and LBLDIS are maintained by the Radiation and Climate Group
    of Atmospheric and Environmental Research R&C.
    TAPE5 file writer (makeFile) is part of PyLBLRTM.
    Input:
        input_dat: Dictionary containing atmospheric data.
        lwc: Liquid water content (LWC) profile (kg/m^3).
        base: Base heights of cloud layers (m).
        top: Top heights of cloud layers (m).
        params: Dictionary containing simulation parameters.
    Output:
        IR brightness temperatures for 3 IRT channels.
    """
    lbl_out = str(os.path.join(params["data_out"] + "lblout/"))
    if not os.path.exists(lbl_out):
        os.makedirs(lbl_out)

    try:
        # Create a TAPE5 file for LBLRTM
        profiles: dict = {
            "wvmr": input_dat["mixr"][:] * 1000.0,
            "tmpc": input_dat["air_temperature"][:] - 273.15,
            "pres": input_dat["air_pressure"][:] / 100.0,
            "hght": (input_dat["height"][:] - input_dat["height"][0]) / 1000.0,
        }
        with contextlib.redirect_stdout(None):
            makeFile(
                "TAPE5",
                650,
                1150,
                0,
                IXSECT=1,
                ZNBD=np.array(params["height"][:], np.float32) / 1000.0,
                IEMIT=0,
                profiles=profiles,
            )

        # Link TAPE3 to the current working directory
        tape3_pth = os.path.dirname(os.path.realpath(__file__)) + "/coeff/TAPE3"
        subprocess.call(f"ln -s {tape3_pth} {os.getcwd()}/TAPE3", shell=True)

        # Run LBLRTM to create OD files for gaseous absorbers
        subprocess.run(
            ["lblrtm"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        subprocess.call(f"mv TAPE* ODdeflt* TMPX* {lbl_out}", shell=True)

        # Calculate tau
        tau, reff = np.zeros(len(base), np.float32), np.zeros(len(base), np.float32)
        lwc_bnd = [0.0, 2e-4, 4e-4]
        reff_a = [4.0, 6.0, 20.0]  # from Karstens et al. [1994]
        if len(base) > 0:
            for i in range(len(base)):
                b_i = np.where(input_dat["height"][:] == base[i])[0][0]
                t_i = np.where(input_dat["height"][:] == top[i])[0][0]
                lwp = 0.0
                if t_i == b_i and b_i - 1 >= 0 and t_i + 1 < len(input_dat["height"]):
                    lwp = lwc[b_i] * (
                        input_dat["height"][b_i : t_i + 1]
                        - input_dat["height"][b_i - 1 : t_i]
                    )
                    reff[i] = reff_a[np.argwhere(lwc[b_i] >= lwc_bnd)[-1][0]]
                elif (
                    t_i == b_i + 1
                    and b_i - 1 >= 0
                    and t_i + 1 < len(input_dat["height"])
                ):
                    lwp = np.sum(
                        np.take(lwc, [b_i, t_i])
                        * (
                            input_dat["height"][b_i : t_i + 1]
                            - input_dat["height"][b_i - 1 : t_i]
                        )
                    )
                    reff[i] = reff_a[
                        np.argwhere(np.max(np.take(lwc, [b_i, t_i])) > lwc_bnd)[-1][0]
                    ]
                elif t_i > b_i + 1:
                    lwp = np.sum(
                        (lwc[b_i : t_i - 1] + lwc[b_i + 1 : t_i])
                        / 2.0
                        * np.diff(input_dat["height"][b_i:t_i])
                    )
                    reff[i] = reff_a[
                        np.argwhere(
                            np.max((lwc[b_i : t_i - 1] + lwc[b_i + 1 : t_i]) / 2.0)
                            > lwc_bnd
                        )[-1][0]
                    ]
                tau[i] = 3.0 / 2.0 * lwp / (1000.0 * reff[i] * 1e-6)

        # Make parameter file for LBLDIS
        make_lbldis_file(
            lbl_out,
            params,
            650.0,
            1150.0,
            int(input_dat["time"]),
            base / 1000.0,
            tau,
            reff,
        )

        # Run LBLDIS
        subprocess.run(
            ["lbldis", f"{lbl_out}lbldis.param", "0", f"{lbl_out}lbldisout"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Read the output file
        with open(f"{lbl_out}lbldisout", "rb") as f:
            lines = f.readlines()
            lines_s = [line.strip().split() for line in lines[2:]]
            wvnum = np.array([float(line[0]) for line in lines_s])
            rad = np.array([float(line[1]) for line in lines_s])
    except (
        ValueError,
        FileNotFoundError,
        RuntimeWarning,
        subprocess.CalledProcessError,
    ) as e:
        print(f"Error during IR radiative transfer calculation: {e}")
        subprocess.call(f"rm ODdeflt* TMPX* TAPE*", shell=True)
        subprocess.call([f"rm {lbl_out}/*"], shell=True)
        return np.ones((3,), np.float32) * -999.0

    # Clean up temporary files
    subprocess.call([f"rm {lbl_out}/*"], shell=True)

    return irspectrum2irt(rad, wvnum)


def make_lbldis_file(
    lbl_out: str,
    params: dict,
    v1: float,
    v2: float,
    time: int,
    base: np.ndarray,
    tau: np.ndarray,
    reff: np.ndarray,
) -> None:
    """Create a LBLDIS parameter file."""
    # Specify paths of solar source and scattering files
    solar_src = (
        os.path.dirname(os.path.realpath(__file__))
        + "/coeff/solar.kurucz.rad.1cm-1binned.full_disk.asc"
    )
    ssp_wat = (
        os.path.dirname(os.path.realpath(__file__))
        + "/coeff/ssp_db.mie_wat.gamma_sigma_0p100"
    )
    ssp_ice = (
        os.path.dirname(os.path.realpath(__file__))
        + "/coeff/ssp_db.mie_ice.gamma_sigma_0p100"
    )

    # Calculate solar position
    time_sun = datetime.datetime.fromtimestamp(time, tz=datetime.timezone.utc)
    lat = params["latitude"]
    lng = params["longitude"]
    sol = suncalc.get_position(time_sun, lat=lat, lng=lng)
    sol_alt = np.round(sol["altitude"] * 180.0 / np.pi, 2)
    sol_azi = np.round(sol["azimuth"] * 180.0 / np.pi + 180.0, 2)

    # Write the LBLDIS parameter file
    with open(os.path.join(lbl_out, "lbldis.param"), "w") as f:
        f.write("LBLDIS parameter file\n")
        f.write("16\t\t" + "Number of streams\n")
        f.write(
            f"{sol_alt} {sol_azi} 1.0\t"
            + "Solar zenith angle (deg), relative azimuth (deg), solar distance ("
            "a.u.)\n"
        )
        f.write(
            "180.\t\t" + "Zenith angle (degrees): 0 -> upwelling, 180 -> downwelling\n"
        )
        f.write(f"{v1} {v2} 1.0\t" + "v_start, v_end, and v_delta [cm-1]\n")
        f.write(
            "1\t\t"
            + "Cloud parameter option flag: 0: reff and numdens, >=1:  reff and tau\n"
        )
        if len(base) > 0:
            f.write(f"{len(base)}\t\t" + "Number of cloud layers\n")
            for i in range(len(base)):
                f.write(
                    f"0 {np.round(base[i], 3)} {reff[i]} -1 {np.round(tau[i], 3)}\n"
                )
        else:
            f.write("0\t\t" + "Number of cloud layers\n")
        f.write(f"{lbl_out}\n")
        f.write(f"{solar_src}\n")
        f.write("2\t\t" + "Number of scattering property databases\n")
        f.write(f"{ssp_wat}\n")
        f.write(f"{ssp_ice}\n")
        f.write(
            "-1.\t\t"
            + "Surface temperature (specifying a negative value takes the value from profile)\n"
        )
        f.write("4\t\t" + "Number of surface spectral emissivity lines (wnum, emis)\n")
        f.write("100  0.985\n")
        f.write("700  0.950\n")
        f.write("800  0.970\n")
        f.write("3000 0.985\n")


def irspectrum2irt(
    ir_spectrum: np.ndarray,
    wavenumber: np.ndarray,
) -> np.ndarray:
    """Convert IR spectrum to broadband IRT."""
    # Load spectral response function
    path = os.path.dirname(os.path.realpath(__file__)) + "/coeff/irt_srf.json"
    srf = loadCoeffsJSON(path)

    # Calculate IRT
    irt = np.zeros((3,), np.float32)
    for i in range(3):
        srf_wnum = np.pad(
            np.flip(10000.0 / srf[f"wnum_irt{i + 1}"]),
            (1, 1),
            "constant",
            constant_values=(0.0, 3000.0),
        )
        srf_resp = np.pad(
            np.flip(srf[f"srf_irt{i + 1}"]),
            (1, 1),
            "constant",
            constant_values=(0.0, 0.0),
        )
        weight = np.interp(wavenumber, srf_wnum, srf_resp, left=0.0, right=0.0)
        mwnum = np.divide(np.sum(wavenumber * weight), np.sum(weight))
        mrad = np.divide(np.sum(ir_spectrum * weight), np.sum(weight))
        irt[i] = invplanck(mwnum, mrad)

    return irt


def invplanck(wavenumber: float, radiance: float) -> float:
    """Calculate brightness temperature from wavenumber and radiance."""
    c1 = 1.191042722e-12
    c2 = 1.4387752
    c1 = c1 * 1e7

    return c2 * wavenumber / (np.log(1.0 + (c1 * wavenumber**3 / radiance)))
