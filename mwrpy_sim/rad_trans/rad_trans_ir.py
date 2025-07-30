import contextlib
import datetime
import os

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
    reff: float = 6.5,
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
        reff: Effective radius of cloud droplets (µm) (default is 6.5 µm).
    Output:
        IR brightness temperatures for 3 IRT channels.
    """
    tape_out = str(os.path.join(params["data_out"] + "TAPE/"))
    if not os.path.exists(tape_out):
        os.makedirs(tape_out)

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
            ZNBD=np.array(params["height"][:], np.float32) / 1000.0,
            IEMIT=0,
            profiles=profiles,
        )

    # Run LNFL to create TAPE3 file for LBLRTM
    os.system(f"lnfl {tape_out}TAPE1 TAPE5 >/dev/null 2>&1")

    # Run LBLRTM to create OD files for gaseous absorbers
    os.system(f"lblrtm >/dev/null 2>&1")
    os.system(f"mv TAPE* {tape_out}")
    os.system(f"mv ODdeflt* {tape_out}")

    # Calculate tau
    tau = np.zeros(len(base), np.float32)
    if len(base) > 0:
        for i in range(len(base)):
            b_i = np.where(input_dat["height"][:] == base[i])[0][0]
            t_i = np.where(input_dat["height"][:] == top[i])[0][0]
            lwp = np.sum(lwc[b_i:t_i] * np.diff(input_dat["height"][b_i : t_i + 1]))
            tau[i] = 3.0 / 2.0 * lwp / (1000.0 * reff * 1e-6)

    # Make parameter file for LBLDIS
    make_lbldis_file(params, 650, 1150, input_dat["time"], base / 1000.0, tau, reff)

    # Run LBLDIS
    os.system(f"lbldis {tape_out}lbldis.param 0 {tape_out}lbldisout >/dev/null 2>&1")

    # Read the output file
    with open(f"{tape_out}lbldisout", "rb") as f:
        lines = f.readlines()
        lines_s = [line.strip().split() for line in lines[2:]]
        wvnum = np.array([float(line[0]) for line in lines_s])
        rad = np.array([float(line[1]) for line in lines_s])

    return irspectrum2irt(rad, wvnum)


def make_lbldis_file(
    params: dict,
    v1: float,
    v2: float,
    time: int,
    base: np.ndarray,
    tau: np.ndarray,
    reff: float,
) -> None:
    """Create a LBLDIS parameter file."""
    # Specify output directory, solar source and scattering files
    data_out = params["data_out"] + "TAPE/"
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
    with open(os.path.join(data_out, "lbldis.param"), "w") as f:
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
                f.write(f"0 {np.round(base[i], 3)} {reff} -1 {np.round(tau[i], 3)}\n")
        else:
            f.write("0\t\t" + "Number of cloud layers\n")
        f.write(f"{data_out}\n")
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
    # Load spetral response function
    path = os.path.dirname(os.path.realpath(__file__)) + "/coeff/irt_srf.json"
    srf = loadCoeffsJSON(path)

    # Calculate IRT
    irt = np.zeros((3, 1), np.float32)
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
        mwnum = np.sum(wavenumber * weight) / np.sum(weight)
        mrad = np.sum(ir_spectrum * weight) / np.sum(weight)
        irt[i] = invplanck(mwnum, mrad)

    return irt


def invplanck(wavenumber: float, radiance: float) -> float:
    """Calculate brightness temperature from wavenumber and radiance."""
    C1 = 1.191042722e-12
    C2 = 1.4387752
    C1 = C1 * 1e7

    return C2 * wavenumber / (np.log(1.0 + (C1 * wavenumber**3 / radiance)))
