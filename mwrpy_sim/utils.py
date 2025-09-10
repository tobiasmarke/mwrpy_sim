import datetime
import glob
import json
import logging
import os
import warnings
from datetime import timezone
from typing import Any, Iterator, Literal, NamedTuple, Tuple

import numpy as np
import yaml
from numpy import ma
from yaml.loader import SafeLoader

Epoch = tuple[int, int, int]


class MetaData(NamedTuple):
    long_name: str
    units: str
    standard_name: str | None = None
    definition: str | None = None
    comment: str | None = None


def append_data(data_in: dict, key: str, array: np.ndarray) -> dict:
    """Appends data to a dictionary field (creates the field if not yet present).

    Args:
        data_in: Dictionary where data will be appended.
        key: Key of the field.
        array: Numpy array to be appended to data_in[key].

    """
    data = data_in.copy()
    if key not in data:
        data[key] = array
    else:
        data[key] = ma.concatenate((data[key], array))
    return data


def get_file_list(path_to_files: str, key: str, ext: str = "nc") -> list[str]:
    """Returns file list for specified path."""
    f_list = sorted(glob.glob(path_to_files + "*" + key + "*." + ext))
    if len(f_list) == 0:
        logging.warning("Error: no files found in directory %s", path_to_files)
    return f_list


def get_processing_dates(args) -> tuple[str, str]:
    """Returns processing dates."""
    if args.date is not None:
        start_date = args.date
        stop_date = start_date
    else:
        start_date = args.start
        stop_date = args.stop
    start_date = str(date_string_to_date(start_date))
    stop_date = str(date_string_to_date(stop_date))
    return start_date, stop_date


def _get_filename(
    source: str, start: datetime.date, stop: datetime.date, site: str
) -> str:
    params = read_config(site, "params")
    if site == "standard_atmosphere":
        filename = f"{site}.nc"
    elif (stop - start).total_seconds() == 0.0:
        filename = f"{site}_{source}_{start.strftime('%Y%m%d')}.nc"
    else:
        filename = (
            f"{site}_{source}_{start.strftime('%Y%m%d')}_{stop.strftime('%Y%m%d')}.nc"
        )

    return str(os.path.join(params["data_out"] + filename))


def isodate2date(date_str: str) -> datetime.date:
    return datetime.datetime.strptime(date_str, "%Y-%m-%d").date()


def date_range(
    start_date: datetime.date, end_date: datetime.date
) -> Iterator[datetime.date]:
    """Returns range between two dates (datetimes)."""
    if start_date == end_date:
        end_date += datetime.timedelta(days=1)
    for n in range(int((end_date - start_date).days)):
        yield start_date + datetime.timedelta(n)


def seconds2date(time_in_seconds: int, epoch: Epoch = (2001, 1, 1)) -> str:
    """Converts seconds since some epoch to datetime (UTC).

    Args:
        time_in_seconds: Seconds since some epoch.
        epoch: Epoch, default is (2001, 1, 1) (UTC).

    Returns:
        [year, month, day, hours, minutes, seconds] formatted as '05' etc (UTC).

    """
    epoch_in_seconds = datetime.datetime.timestamp(
        datetime.datetime(*epoch, tzinfo=timezone.utc)
    )
    timestamp = time_in_seconds + epoch_in_seconds
    return datetime.datetime.utcfromtimestamp(timestamp).strftime("%Y%m%d%H")


def seconds_since_epoch(date: str, epoch: Epoch = (1970, 1, 1)) -> int:
    time_in_seconds = datetime.datetime.timestamp(
        datetime.datetime(
            *(int(date[0:4]), int(date[4:6]), int(date[6:8]), int(date[8:])),
            tzinfo=timezone.utc,
        )
    )
    epoch_in_seconds = datetime.datetime.timestamp(
        datetime.datetime(*epoch, tzinfo=timezone.utc)
    )
    return int(time_in_seconds) + int(epoch_in_seconds)


def str_to_numeric(value: str) -> int | float:
    """Converts string to number (int or float)."""
    try:
        return int(value)
    except ValueError:
        return float(value)


def isscalar(array: Any) -> bool:
    """Tests if input is scalar.
    By "scalar" we mean that array has a single value.

    Examples:
        >>> isscalar(1)
            True
        >>> isscalar([1])
            True
        >>> isscalar(np.array(1))
            True
        >>> isscalar(np.array([1]))
            True
    """
    arr = ma.array(array)
    if not hasattr(arr, "__len__") or arr.shape == () or len(arr) == 1:
        return True
    return False


def read_config(site: str | None, key: Literal["global_specs", "params"]) -> dict:
    data = _read_config_yaml()[key]
    if site is not None:
        data.update(_read_site_config_yaml(site)[key])
    return data


def _read_config_yaml() -> dict:
    dir_name = os.path.dirname(os.path.realpath(__file__))
    inst_file = os.path.join(dir_name, "site_config", "config.yaml")
    with open(inst_file, "r", encoding="utf8") as f:
        return yaml.load(f, Loader=SafeLoader)


def _read_site_config_yaml(site: str) -> dict:
    dir_name = os.path.dirname(os.path.realpath(__file__))
    site_file = os.path.join(dir_name, "site_config", site + ".yaml")
    if not os.path.isfile(site_file):
        raise NotImplementedError(f"Error: site config file {site_file} not found")
    with open(site_file, "r", encoding="utf8") as f:
        return yaml.load(f, Loader=SafeLoader)


def read_bandwidth_coefficients() -> dict:
    coeff_bdw: dict = {}

    path = (
        os.path.dirname(os.path.realpath(__file__))
        + "/rad_trans/coeff/o2_bandpass_interp_freqs.json"
    )
    FFI = loadCoeffsJSON(path)
    coeff_bdw["bdw_fre"] = FFI["FFI"].T

    path = (
        os.path.dirname(os.path.realpath(__file__))
        + "/rad_trans/coeff/o2_bandpass_interp_norm_resp.json"
    )
    FRIN = loadCoeffsJSON(path)
    coeff_bdw["bdw_wgh"] = FRIN["FRIN"].T

    coeff_bdw["f_all"], coeff_bdw["ind1"] = (
        np.empty(0, np.float32),
        np.zeros(1, np.int32),
    )
    for ff in range(7):
        ifr = np.where(coeff_bdw["bdw_wgh"][ff, :] > 0.0)[0]
        coeff_bdw["f_all"] = np.hstack(
            (coeff_bdw["f_all"], coeff_bdw["bdw_fre"][ff, ifr])
        )
        coeff_bdw["ind1"] = np.hstack(
            (
                coeff_bdw["ind1"],
                coeff_bdw["ind1"][ff] + len(ifr),
            )
        )

    return coeff_bdw


def read_beamwidth_coefficients() -> np.ndarray:
    ape_ini = np.linspace(-9.9, 9.9, 199)
    ape_ang = ape_ini[GAUSS(ape_ini, 0.0) > 1e-3]
    ape_ang = ape_ang[ape_ang >= 0.0]

    return ape_ang


def date_string_to_date(date_string: str) -> datetime.date:
    """Convert YYYY-MM-DD to Python date."""
    date_arr = [int(x) for x in date_string.split("-")]
    return datetime.date(*date_arr)


def get_time() -> str:
    """Returns current UTC-time."""
    return f"{datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} +00:00"


def get_date_from_past(n: int, reference_date: str | None = None) -> str:
    """Return date N-days ago.

    Args:
        n: Number of days to skip (can be negative, when it means the future).
        reference_date: Date as "YYYY-MM-DD". Default is the current date.

    Returns:
        str: Date as "YYYY-MM-DD".
    """
    reference = reference_date or get_time().split()[0]
    the_date = date_string_to_date(reference) - datetime.timedelta(n)
    return str(the_date)


def loadCoeffsJSON(path) -> dict:
    """Load coefficients required for O2 absorption."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf8") as f:
            try:
                var_all = {**json.load(f)}
                for key in var_all.keys():
                    var_all[key] = np.asarray(var_all[key])
            except json.decoder.JSONDecodeError:
                print(path)
                raise
    return {**var_all}


def dcerror(x, y):
    """SIXTH-ORDER APPROX TO THE COMPLEX ERROR FUNCTION OF z=X+iY."""
    a = [
        122.607931777104326,
        214.382388694706425,
        181.928533092181549,
        93.155580458138441,
        30.180142196210589,
        5.912626209773153,
        0.564189583562615,
    ]
    b = [
        122.607931773875350,
        352.730625110963558,
        457.334478783897737,
        348.703917719495792,
        170.354001821091472,
        53.992906912940207,
        10.479857114260399,
    ]

    ZH = complex(np.abs(y), -x)
    ASUM = (
        ((((a[6] * ZH + a[5]) * ZH + a[4]) * ZH + a[3]) * ZH + a[2]) * ZH + a[1]
    ) * ZH + a[0]
    BSUM = (
        (((((ZH + b[6]) * ZH + b[5]) * ZH + b[4]) * ZH + b[3]) * ZH + b[2]) * ZH + b[1]
    ) * ZH + b[0]
    w = ASUM / BSUM
    if y >= 0:
        DCERROR = w
    else:
        DCERROR = 2.0 * np.exp(-(complex(x, y) ** 2)) - w

    return DCERROR


def GAUSS(ape_ang, theta):
    ape_sigma = (2.35 * 0.5) / np.sqrt(-1.0 * np.log(0.5))
    arg = np.abs((ape_ang - theta) / ape_sigma)
    arg = arg[arg < 9.0]

    return np.exp(-arg * arg / 2.0) * arg


def exponential_integration(
    zeroflg: bool, x: np.ndarray, ds: np.ndarray, ibeg: int, iend: int, factor: float
) -> Tuple[float, np.ndarray]:
    """EXPonential INTegration: Integrate the profile in array x over the
    layers defined in array ds, saving the integrals over each layer.

    Args:
        zeroflg (bool): Flag to handle zero values (0:layer=0, 1:layer=avg).
        x (numpy.ndarray): Profile array.
        ds (numpy.ndarray): Array of layer depths (km).
        ibeg (int): Lower integration limit (profile level number).
        iend (int): Upper integration limit (profile level number).
        factor (float): Factor by which result is multiplied (e.g., unit change).

    Returns:
        Tuple[float, numpy.ndarray]:
        * xds (numpy.ndarray): Array containing integrals over each layer ds
        * sxds (numpy.ndarray): Integral of x*ds over levels ibeg to iend
    adapted from pyrtlib
    """
    sxds = 0.0
    xds = np.zeros(ds.shape)
    for i in range(ibeg, iend):
        # Check for negative x value. If found, output message and return.
        if x[i - 1] < 0.0 or x[i] < 0.0:
            warnings.warn("Error encountered in exponential_integration")
            return sxds, xds
            # Find a layer value for x in cases where integration algorithm fails.
        if np.abs(x[i] - x[i - 1]) < 1e-09:
            xlayer = x[i]
        elif x[i - 1] == 0.0 or x[i] == 0.0:
            if not zeroflg:
                xlayer = 0.0
            else:
                xlayer = np.dot((x[i] + x[i - 1]), 0.5)
        else:
            # Find a layer value for x assuming exponential decay over the layer.
            xlayer = (x[i] - x[i - 1]) / np.log(x[i] / x[i - 1])
        # Integrate x over the layer and save the result in xds.
        xds[i] = np.dot(xlayer, ds[i])
        sxds = sxds + xds[i]

    sxds = np.dot(sxds, factor)

    return sxds, xds.reshape(iend)
