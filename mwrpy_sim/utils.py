import datetime
import glob
import json
import logging
import os
from datetime import timezone
from typing import Any, Iterator, Literal, NamedTuple

import numpy as np
import yaml
from numpy import ma
from yaml.loader import SafeLoader

Epoch = tuple[int, int, int]


class MetaData(NamedTuple):
    dimensions: tuple[str, ...]
    long_name: str
    units: str
    standard_name: str | None = None
    definition: str | None = None
    comment: str | None = None


def append_data(data_in: dict, output_dict: dict) -> dict:
    """Appends data to a dictionary field (creates the field if not yet present).

    Args:
        data_in: Dictionary where data will be appended.
        output_dict: Dictionary with data to append.

    """
    data = data_in.copy()
    for key, array in output_dict.items():
        data[key] = array if key not in data else ma.concatenate((data[key], array))
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


def isodate2date(date_str: str, fmt: str = "%Y-%m-%d") -> datetime.date:
    return datetime.datetime.strptime(date_str, fmt).date()


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
    time_in_seconds = (
        datetime.datetime.timestamp(
            datetime.datetime(
                *(int(date[0:4]), int(date[4:6]), int(date[6:8]), int(date[8:])),
                tzinfo=timezone.utc,
            )
        )
        if len(date) == 10
        else datetime.datetime.timestamp(
            datetime.datetime(
                *(
                    int(date[0:4]),
                    int(date[4:6]),
                    int(date[6:8]),
                    int(date[8:10]),
                    int(date[10:]),
                ),
                tzinfo=timezone.utc,
            )
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
