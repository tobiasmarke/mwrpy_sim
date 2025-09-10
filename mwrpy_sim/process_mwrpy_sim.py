import datetime
import logging
import os
import time

import netCDF4 as nc
import numpy as np

from mwrpy_sim import sim_mwr
from mwrpy_sim.data_tools.era5_download.get_era5 import era5_request
from mwrpy_sim.data_tools.prepare_input import (
    check_height,
    prepare_era5_mod,
    prepare_era5_pres,
    prepare_icon,
    prepare_ifs,
    prepare_radiosonde,
    prepare_standard_atmosphere,
    prepare_vaisala,
)
from mwrpy_sim.plot.plotting import plot_sim_data
from mwrpy_sim.rad_trans.rad_trans_meta import get_data_attributes
from mwrpy_sim.rad_trans.run_rad_trans import rad_trans
from mwrpy_sim.utils import (
    _get_filename,
    append_data,
    date_range,
    get_file_list,
    get_processing_dates,
    isodate2date,
    read_bandwidth_coefficients,
    read_beamwidth_coefficients,
    read_config,
    seconds2date,
)


def main(args):
    logging.basicConfig(level="INFO")
    start = time.process_time()
    _start_date, _stop_date = get_processing_dates(args)
    start_date = isodate2date(_start_date)
    stop_date = isodate2date(_stop_date)
    global_specs = read_config(args.site, "global_specs")
    params = read_config(args.site, "params")
    file_name = _get_filename(args.source, start_date, stop_date, args.site)
    if args.command == "get_era5":
        era5_request(args.site, params, start_date, stop_date)
    elif args.command in ("process", "no-plot"):
        data_nc = process_input(args.source, args.site, start_date, stop_date, params)
        if (len(data_nc) > 0) & ("time" in data_nc):
            output_dir = os.path.dirname(file_name)
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
            sim_in = sim_mwr.Sim(data_nc)
            sim_in.data = get_data_attributes(sim_in.data, args.source)
            logging.info(f"Saving output file {file_name}")
            sim_mwr.save_sim(sim_in, file_name, global_specs, args.source, params)
    if args.command in ("process", "plot"):
        if not os.path.isfile(file_name):
            logging.error(f"File {file_name} does not exist.")
        else:
            sim_data = nc.Dataset(file_name)
            plot_sim_data(
                sim_data,
                args.site,
                args.source,
                args.date,
                args.start,
                args.stop,
                params["plot_pth"],
            )
    elapsed_time = time.process_time() - start
    logging.info(f"Processing took {elapsed_time:.1f} seconds")


def process_input(
    source: str,
    site: str,
    start_date: datetime.date,
    stop_date: datetime.date,
    params: dict,
) -> dict:
    data_nc: dict = {}
    config = read_config(None, "global_specs")
    if source == "ifs":
        for date in date_range(start_date, stop_date):
            data_in = (
                params["data_ifs"] + date.strftime("%Y/") + date.strftime("%Y%m%d")
            )
            file_name = get_file_list(data_in, "ecmwf")
            if len(file_name) == 1:
                with nc.Dataset(file_name[0]) as ifs_data:
                    for index, hour in enumerate(ifs_data["time"][:-1]):
                        date_i = datetime.datetime.combine(
                            date, datetime.time(int(hour))
                        ).strftime("%Y%m%d%H")
                        if date_i[-2:] == "00":
                            logging.info(
                                f"Radiative transfer using {source} data "
                                f"for {site}, {date_i[:-2]}"
                            )
                        input_ifs = prepare_ifs(
                            ifs_data, index, date_i, params["altitude"]
                        )
                        if len(input_ifs) > 0:
                            try:
                                output_hour = call_rad_trans(input_ifs, params)
                            except ValueError:
                                logging.info(f"Skipping time {date_i}")
                                continue
                            for key, array in output_hour.items():
                                data_nc = append_data(data_nc, key, array)

    elif source == "era5" and config["era5"][:] == "model":
        file_names = np.array([], dtype=str)
        for f_type in ("sfc", "pro"):
            if (stop_date - start_date).total_seconds() == 0.0:
                file_names = np.append(
                    file_names,
                    get_file_list(
                        params["data_era5"],
                        site + f"_era5_input_{f_type}_" + start_date.strftime("%Y%m%d"),
                    ),
                )
            else:
                file_names = np.append(
                    file_names,
                    get_file_list(
                        params["data_era5"],
                        site
                        + f"_era5_input_{f_type}_"
                        + start_date.strftime("%Y%m%d")
                        + "_"
                        + stop_date.strftime("%Y%m%d"),
                    ),
                )
        if len(file_names) == 2:
            with (
                nc.Dataset(str(np.sort(file_names)[0])) as era5_data_pro,
                nc.Dataset(str(np.sort(file_names)[1])) as era5_data_sfc,
            ):
                for index, hour in enumerate(era5_data_sfc["valid_time"]):
                    date_i = seconds2date(hour, (1970, 1, 1))
                    if date_i[-2:] == "00":
                        logging.info(
                            f"Radiative transfer using {source} data "
                            f"for {site}, {date_i[:-2]}"
                        )
                    input_era5 = prepare_era5_mod(
                        era5_data_sfc, era5_data_pro, index, date_i
                    )
                    try:
                        output_hour = call_rad_trans(input_era5, params)
                    except ValueError:
                        logging.info(f"Skipping time {date_i}")
                        continue
                    for key, array in output_hour.items():
                        data_nc = append_data(data_nc, key, array)

    elif source == "era5" and config["era5"][:] == "pressure":
        if (stop_date - start_date).total_seconds() == 0.0:
            file_name = get_file_list(
                params["data_era5"],
                site + "_era5_input_pres_" + start_date.strftime("%Y%m%d"),
            )
        else:
            file_name = get_file_list(
                params["data_era5"],
                site
                + "_era5_input_pres_"
                + start_date.strftime("%Y%m%d")
                + "_"
                + stop_date.strftime("%Y%m%d"),
            )
        if len(file_name) == 1:
            with (
                nc.Dataset(str(np.sort(file_name)[0])) as era5_data,
            ):
                for index, hour in enumerate(era5_data["valid_time"]):
                    date_i = seconds2date(hour, (1970, 1, 1))
                    if date_i[-2:] == "00":
                        logging.info(
                            f"Radiative transfer using {source} data "
                            f"for {site}, {date_i[:-2]}"
                        )
                    input_era5 = prepare_era5_pres(era5_data, index, date_i)
                    try:
                        output_hour = call_rad_trans(input_era5, params)
                    except ValueError:
                        logging.info(f"Skipping time {date_i}")
                        continue
                    for key, array in output_hour.items():
                        data_nc = append_data(data_nc, key, array)

    elif source == "radiosonde":
        for date in date_range(start_date, stop_date):
            f_names = get_file_list(
                params["data_rs"] + date.strftime("%Y/%m/%d/"), "radiosonde"
            )
            for file in f_names:
                if os.path.isfile(file):
                    with nc.Dataset(file) as rs_data:
                        if file == f_names[0]:
                            logging.info(
                                f"Radiative transfer using {source} data "
                                f"for {site}, {date.strftime('%Y%m%d')}"
                            )
                        input_rs = prepare_radiosonde(rs_data)
                    try:
                        output_hour = call_rad_trans(input_rs, params)
                    except ValueError:
                        logging.info(f"Skipping file {file}")
                        continue
                    for key, array in output_hour.items():
                        data_nc = append_data(data_nc, key, array)

    elif source == "vaisala":
        for date in date_range(start_date, stop_date):
            file_name = get_file_list(
                params["data_vs"], site + "_" + date.strftime("%Y%m%d")
            )
            for name in file_name:
                if os.path.isfile(name):
                    with nc.Dataset(name) as vs_data:
                        if name == file_name[0]:
                            logging.info(
                                f"Radiative transfer using {source} data "
                                f"for {site}, {date.strftime('%Y%m%d')}"
                            )
                        input_vs = prepare_vaisala(vs_data, params["altitude"])
                        input_vs["height"] -= params["altitude"]
                    try:
                        output_hour = call_rad_trans(input_vs, params)
                    except ValueError:
                        logging.info(f"Skipping file {file_name[0]}")
                        continue
                    for key, array in output_hour.items():
                        data_nc = append_data(data_nc, key, array)

    elif source == "icon":
        for date in date_range(start_date, stop_date):
            file_name = get_file_list(
                params["data_icon"]
                + date.strftime("%Y/")
                + date.strftime("%m/")
                + date.strftime("%Y%m%d")
                + "_r600m_f2km/",
                "METEOGRAM_patch001_" + date.strftime("%Y%m%d") + "_joyce",
            )
            if os.path.isfile(file_name[0]) and os.path.getsize(file_name[0]) > 0:
                with nc.Dataset(file_name[0]) as icon_data:
                    _, hour_index, _ = np.intersect1d(
                        icon_data["time"][:].data / 3600.0,
                        np.linspace(0, 23, 24),
                        return_indices=True,
                    )
                    for index in hour_index:
                        date_i = datetime.datetime.combine(
                            date,
                            datetime.time(int(icon_data["time"][index].data / 3600.0)),
                        ).strftime("%Y%m%d%H")
                        if date_i[-2:] == "00":
                            logging.info(
                                f"Radiative transfer using {source} data "
                                f"for {site}, {date_i[:-2]}"
                            )
                        input_icon = prepare_icon(icon_data, index, date_i)
                        try:
                            output_hour = call_rad_trans(input_icon, params)
                        except ValueError:
                            logging.info(f"Skipping time {date_i}")
                            continue
                        for key, array in output_hour.items():
                            data_nc = append_data(data_nc, key, array)

    if source == "standard_atmosphere":
        logging.info(f"Radiative transfer using {source} data")
        input_sa = prepare_standard_atmosphere()
        data_nc = call_rad_trans(input_sa, params)

    data_nc["height"] = np.array(params["height"]) + params["altitude"]
    data_nc["frequency"] = np.array(params["frequency"])
    data_nc["wavelength"] = np.array(params["wavelength"])
    data_nc["elevation_angle"] = np.array(params["elevation_angle"])

    return data_nc


def call_rad_trans(data_in: dict, params: dict) -> dict:
    # Channel bandwidth
    coeff_bdw = read_bandwidth_coefficients()
    # Antenna beamwidth
    ape_ang = read_beamwidth_coefficients()
    data_in = check_height(data_in, params["altitude"])
    data_nc = rad_trans(
        data_in,
        params,
        coeff_bdw,
        ape_ang,
    )
    return data_nc


class SimIn:
    """Class for radiative transfer input files."""

    def __init__(self, data_in: dict):
        self.data: dict = {}
        self._init_data(data_in)

    def _init_data(self, data_in: dict):
        for key, data in data_in.items():
            self.data[key] = data
