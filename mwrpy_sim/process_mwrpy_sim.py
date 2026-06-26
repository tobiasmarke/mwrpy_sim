import datetime
import logging
import os
import shutil

import netCDF4 as nc
import numpy as np
from openMWR.site import create_site

from mwrpy_sim import sim_mwr
from mwrpy_sim.data_tools import prepare_input as prep
from mwrpy_sim.data_tools.era5_download.get_era5 import era5_request
from mwrpy_sim.plot.plotting import plot_sim_data
from mwrpy_sim.rad_trans.rad_trans_meta import get_data_attributes
from mwrpy_sim.rad_trans.run_rad_trans import rad_trans
from mwrpy_sim.rad_trans.run_rad_trans_day import rad_trans_day
from mwrpy_sim.utils import (
    _get_filename,
    append_data,
    date_range,
    get_file_list,
    get_processing_dates,
    isodate2date,
    read_config,
    seconds2date,
)


def main(args):
    logging.basicConfig(level="INFO")
    logging.info(f"Processing started at {str(datetime.datetime.now()).split('.')[0]}")
    _start_date, _stop_date = get_processing_dates(args)
    start_date = isodate2date(_start_date)
    stop_date = isodate2date(_stop_date)
    global_specs = read_config(args.site, "global_specs")
    params = read_config(args.site, "params")
    file_name = _get_filename(args.source, start_date, stop_date, args.site)
    data_dir = "./mwrpy_sim/rad_trans/openMWR/"
    os.makedirs(data_dir, exist_ok=True)
    if global_specs["calc_ir"]:
        create_site(args.site, data_dir=data_dir, gen=params["gen"])
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
    shutil.rmtree(data_dir, ignore_errors=True)
    logging.info(f"Processing ended at {str(datetime.datetime.now()).split('.')[0]}")


def process_input(
    source: str,
    site: str,
    start_date: datetime.date,
    stop_date: datetime.date,
    params: dict,
) -> dict:
    data_nc: dict = {}
    config = read_config(None, "global_specs")
    if (source == "ifs") or (source == "era5" and config["era5"][:] == "cloudnet"):
        for date in date_range(start_date, stop_date):
            data_in = params["data_cn"] + date.strftime("%Y/") + date.strftime("%Y%m%d")
            key = "ecmwf" if source == "ifs" else "era5"
            file_name = get_file_list(data_in, key)
            if len(file_name) == 1:
                with nc.Dataset(file_name[0]) as cn_data:
                    if len(cn_data["time"]) == 25 and np.all(
                        ~cn_data["temperature"][:-1, 0].mask
                    ):
                        logging.info(
                            f"Radiative transfer using {source} data for {site}, {date}"
                        )
                        date_arr = [
                            datetime.datetime.combine(
                                date, datetime.time(int(hour))
                            ).strftime("%Y%m%d%H")
                            for hour in cn_data["time"][:-1]
                        ]
                        input_cn = prep.prepare_cn(cn_data, date_arr)
                        input_cn = prep.check_height_day(input_cn, params["altitude"])
                        try:
                            output_day = rad_trans_day(input_cn, params)
                        except ValueError:
                            logging.info(f"Skipping day {date_arr[0][:8]}")
                            continue
                        data_nc = append_data(data_nc, output_day)

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
                    input_era5 = prep.prepare_era5_mod(
                        era5_data_sfc, era5_data_pro, index, date_i
                    )
                    try:
                        output_hour = call_rad_trans(input_era5, params)
                    except ValueError:
                        logging.info(f"Skipping time {date_i}")
                        continue
                    data_nc = append_data(data_nc, output_hour)

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
                    input_era5 = prep.prepare_era5_pres(era5_data, index, date_i)
                    try:
                        output_hour = call_rad_trans(input_era5, params)
                    except ValueError:
                        logging.info(f"Skipping time {date_i}")
                        continue
                    data_nc = append_data(data_nc, output_hour)

    elif source == "gruan":
        for date in date_range(start_date, stop_date):
            f_names = get_file_list(
                params["data_rs"] + date.strftime("%Y/"), "RS41-GDP"
            )
            day_files1 = [s for s in f_names if date.strftime("%Y%m%d") in s]
            if len(day_files1) > 0:
                day_files = []
                for hh in ["00", "06", "12", "18"]:
                    tmp = [
                        s for s in day_files1 if date.strftime("%Y%m%d") + "T" + hh in s
                    ]
                    if len(tmp) == 1:
                        day_files.append(tmp[0])
                    elif len(tmp) > 1:
                        day_files.append(tmp[0])
                for file in day_files:
                    if file == day_files[0]:
                        logging.info(
                            f"Radiative transfer using {source} data "
                            f"for {site}, {date.strftime('%Y%m%d')}"
                        )
                    with nc.Dataset(file) as rs_data:
                        input_rs = prep.prepare_gruan(rs_data)
                        if len(input_rs) == 0:
                            logging.info(
                                f"Minimum altitude of 10 km not reached. Skipping file {file}"
                            )
                            continue
                        else:
                            try:
                                output_hour = call_rad_trans(input_rs, params)
                            except Exception as e:
                                logging.info(f"Skipping file {file}: {e}")
                                continue
                            data_nc = append_data(data_nc, output_hour)

    elif source == "vaisala":
        for date in date_range(start_date, stop_date):
            file_name = get_file_list(
                params["data_vs"],
                date.strftime("%Y%m%d"),
                "nc",
            )
            for name in file_name:
                if os.path.isfile(name):
                    with nc.Dataset(name) as vs_data:
                        if "Height" in vs_data.variables:
                            logging.info(
                                f"Radiative transfer using {source} data "
                                f"for {site}, {date.strftime('%Y%m%d')}"
                            )
                            input_vs = prep.prepare_vaisala(vs_data)
                        try:
                            output_hour = call_rad_trans(input_vs, params)
                        except ValueError:
                            logging.info(f"Skipping file {file_name[0]}")
                            continue
                        data_nc = append_data(data_nc, output_hour)

    if source == "standard_atmosphere":
        logging.info(f"Radiative transfer using {source} data")
        input_sa = prep.prepare_standard_atmosphere()
        data_nc = call_rad_trans(input_sa, params)

    data_nc["height"] = np.array(params["height"]) + params["altitude"]
    data_nc["frequency"] = np.array(params["frequency"])
    data_nc["wavelength"] = np.array(params["wavelength"])
    data_nc["elevation_angle"] = np.array(params["elevation_angle"])

    return data_nc


def call_rad_trans(data_in: dict, params: dict) -> dict:
    data_in = prep.check_height(data_in, params["altitude"], 5.0)
    data_nc = rad_trans(
        data_in,
        params,
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
