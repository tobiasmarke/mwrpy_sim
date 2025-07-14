import datetime

import cdsapi
import numpy as np

from mwrpy_sim.utils import _get_filename, read_config


def era5_request(
    site: str, params: dict, start_date: datetime.date, stop_date: datetime.date
):
    """Function to download ERA5 data from CDS API for specified site and dates.

    Args:
        site: Name of site
        params: config dictionary
        start_date: first day of request
        stop_date: last day of request
    """
    lat_box = get_corner_coord(
        params["latitude"], params["lat_offset"], params["lat_res"]
    )
    lon_box = get_corner_coord(
        params["longitude"], params["lon_offset"], params["lon_res"]
    )
    area_str = f"{lat_box[1]:.3f}/{lon_box[0]:.3f}/{lat_box[0]:.3f}/{lon_box[1]:.3f}"
    lat_res, lon_res = params["lat_res"], params["lon_res"]
    grid_str = f"{lat_res:.3f}/{lon_res:.3f}"
    c = cdsapi.Client()

    config = read_config(None, "global_specs")
    if config["era5"][:] == "model":
        output_file_sfc = _get_filename("era5_input_sfc", start_date, stop_date, site)
        output_file_pro = _get_filename("era5_input_pro", start_date, stop_date, site)

        c.retrieve(
            "reanalysis-era5-complete",
            {
                "class": "ea",
                "dataset": "era5",
                "date": str(start_date) + "/to/" + str(stop_date)
                if start_date != stop_date
                else str(start_date),
                "expver": "1",
                "levelist": "1",
                "levtype": "ml",
                "param": "129/152",
                "stream": "oper",
                "time": "00/to/23/by/1",
                "type": "an",
                "grid": grid_str,
                "area": area_str,
                "format": "netcdf",
            },
            output_file_sfc,
        )
        c.retrieve(
            "reanalysis-era5-complete",
            {
                "class": "ea",
                "dataset": "era5",
                "date": str(start_date) + "/to/" + str(stop_date)
                if start_date != stop_date
                else str(start_date),
                "expver": "1",
                "levelist": "1/to/137",
                "levtype": "ml",
                "param": "130/133/246/248",
                "stream": "oper",
                "time": "00/to/23/by/1",
                "type": "an",
                "grid": grid_str,
                "area": area_str,
                "format": "netcdf",
            },
            output_file_pro,
        )
    else:
        output_file_pres = _get_filename("era5_input_pres", start_date, stop_date, site)
        dataset = "reanalysis-era5-pressure-levels"
        request = {
            "product_type": ["reanalysis"],
            "variable": [
                "geopotential",
                "fraction_of_cloud_cover",
                "relative_humidity",
                "specific_cloud_liquid_water_content",
                "specific_humidity",
                "temperature",
            ],
            "date": str(start_date) + "/to/" + str(stop_date)
            if start_date != stop_date
            else str(start_date),
            "time": "00/to/23/by/1",
            "pressure_level": [
                "1",
                "2",
                "3",
                "5",
                "7",
                "10",
                "20",
                "30",
                "50",
                "70",
                "100",
                "125",
                "150",
                "175",
                "200",
                "225",
                "250",
                "300",
                "350",
                "400",
                "450",
                "500",
                "550",
                "600",
                "650",
                "700",
                "750",
                "775",
                "800",
                "825",
                "850",
                "875",
                "900",
                "925",
                "950",
                "975",
                "1000",
            ],
            "grid": grid_str,
            "area": area_str,
            "data_format": "netcdf",
            "download_format": "unarchived",
        }

        client = cdsapi.Client()
        client.retrieve(dataset, request, output_file_pres)


def get_corner_coord(stn_coord, offset, resol):
    """Get corners of a coordinate box around station coordinates
    which match model grid points.
    """
    stn_coord_rounded = (
        round(stn_coord / resol) * resol
    )  # round centre coordinate to model resolution
    return stn_coord_rounded + np.array(offset)
