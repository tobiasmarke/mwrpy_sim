# MWRpy_sim

Repository for Simulating Microwave Radiometer and Infrared Brightness Temperatures

Note: The code and data files for IR radiative transfer calculations are not included in this repository.

## Installation

From GitHub:

```shell
git clone https://github.com/tobiasmarke/mwrpy_sim.git
cd mwrpy_sim
python3 -m venv venv
source venv/bin/activate
pip3 install --upgrade pip
pip3 install .
```

MWRpy_sim requires Python 3.10 or newer.

## Configuration

The folder `mwrpy_sim/site_config/` contains site-specific configuration files,
defining the input and output data paths etc., and the file `config.yaml`, which
defines the elevation angles, frequencies/wavelengths and height grid. In addition,
the following global specifications can be set:

| global_specs   | Options                     | Description                                  |
| :------------- | :-------------------------- | :------------------------------------------- |
| `mw_model`     | `R22`, `R24`                | Model for absorption calculations.           |
| `corrections`  | `bandwidth`, `beamwidth`    | Corrections for MWR characteristics.         |
| `refractivity` | `Rueeger2002`, `Thayer1974` | Model for refractivity corrections.          |
| `clouds`       | `True`, `False`             | Include (True) or skip (False) cloudy cases. |
| `calc_ir`      | `True`, `False`             | Calculate IR brightness temperatures.        |
| `era5`         | `pressure`, `model`         | ERA5 data source.                            |

## Command line usage

MWRpy_sim can be run using the command line tool `mwrpy_sim/cli.py`:

    usage: mwrpy_sim/cli.py [-h] -s SITE [-d YYYY-MM-DD] [--start YYYY-MM-DD]
                           [--stop YYYY-MM-DD] [{radiosonde}]

Arguments:

| Short | Long       | Default           | Description                                                                        |
| :---- | :--------- | :---------------- | :--------------------------------------------------------------------------------- |
| `-s`  | `--site`   |                   | Site to process data from, e.g, `lindenberg`. Required.                            |
|       | `--source` |                   | Data source for radiative transfer calculations, e.g. `ifs`. Required.             |
| `-d`  | `--date`   |                   | Single date to be processed. Alternatively, `--start` and `--stop` can be defined. |
|       | `--start`  | `current day - 1` | Starting date.                                                                     |
|       | `--stop`   | `current day `    | Stopping date.                                                                     |

Supported data sources are:

| Input Type            | Description                    |
| :-------------------- | :----------------------------- |
| `ifs`                 | IFS data from ECMWF.           |
| `radiosonde`          | Radiosonde data from DWD.      |
| `vaisala`             | Vaisala radiosonde data.       |
| `era5`                | ERA5 data from ECMWF.          |
| `get_era5`            | Download ERA5 data from ECMWF. |
| `standard_atmosphere` | US Standard Atmosphere 1976.   |

Commands:

| Command    | Description                      |
| :--------- | :------------------------------- |
| `process`  | Process and plot data (default). |
| `plot`     | Plot input data statistics.      |
| `no-plot`  | Process data only (no plots).    |
| `get_era5` | Download ERA5 data from ECMWF.   |
