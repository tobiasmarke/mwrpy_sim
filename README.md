# MWRpy_sim

Repository for Simulating Microwave Radiometer Brightness Temperatures

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
defines the elevation angles, frequencies and height grid.

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
