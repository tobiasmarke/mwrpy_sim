"""SimArray Class."""

from datetime import datetime, timezone

import netCDF4
import numpy as np

from mwrpy_sim import utils, version
from mwrpy_sim.utils import MetaData


class SimArray:
    """Stores netCDF4 variables, numpy arrays and scalars as SimArrays.

    Args:
        variable: The netCDF4 :class:`Variable` instance,
        numpy array (masked or regular), or scalar (float, int).
        name: Name of the variable.
        units_from_user: Units of the variable.

    Attributes:
        name (str): Name of the variable.
        data (ndarray): The actual data.
        data_type (str): 'i4' for integers, 'f4' for floats.
        units (str): The `units_from_user` argument if it is given. Otherwise,
            copied from the original netcdf4 variable. Empty if input is just data.
    """

    def __init__(
        self,
        variable: netCDF4.Variable | np.ndarray | float | int,
        name: str,
        units_from_user: str | None = None,
        dimensions: str | None = None,
    ):
        self.variable = variable
        self.name = name
        self.data = self._init_data()
        self.units = self._init_units(units_from_user)
        self.data_type = self._init_data_type()
        self.dimensions = dimensions

    def fetch_attributes(self) -> list:
        """Returns list of user-defined attributes."""
        attributes = []
        for attr in self.__dict__:
            if attr not in ("name", "data", "variable", "dimensions"):
                attributes.append(attr)
        return attributes

    def set_attributes(self, attributes: MetaData) -> None:
        """Overwrites existing instance attributes."""
        for key in attributes._fields:  # To iterate namedtuple fields.
            data = getattr(attributes, key)
            if data:
                setattr(self, key, data)

    def _init_data(self) -> np.ndarray:
        if isinstance(self.variable, netCDF4.Variable):
            return self.variable[:]
        if isinstance(self.variable, np.ndarray):
            return self.variable
        if isinstance(self.variable, (int, float)):
            return np.array(self.variable)
        if isinstance(self.variable, str):
            try:
                numeric_value = utils.str_to_numeric(self.variable)
                return np.array(numeric_value)
            except ValueError:
                pass
        raise ValueError(f"Incorrect SimArray input: {self.variable}")

    def _init_units(self, units_from_user: str | None) -> str:
        if units_from_user is not None:
            return units_from_user
        return getattr(self.variable, "units", "")

    def _init_data_type(self) -> str:
        if self.data.dtype in (np.float32, np.float64):
            return "f4"
        return "i4"


class Sim:
    """Base class for Sim MWR."""

    def __init__(self, raw_data: dict):
        self.raw_data = raw_data
        self.data = self._init_data()

    def _init_data(self) -> dict:
        data = {}
        for key in self.raw_data:
            data[key] = SimArray(self.raw_data[key], key)
        return data


def save_sim(sim: Sim, output_file: str, att: dict, source: str) -> None:
    """Saves the Sim MWR file."""
    dims = {
        "time": len(sim.data["time"].data),
        "frequency": len(sim.data["frequency"].data),
        "wavelength": len(sim.data["wavelength"].data),
        "height": len(sim.data["height"].data),
        "elevation_angle": len(sim.data["elevation_angle"].data),
    }

    with init_file(output_file, dims, sim.data, att) as rootgrp:
        setattr(rootgrp, "source", source)
        if not "era5" in source:
            rootgrp.delncattr("era5")


def init_file(
    file_name: str, dimensions: dict, sim_arrays: dict, att_global: dict
) -> netCDF4.Dataset:
    """Initializes a Sim MWR file for writing.

    Args:
        file_name: File name to be generated.
        dimensions: Dictionary containing dimension for this file.
        sim_arrays: Dictionary containing :class:`SimArray` instances.
        att_global: Dictionary containing site specific global attributes
    """
    nc_file = netCDF4.Dataset(file_name, "w", format="NETCDF4_CLASSIC")
    for key, dimension in dimensions.items():
        nc_file.createDimension(key, dimension)
    _write_vars2nc(nc_file, sim_arrays)
    _add_standard_global_attributes(nc_file, att_global)
    return nc_file


def _get_dimensions(nc_file: netCDF4.Dataset, data: np.ndarray) -> tuple | tuple[str]:
    """Finds correct dimensions for a variable."""
    if utils.isscalar(data):
        return ()
    file_dims = nc_file.dimensions
    array_dims = data.shape
    dim_names = []
    for length in array_dims:
        dim = [key for key in file_dims.keys() if file_dims[key].size == length][0]
        dim_names.append(dim)
    return tuple(dim_names)


def _write_vars2nc(nc_file: netCDF4.Dataset, mwr_variables: dict) -> None:
    """Iterates over instances and writes to netCDF file."""
    for obj in mwr_variables.values():
        if obj.data_type == "f4":
            fill_value = -999.0
        else:
            fill_value = -99

        size = obj.dimensions or _get_dimensions(nc_file, obj.data)
        if obj.name == "time":
            size = "time"
        if obj.name == "elevation_angle":
            size = "elevation_angle"
        if obj.name in (
            "iwv",
            "lwp",
            "lwp_pro",
            "cbh",
            "cbh_pro",
            "k_index",
            "ko_index",
            "total_totals_index",
            "lifted_index",
            "showalter_index",
            "cape",
        ):
            size = "time"
        if obj.name in ("tb", "tb_pro", "tb_clr"):
            size = ("time", "frequency", "elevation_angle")
        if obj.name in ("irt", "irt_pro", "irt_clr"):
            size = ("time", "wavelength")
        if obj.name in (
            "air_temperature",
            "absolute_humidity",
            "air_pressure",
            "relative_humidity",
            "lwc",
            "lwc_pro",
        ):
            size = ("time", "height")
        nc_variable = nc_file.createVariable(
            obj.name, obj.data_type, size, zlib=True, fill_value=fill_value
        )
        nc_variable[:] = obj.data
        for attr in obj.fetch_attributes():
            setattr(nc_variable, attr, getattr(obj, attr))


def _add_standard_global_attributes(nc_file: netCDF4.Dataset, att_global) -> None:
    nc_file.mwrpy_sim_version = version.__version__
    nc_file.processed = (
        datetime.now(tz=timezone.utc).strftime("%d %b %Y %H:%M:%S") + " UTC"
    )
    for name, value in att_global.items():
        if value is None:
            value = ""
        if isinstance(value, (int, float, bool)):
            value = str(value)
        setattr(nc_file, name, value)
