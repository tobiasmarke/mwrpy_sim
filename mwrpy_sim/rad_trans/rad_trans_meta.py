"""Module for Radiative Transfer Metadata."""

from mwrpy_sim.utils import MetaData


def get_data_attributes(sim_variables: dict, source: str) -> dict:
    """Adds Metadata for Sim MWR variables for NetCDF file writing.

    Args:
        sim_variables: SimArray instances.
        source: Data type of the netCDF file.

    Returns:
        Dictionary

    Raises:
        RuntimeError: Specified data type is not supported.

    Example:
        from rad_trans.rad_trans_meta import get_data_attributes
        att = get_data_attributes('data','data_type')
    """
    if source not in (
        "ifs",
        "radiosonde",
        "vaisala",
        "era5",
        "icon",
        "standard_atmosphere",
    ):
        raise RuntimeError([source + " not supported for file writing."])

    for key in ATTRIBUTES_SOURCE:
        ATTRIBUTES_SOURCE[key] = ATTRIBUTES_SOURCE[key]._replace(
            long_name=ATTRIBUTES_SOURCE[key].long_name + source
        )
    attributes = dict(ATTRIBUTES_COM, **ATTRIBUTES_SOURCE)

    for key in list(sim_variables):
        if key in attributes:
            sim_variables[key].set_attributes(attributes[key])
        else:
            del sim_variables[key]

    index_map = {v: i for i, v in enumerate(attributes)}
    ret_variables = dict(
        sorted(sim_variables.items(), key=lambda pair: index_map[pair[0]])
    )

    return ret_variables


ATTRIBUTES_COM = {
    "time": MetaData(
        long_name="Time (UTC) of the measurement",
        units="seconds since 1970-01-01 00:00:00.000",
        dimensions=("time",),
    ),
    "height": MetaData(
        long_name="Height above mean sea level",
        standard_name="height_above_mean_sea_level",
        units="m",
        dimensions=("height",),
    ),
    "frequency": MetaData(
        long_name="Nominal centre frequency of microwave channels",
        standard_name="radiation_frequency",
        units="GHz",
        dimensions=("frequency",),
    ),
    "wavelength": MetaData(
        long_name="Nominal centre wavelength of infrared channels",
        standard_name="radiation_wavelength",
        units="µm",
        dimensions=("wavelength",),
    ),
    "elevation_angle": MetaData(
        long_name="Sensor elevation angle",
        units="degree",
        comment="0=horizon, 90=zenith",
        dimensions=("elevation_angle",),
    ),
}


ATTRIBUTES_SOURCE = {
    "tb": MetaData(
        long_name="Microwave brightness temperature simulated from ",
        standard_name="brightness_temperature",
        units="K",
        dimensions=("time", "frequency", "elevation_angle"),
    ),
    "tb_pro": MetaData(
        long_name="Microwave brightness temperature (prognostic) simulated from ",
        standard_name="brightness_temperature",
        units="K",
        dimensions=("time", "frequency", "elevation_angle"),
    ),
    "tb_clr": MetaData(
        long_name="Microwave brightness temperature (no liquid water) simulated from ",
        standard_name="brightness_temperature",
        units="K",
        dimensions=("time", "frequency", "elevation_angle"),
    ),
    "irt": MetaData(
        long_name="Infrared brightness temperature simulated from ",
        standard_name="brightness_temperature",
        units="K",
        dimensions=("time", "wavelength", "elevation_angle"),
    ),
    "irt_pro": MetaData(
        long_name="Infrared brightness temperature (prognostic) simulated from ",
        standard_name="brightness_temperature",
        units="K",
        dimensions=("time", "wavelength", "elevation_angle"),
    ),
    "irt_clr": MetaData(
        long_name="Infrared brightness temperature (no liquid water) simulated from ",
        standard_name="brightness_temperature",
        units="K",
        dimensions=("time", "wavelength", "elevation_angle"),
    ),
    "air_temperature": MetaData(
        long_name="Temperature profile interpolated from ",
        standard_name="air_temperature",
        units="K",
        dimensions=("time", "height"),
    ),
    "air_pressure": MetaData(
        long_name="Pressure profile interpolated from ",
        standard_name="air_pressure",
        units="Pa",
        dimensions=("time", "height"),
    ),
    "absolute_humidity": MetaData(
        long_name="Absolute humidity profile interpolated from ",
        units="kg m-3",
        dimensions=("time", "height"),
    ),
    "relative_humidity": MetaData(
        long_name="Relative humidity profile interpolated from ",
        standard_name="relative_humidity",
        units="1",
        dimensions=("time", "height"),
    ),
    "lwc": MetaData(
        long_name="Liquid water content profile interpolated from ",
        standard_name="mass_concentration_of_liquid_water_in_air",
        units="kg m-3",
        dimensions=("time", "height"),
    ),
    "lwc_pro": MetaData(
        long_name="Liquid water content profile (prognostic) interpolated from ",
        standard_name="mass_concentration_of_liquid_water_in_air",
        units="kg m-3",
        dimensions=("time", "height"),
    ),
    "lwp": MetaData(
        long_name="Column-integrated liquid water path derived from ",
        standard_name="atmosphere_cloud_liquid_water_content",
        units="kg m-2",
        dimensions=("time",),
    ),
    "lwp_pro": MetaData(
        long_name="Column-integrated liquid water path (prognostic) derived from ",
        standard_name="atmosphere_cloud_liquid_water_content",
        units="kg m-2",
        dimensions=("time",),
    ),
    "iwv": MetaData(
        long_name="Column-integrated water vapour derived from ",
        standard_name="atmosphere_mass_content_of_water_vapor",
        units="kg m-2",
        dimensions=("time",),
    ),
    "cbh": MetaData(
        long_name="Cloud base heights above mean sea level derived from ",
        standard_name="cloud_base_height_above_mean_sea_level",
        units="m",
        dimensions=("time",),
    ),
    "cbh_pro": MetaData(
        long_name="Cloud base heights above mean sea level (prognostic) derived from ",
        standard_name="cloud_base_height_above_mean_sea_level",
        units="m",
        dimensions=("time",),
    ),
    "cth": MetaData(
        long_name="Cloud top heights above mean sea level derived from ",
        standard_name="cloud_top_height_above_mean_sea_level",
        units="m",
        dimensions=("time",),
    ),
    "cth_pro": MetaData(
        long_name="Cloud top heights above mean sea level (prognostic) derived from ",
        standard_name="cloud_top_height_above_mean_sea_level",
        units="m",
        dimensions=("time",),
    ),
    "k_index": MetaData(
        long_name="K-index derived from ",
        standard_name="k_index",
        units="K",
        comment="K-index is a measure of atmospheric instability",
        dimensions=("time",),
    ),
    "ko_index": MetaData(
        long_name="KO-index derived from ",
        standard_name="ko_index",
        units="K",
        comment="KO-index is a measure of atmospheric stability",
        dimensions=("time",),
    ),
    "total_totals_index": MetaData(
        long_name="Total Totals Index derived from ",
        standard_name="total_totals_index",
        units="K",
        comment="Total Totals Index is a measure of atmospheric instability",
        dimensions=("time",),
    ),
    "lifted_index": MetaData(
        long_name="Lifted Index derived from ",
        standard_name="lifted_index",
        units="K",
        comment="Lifted Index is a measure of atmospheric stability",
        dimensions=("time",),
    ),
    "showalter_index": MetaData(
        long_name="Showalter Index derived from ",
        standard_name="showalter_index",
        units="K",
        comment="Showalter Index is a measure of atmospheric stability",
        dimensions=("time",),
    ),
    "cape": MetaData(
        long_name="Convective available potential energy derived from ",
        standard_name="convective_available_potential_energy",
        units="J kg-1",
        comment="CAPE is a measure of the amount of energy available for convection",
        dimensions=("time",),
    ),
}
