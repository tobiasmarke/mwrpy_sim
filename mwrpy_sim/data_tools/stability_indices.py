import metpy.calc as mpcalc
import numpy as np
from metpy.units import units

from mwrpy_sim.atmos import eq_pot_tem, mixr, t_dew_rh


def modify_prof_500m(
    height: np.ndarray,
    temperature: np.ndarray,
    pressure: np.ndarray,
    dewpoint: np.ndarray,
    mixing_ratio: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Modify profiles below 500 m with average value.

    Input:
        height: Height array (m)
        temperature: Temperature array (K)
        pressure: Pressure array (Pa)
        dewpoint: Dewpoint temperature array (K)
        mixing_ratio: Mixing ratio array (kg/kg)

    Output:
        Modified profiles.
    """
    ind_500 = np.where(height <= 500)[0]
    t = temperature.copy()
    p = pressure.copy()
    td = dewpoint.copy()
    if len(ind_500) > 0:
        t[:, ind_500] = np.mean(t[:, ind_500])
        p[:, ind_500] = np.mean(p[:, ind_500])
        td[:, ind_500] = np.mean(td[:, ind_500])
    else:
        ind_500 = np.array([0])
    return (
        t[:, ind_500[-1] :],
        p[:, ind_500[-1] :],
        td[:, ind_500[-1] :],
    )


def p_ind(lev: int, p: np.ndarray):
    """Get pressure level index.

    Input:
        lev: Pressure level (hPa)
        p: Pressure array (Pa)

    Output:
        Pressure level index
    """
    diff = np.abs(p - lev * 100)
    ind = np.argmin(diff) if np.min(diff) < 2000.0 else None
    return ind


def ko_index(
    Teq700: float | np.ndarray,
    Teq500: float | np.ndarray,
    Teq1000: float | np.ndarray,
    Teq850: float | np.ndarray,
) -> float | np.ndarray:
    """Calculate KO-index.

    Input:
        T700: Temperature at 700 hPa (K)
        T500: Temperature at 500 hPa (K)
        T1000: Temperature at 1000 hPa (K)
        T850: Temperature at 850 hPa (K)
    Output:
        KO-index value
    """
    return 0.5 * (Teq700 - Teq500) - 0.5 * (Teq1000 - Teq850)


def calc_stability_indices(data_dict: dict, height: np.ndarray) -> dict:
    """Calculate stability indices from temperature, pressure, height and relative humidity.

    Input:
        data_dict: Dictionary containing atmospheric data.

    Output:
        output_dict: Dictionary including stability indices.
    """
    # Calculate additional variables
    mix_rat = mixr(
        data_dict["air_temperature"][:, :],
        data_dict["absolute_humidity"][:, :],
        data_dict["air_pressure"][:, 0],
        height,
    )
    eq_pot_t = eq_pot_tem(
        data_dict["air_temperature"][:, :],
        mix_rat,
        data_dict["air_pressure"][:, 0],
        height,
    )
    t_dew = t_dew_rh(
        data_dict["air_temperature"][:, :], data_dict["relative_humidity"][:, :]
    )
    # Modify profiles below 500 m
    t_mod, p_mod, td_mod = modify_prof_500m(
        np.array(height),
        data_dict["air_temperature"][:, :],
        data_dict["air_pressure"][:, :],
        t_dew[:, :],
        mix_rat[:],
    )

    # Calculate stability indices
    indices = [
        "k_index",
        "ko_index",
        "total_totals_index",
        "lifted_index",
        "showalter_index",
        "cape",
    ]
    output_dict = {}
    for index in indices:
        output_dict[index] = np.ma.masked_all(
            len(
                data_dict["time"],
            ),
            np.float32,
        )
    for ind in range(len(data_dict["time"])):
        mixed_prof = mpcalc.parcel_profile(
            units.Quantity(p_mod[ind, :], "Pa"),
            units.Quantity(t_mod[ind, 0], "K"),
            units.Quantity(td_mod[ind, 0], "K"),
        )

        # Calculate k index
        output_dict["k_index"][ind] = np.ma.masked_invalid(
            float(
                mpcalc.k_index(
                    units.Quantity(data_dict["air_pressure"][ind, :], "Pa"),
                    units.Quantity(data_dict["air_temperature"][ind, :], "K"),
                    units.Quantity(t_dew[ind, :], "K"),
                ).magnitude
            )
        )

        # Calculate ko index
        p_levs = [700, 500, 1000, 850]
        if np.all([p_ind(p, data_dict["air_pressure"][ind, :]) for p in p_levs]):
            output_dict["ko_index"][ind] = float(
                ko_index(
                    eq_pot_t[ind, p_ind(700, data_dict["air_pressure"][ind, :])],
                    eq_pot_t[ind, p_ind(500, data_dict["air_pressure"][ind, :])],
                    eq_pot_t[ind, p_ind(1000, data_dict["air_pressure"][ind, :])],
                    eq_pot_t[ind, p_ind(850, data_dict["air_pressure"][ind, :])],
                ),
            )

        # Calculate total totals index
        output_dict["total_totals_index"][ind] = float(
            mpcalc.total_totals_index(
                units.Quantity(data_dict["air_pressure"][ind, :], "Pa"),
                units.Quantity(data_dict["air_temperature"][ind, :], "K"),
                units.Quantity(t_dew[ind, :], "K"),
            ).magnitude
        )

        # Calculate lifted index
        output_dict["lifted_index"][ind] = float(
            mpcalc.lifted_index(
                units.Quantity(p_mod[ind, :], "Pa"),
                units.Quantity(t_mod[ind, :], "K"),
                mixed_prof,
            ).magnitude
        )

        # Calculate showalter index
        try:
            output_dict["showalter_index"][ind] = float(
                mpcalc.showalter_index(
                    units.Quantity(data_dict["air_pressure"][ind, :], "Pa"),
                    units.Quantity(data_dict["air_temperature"][ind, :], "K"),
                    units.Quantity(t_dew[ind, :], "K"),
                ).magnitude
            )
        except TypeError:
            pass

        # Calculate convective available potential energy (CAPE)
        output_dict["cape"][ind] = float(
            mpcalc.cape_cin(
                units.Quantity(p_mod[ind, :], "Pa"),
                units.Quantity(t_mod[ind, :], "K"),
                units.Quantity(td_mod[ind, :], "K"),
                mixed_prof,
            )[0].magnitude
        )

    return output_dict
