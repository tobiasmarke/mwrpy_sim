import metpy.calc as mpcalc
import numpy as np
from metpy.units import units

from mwrpy_sim.atmos import eq_pot_tem, mixr, t_dew_rh


def modify_prof_500m(
    z: np.ndarray, t: np.ndarray, p: np.ndarray, td: np.ndarray, mr: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Modify profiles below 500 m with average value.

    Input:
        z: Height array (m)
        t: Temperature array (K)
        p: Pressure array (Pa)
        td: Dew point temperature array (K)
        mr: Mixing ratio array (kg/kg)

    Output:
        Modified temperature profile
    """
    ind_500 = np.where(z <= 500)[0]
    if len(ind_500) > 0:
        t[:, ind_500] = np.mean(t[:, ind_500])
        p[:, ind_500] = np.mean(p[:, ind_500])
        td[:, ind_500] = np.mean(td[:, ind_500])
        mr[:, ind_500] = np.mean(mr[:, ind_500])
    return (
        z[ind_500[-1] :],
        t[:, ind_500[-1] :],
        p[:, ind_500[-1] :],
        td[:, ind_500[-1] :],
        mr[:, ind_500[-1] :],
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


def calc_stability_indices(data_dict: dict, height: np.ndarray) -> None:
    """Calculate stability indices from temperature, pressure, height and relative humidity.

    Input:
        data_dict: Dictionary containing atmospheric data.

    Output:
        None, but modifies the input dictionary to include stability indices.
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
    z_mod, t_mod, p_mod, td_mod, mr_mod = modify_prof_500m(
        np.array(height),
        data_dict["air_temperature"][:, :],
        data_dict["air_pressure"][:, :],
        t_dew[:, :],
        mix_rat[:, :],
    )
    mixed_prof = mpcalc.parcel_profile(
        p_mod[0, :] * units.Pa, t_mod[0, 0] * units.K, units.Quantity(td_mod[0, 0], "K")
    )

    # Calculate k index
    data_dict["k_index"] = np.expand_dims(
        mpcalc.k_index(
            data_dict["air_pressure"][0, :] * units.Pa,
            data_dict["air_temperature"][0, :] * units.K,
            units.Quantity(t_dew[0, :], "K"),
        ).magnitude,
        0,
    )

    # Calculate ko index
    p_ind_1000 = (
        0
        if p_ind(1000, data_dict["air_pressure"]) is None
        else p_ind(1000, data_dict["air_pressure"])
    )
    data_dict["ko_index"] = np.expand_dims(
        ko_index(
            eq_pot_t[:, p_ind(700, data_dict["air_pressure"])],
            eq_pot_t[:, p_ind(500, data_dict["air_pressure"])],
            eq_pot_t[:, p_ind_1000],
            eq_pot_t[:, p_ind(850, data_dict["air_pressure"])],
        ),
        0,
    )

    # Calculate total totals index
    data_dict["total_totals_index"] = np.expand_dims(
        mpcalc.total_totals_index(
            data_dict["air_pressure"][0, :] * units.Pa,
            data_dict["air_temperature"][0, :] * units.K,
            units.Quantity(t_dew[0, :], "K"),
        ).magnitude,
        0,
    )

    # Calculate lifted index
    data_dict["lifted_index"] = np.expand_dims(
        mpcalc.lifted_index(
            p_mod[0, :] * units.Pa,
            t_mod[0, :] * units.K,
            mixed_prof,
        ).magnitude,
        0,
    )

    # Calculate showalter index
    data_dict["showalter_index"] = np.expand_dims(
        mpcalc.showalter_index(
            data_dict["air_pressure"][0, :] * units.Pa,
            data_dict["air_temperature"][0, :] * units.K,
            units.Quantity(t_dew[0, :], "K"),
        ).magnitude,
        0,
    )

    # Calculate convective available potential energy (CAPE)
    data_dict["cape"] = np.expand_dims(
        mpcalc.cape_cin(
            p_mod[0, :] * units.Pa,
            t_mod[0, :] * units.K,
            units.Quantity(td_mod[0, :], "K"),
            mixed_prof,
        )[0].magnitude,
        0,
    )
