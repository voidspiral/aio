# License: 3-clause BSD
# Copyright (c) 2016-2018, Ml4AAD Group (http://www.ml4aad.org/)

from typing import List

import numpy as np

from oprael.utils.config_space import Configuration, ConfigurationSpace


def convert_configurations_to_array(configs: List[Configuration]) -> np.ndarray:
    """Impute inactive hyperparameters in configurations with their default.

    Necessary to apply an EPM to the data.

    Parameters
    ----------
    configs : List[Configuration]
        List of configuration objects.

    Returns
    -------
    np.ndarray
        Array with configuration hyperparameters. Inactive values are imputed
        with their default value.
    """
    configs_array = np.array([config.get_array() for config in configs],
                             dtype=np.float64)
    #print("lzy configs")
    #print("________________________________________")
    #print(configs)
    #print("________________________________________")
    #print(configs[0])
    #print("________________________________________")
    #print(dir(configs[0]))
    # lzy config
    configuration_space = configs[0].config_space
    return impute_default_values(configuration_space, configs_array)


def impute_default_values(
        configuration_space: ConfigurationSpace,
        configs_array: np.ndarray
) -> np.ndarray:
    """Impute inactive hyperparameters in configuration array with their default.

    Necessary to apply an EPM to the data.

    Parameters
    ----------
    configuration_space : ConfigurationSpace
    
    configs_array : np.ndarray
        Array of configurations.

    Returns
    -------
    np.ndarray
        Array with configuration hyperparameters. Inactive values are imputed
        with their default value.
    """
    for hp in configuration_space.get_hyperparameters():
        #print("________________________________________")
        #print(dir(hp))
        # lzy
        default = hp._normalized_default_value
        idx = configuration_space.get_idx_by_hyperparameter_name(hp.name)
        nonfinite_mask = ~np.isfinite(configs_array[:, idx])
        configs_array[nonfinite_mask, idx] = default

    return configs_array
