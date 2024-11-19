# Helper Functions

import hydra
from loguru import logger
from omegaconf import DictConfig


def instantiate_callbacks(callbacks_cfg: DictConfig):
    """Instantiates callbacks from config.

    Params:
        callbacks_cfg: A DictConfig object containing callback configurations.
        
    Returns:
        A list of instantiated callbacks.
    """
    callbacks = []

    if not callbacks_cfg:
        logger.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            logger.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig):
    """Instantiates loggers from config.

    Params:
        logger_cfg: A DictConfig object containing logger configurations.
    Returns:
        A list of instantiated loggers.
    """
    loggers = []

    if not logger_cfg:
        logger.warning("No logger configs found! Skipping...")
        return loggers

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            logger.info(f"Instantiating logger <{lg_conf._target_}>")
            loggers.append(hydra.utils.instantiate(lg_conf))

    return loggers