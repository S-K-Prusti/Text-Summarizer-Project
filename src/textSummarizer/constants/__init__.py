#to read the config.yaml and params.yaml we r using constants as these paths are not going to change so we r keeping it here as constants

from pathlib import Path

CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")