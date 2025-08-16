from textSummarizer.constants import *
from textSummarizer.utils.common import read_yaml, create_directories
from textSummarizer.entity import(DataIngestionConfig)




class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config["artifacts_root"]])  # fixed dict access

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config["data_ingestion"]  # fixed dict access

        create_directories([config["root_dir"]])  # fixed dict access

        data_ingestion_config = DataIngestionConfig(
            root_dir=config["root_dir"],
            source_URL=config["source_URL"],
            local_data_files=config["local_data_files"],
            unzip_dir=config["unzip_dir"]
        )

        return data_ingestion_config
