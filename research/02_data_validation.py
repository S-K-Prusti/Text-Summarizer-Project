#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os


# In[5]:


from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    ALL_REQUIRED_FILES: list


# In[6]:


from textSummarizer.constants import *
from textSummarizer.utils.common import read_yaml, create_directories


# In[7]:


class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config["artifacts_root"]]) 


    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            ALL_REQUIRED_FILES=config.ALL_REQUIRED_FILES,
        )

        return data_validation_config


# In[8]:


import os
from textSummarizer.logging import logger


# In[ ]:


class DataValidation:
    def __init__(self,config: DataValidationConfig):
       self.config = config


    def validate_all_files_exist(self) -> bool:
        try:
            validation_status=None

            all_files = os.listdir(os.path.join("artifacts","data_ingestion","samsum_dataset"))

            for file in all_files :
                if file not in self.config.ALL_REQUIRED_FILES:
                    validation_status=False
                    with open(self.config.STATUS_FILE,'w') as f:
                        f.write(f"Validation Status: {validation_status}")
                else:
                    validation_status=True
                    with open(self.config.STATUS_FILE,'w') as f:
                        f.write(f"Validation Status: {validation_status}")

            return validation_status
        
        except Exception as e:
            raise e


# In[15]:


try:
    config= ConfigurationManager()
    data_validation_config = config.get_data_validation_config()
    data_validation = DataValidation(config= data_validation_config)
    data_validation.validate_all_files_exist()
except Exception as e:
    raise e


# In[ ]:




