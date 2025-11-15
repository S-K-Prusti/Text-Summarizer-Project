#!/usr/bin/env python
# coding: utf-8

# In[37]:


import os


# In[41]:


from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir:Path
    source_URL:str
    local_data_files:Path
    unzip_dir:Path


# In[42]:


from textSummarizer.constants import *
from textSummarizer.utils.common import read_yaml, create_directories


# In[43]:


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


# In[44]:


import os
import urllib.request as request
import zipfile
from textSummarizer.logging import logger
from textSummarizer.utils.common import get_size


# In[45]:


class DataIngestion:
    def __init__(self,config: DataIngestionConfig):
       self.config = config
    
    

    def download_file(self):
        if not os.path.exists(self.config.local_data_files):
            filename, headers = request.urlretrieve(
                url= self.config.source_URL,
                filename = self.config.local_data_files
            )
            logger.info(f"{filename} download! with following info: \n{headers}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_files))}")



    def extract_zip_file(self):
        '''
        zip_file_path:str
        extracts the zip file into the data directory
        function returns none
        '''

        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_files, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
        


# In[51]:


try:
    config= ConfigurationManager()
    data_ingestion_config = config.get_data_ingestion_config()
    data_ingestion = DataIngestion(config= data_ingestion_config)
    data_ingestion.download_file()
    data_ingestion.extract_zip_file()
except Exception as e:
    raise e


# In[ ]:




