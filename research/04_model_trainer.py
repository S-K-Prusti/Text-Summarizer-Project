#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os


# In[5]:


from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    data_path: Path
    model_ckpt: Path
    num_train_epochs: int
    warmup_steps: int
    per_device_train_batch_size: int
    weight_decay: float
    logging_steps: 100
    eval_strategy: str
    eval_steps: int
    save_steps: float
    gradient_accumulation_steps: int


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

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.TrainingArguments

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            model_ckpt=config.model_ckpt,
            num_train_epochs= params.num_train_epochs,
            warmup_steps= params.warmup_steps,
            per_device_train_batch_size= params.per_device_train_batch_size,
            weight_decay= params.weight_decay,
            logging_steps= params.logging_steps,
            eval_strategy= params.eval_strategy,
            eval_steps= params.eval_steps,
            save_steps= params.save_steps,
            gradient_accumulation_steps= params.gradient_accumulation_steps
        )

        return model_trainer_config


# In[8]:


from transformers import TrainingArguments, Trainer 
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
import torch


# In[ ]:


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config


    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


        # loading the data
        dataset_samsum_pt = load_from_disk(self.config.data_path)

        # trainer_args = TrainingArguments(
        #     output_dir=self.config.root_dir, num_train_epochs=self.config.num_train_epochs, warmup_steps=self.config.warmup_steps,
        #     per_device_train_batch_size=self.config.per_device_train_batch_size, per_device_eval_batch_size=self.config.per_device_train_batch_size,
        #     weight_decay=self.config.weight_decay, logging_steps=self.config.logging_steps, 
        #     evaluation_strategy=self.config.evaluation_startegy, eval_steps=self.config.eval_steps, save_steps=1e6,
        #     gradient_accumulation_steps=self.config.gradient_accumulation_steps
        # )

        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir, num_train_epochs=3, warmup_steps=500,
            per_device_train_batch_size=1, per_device_eval_batch_size=1,
            weight_decay=0.01, logging_steps=1000, save_total_limit=1,
            eval_strategy='no', eval_steps=500, save_steps=2500,
            gradient_accumulation_steps=8,fp16=False
        )


        trainer = Trainer(model=model, args=trainer_args,
                  tokenizer=tokenizer, data_collator=seq2seq_data_collator,
                  train_dataset=dataset_samsum_pt["train"],
                  eval_dataset=dataset_samsum_pt["validation"])
        
        trainer.train()

        # save model
        model.save_pretrained(os.path.join(self.config.root_dir,"pegasus-samsum-model"))
        # save tokenizer
        tokenizer.save_pretrained(os.path.join(self.config.root_dir,"tokenizer"))


# In[21]:


import transformers
print(transformers.__version__)


# In[11]:


try:
    config= ConfigurationManager()
    model_trainer_config = config.get_model_trainer_config()
    model_trainer_config = ModelTrainer(config=model_trainer_config)
    model_trainer_config.train()
except Exception as e:
    raise e


# In[ ]:




