from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import TrainingArguments, Trainer
import torch
import os
from datasets import load_from_disk
from src.textSummerizer.entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self,config:ModelTrainerConfig):
        self.config= config
        
    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model_pegasus =AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)
        
        #loading the data
        datasest_samsum_pt =load_from_disk(self.config.data_path)
        
        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir,
            num_train_epochs=10,             # Number of times to iterate over the entire dataset
            warmup_steps=500,               # Steps for learning rate warmup before training starts
            per_device_train_batch_size=1,  # Batch size during training (per device/GPU)
            per_device_eval_batch_size=1,   # Batch size during evaluation (per device/GPU)
            weight_decay=0.01,              # L2 regularization to prevent overfitting
            logging_steps=10,               # Log metrics every 10 steps
            evaluation_strategy='steps',    # Evaluation occurs at specific steps (not after each epoch)
            eval_steps=500,                 # Run evaluation every 500 steps
            save_steps=1e6,                 # Save model checkpoint after 1 million steps (effectively never during training)
            gradient_accumulation_steps=16  # Accumulate gradients over 16 steps before updating weights
        )
        dataset_samsum_pt =load_from_disk(self.config.data_path)
        
        trainer = Trainer(model = model_pegasus,
                  args = trainer_args,
                  tokenizer =tokenizer,data_collator=seq2seq_data_collator,
                  train_dataset =dataset_samsum_pt['test'],
                  eval_dataset = dataset_samsum_pt['validation'])

        trainer.train()
        
        #Saving the model
        model_pegasus.save_pretrained(os.path.join(self.config.root_dir,"pegasus-samsum-model"))
        
        #saving the tokenizer
        tokenizer.save_pretrained(os.path.join(self.config.root_dir,"tokenizer"))
