from src.textSummerizer.constants import *
from src.textSummerizer.utils.common import read_yaml, create_directories
from src.textSummerizer.entity import DataIngestionConfig
from src.textSummerizer.entity import DataTransformationConfig,ModelTrainerConfig
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer



class ConfigurationManager:
    def __init__(self,config_path=CONFIG_FILE_PATH,
             params_filepath = PARAMS_FILE_PATH):
        self.config = read_yaml(config_path)
        self.params =read_yaml(params_filepath)
        
        create_directories([self.config.artifact_root])
        
    def get_data_ingestion_config(self)->DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        
        data_ingestion_config = DataIngestionConfig(
            root_dir = config.root_dir,
            source_URL= config.source_URL,
            local_data_file= config.local_data_file,
            unzip_dir= config.unzip_dir
        )
        return data_ingestion_config
    
    def get_data_transformation_config(self)->DataTransformationConfig:
        config =self.config.data_transformation
        
        create_directories([config.root_dir])
        
        data_transformation_config= DataTransformationConfig(
            root_dir = config.root_dir,
            data_path =config.data_path,
            tokenizer_name= config.tokenizer_name
            
        )
        
        return data_transformation_config 
    
    def get_model_trainer_config(self)->ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.TrainingArguments
        
        create_directories([config.root_dir])
        
        model_trainer_config = ModelTrainerConfig(
            
                root_dir= config.root_dir,
                data_path=config.data_path,
                model_ckpt= config.model_ckpt,
                num_train_epochs= params.num_train_epochs,
                warmup_steps= params.warmup_steps,
                per_device_train_batch_size= params.per_device_train_batch_size ,
                weight_decay= params.weight_decay ,
                logging_steps= params.logging_steps ,
                evaluation_strategy= params.evaluation_strategy ,
                eval_steps= params.eval_steps ,
                save_steps= params.save_steps ,
                gradient_accumulation_steps=params.gradient_accumulation_steps
        )
        
        return model_trainer_config 
    def generate_batch_sized_chunks(list_of_elements, batch_size):
        
        """split the dataset into smaller batches that we can process simultaneously
        Yield successive batch-sized chunks from list_of_elements."""

        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i : i + batch_size]
            
            
    def calculate_metric_on_test_ds(dataset, metric, model, tokenizer,
                                batch_size=16, device= "cuda" if torch.cuda.is_available() else "cpu",
                                column_text="article",
                                column_summary="highlights"):
        article_batches = list(generate_batch_sized_chunks(dataset[column_text], batch_size))
        target_batches = list(generate_batch_sized_chunks(dataset[column_summary], batch_size))

        for article_batch, target_batch in tqdm(
            zip(article_batches, target_batches), total=len(article_batches)):


            inputs = tokenizer(article_batch, max_length=1024,  truncation=True,
                            padding="max_length", return_tensors="pt")

            summaries = model.generate(input_ids=inputs["input_ids"].to(device),
                                    attention_mask=inputs["attention_mask"].to(device),
                                            length_penalty=0.8, num_beams=8, max_length=128)

            decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True,
                                    clean_up_tokenization_spaces=True)
                for s in summaries]

            metric.add_batch(predictions=decoded_summaries, references=target_batch)

        score = metric.compute()
        return score

    def evaluate(self):
        device ="cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        model_pegasus =AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path).to(device)
        
        #loading the dataset
        dataset_samsum_pt =load_from_disk(self.config.data_path)
        
        rouge_names= ["rouge1","rouge2","rougeL","rougeLsum"]
        
        rouge_metric = rouge_metric
        
        score =self.calculate_metric_on_test_ds(dataset_samsum_pt['test'][0:10], rouge_metric,model_pegasus, tokenizer, batch_size =2 ,column_text = 'dialogue', column_summary= 'summary')
        
        rouge_dict = dict((rn,score[rn].mid.fmeasure)for rn in rouge_names)
        
        df = pd.DataFrame(rouge_dict,index =['pegasus'])
        df.to_csv(self.config.metric_file_name, index= False)