from src.textSummerizer.config.configuration import ConfigurationManager
from src.textSummerizer.components.data_transformation import DataTransformation
from src.textSummerizer.logging import logger
from src.textSummerizer.components.Model_trainer import ModelTrainer

STAGE_NAME = "Model Trainer"


class ModelTrainerConfig:
    def __init__(self):
        pass
    
    def initiate_model_trainer(self):
        
        config =ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(config =model_trainer_config)
        model_trainer.train()
        
if __name__ == "__main__":
    
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainerConfig()
        obj.initiate_model_trainer()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<< \n\nx========x")
    except Exception as e:
        logger.exception(e)
        raise e

        
        