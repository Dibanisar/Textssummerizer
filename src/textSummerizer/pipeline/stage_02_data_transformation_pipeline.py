from src.textSummerizer.config.configuration import ConfigurationManager
from src.textSummerizer.components.data_transformation import DataTransformation
from src.textSummerizer.logging import logger

STAGE_NAME = "Data Pipeline Stage"

class DataTransformationPipeline:
    def __init__(self):
        pass
    
    def initiate_data_transformation(self):
        
        config = ConfigurationManager()
        data_transformation_config =config.get_data_transformation_config()
        data_transformation = DataTransformation(config =data_transformation_config)
        data_transformation.convert()
        
if __name__ == "__main__":
    
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationPipeline()
        obj.initiate_data_transformation()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<< \n\nx========x")
    except Exception as e:
        logger.exception(e)
        raise e
