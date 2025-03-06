from src.textSummerizer.logging import logger
from src.textSummerizer.pipeline.stage_01_data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.textSummerizer.pipeline.stage_02_data_transformation_pipeline import DataTransformationPipeline

STAGE_NAME = "Data Ingestion stage"

if __name__ == "__main__":
    
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.initiate_data_ingestion()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<< \n\nx========x")
    except Exception as e:
        logger.exception(e)
        raise e
    
    
STAGE_NAME = "Data Transformation Stage"
  
if __name__ == "__main__":
    
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationPipeline()
        obj.initiate_data_transformation()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<< \n\nx========x")
    except Exception as e:
        logger.exception(e)
        raise e