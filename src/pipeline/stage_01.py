from src.configuration.configuration import ConfigurationManager
from src.components.data_ingestion import DataIngestion
from src.logger import logging

STAGE_NAME = "Data Ingestion stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass


    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()

if __name__ == "__main__":
    try:
        logging.info(f"Starting {STAGE_NAME} of the pipeline")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logging.info(f"{STAGE_NAME} of the pipeline is completed")
        
    except Exception as e:
        logging.exception(e)
        raise e