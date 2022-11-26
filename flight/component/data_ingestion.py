from flight.entity.config_entity import DataIngestionConfig
import sys,os
from flight.exception import FlightException
from flight.logger import logging
from flight.entity.artifact_entity import DataIngestionArtifact
import zipfile 
import numpy as np
from six.moves import urllib
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

class DataIngestion:

    def __init__(self,data_ingestion_config:DataIngestionConfig ):
        try:
            logging.info(f"{'>>'*20}Data Ingestion log started.{'<<'*20} ")
            self.data_ingestion_config = data_ingestion_config

        except Exception as e:
            raise FlightException(e,sys)
    

    def download_flight_data(self,) -> str:
        try:
            #extraction remote url to download dataset
            download_url = self.data_ingestion_config.dataset_download_url

            #folder location to download file
            tgz_download_dir = self.data_ingestion_config.tgz_download_dir
            
            os.makedirs(tgz_download_dir,exist_ok=True)

            flight_file_name = 'flight.zip'

            tgz_file_path = os.path.join(tgz_download_dir, flight_file_name)

            logging.info(f"Downloading file from :[{download_url}] into :[{tgz_file_path}]")
            urllib.request.urlretrieve(download_url, tgz_file_path)
            logging.info(f"File :[{tgz_file_path}] has been downloaded successfully.")
            return tgz_file_path

        except Exception as e:
            raise FlightException(e,sys) from e

    def extract_tgz_file(self,tgz_file_path:str):
        try:
            raw_data_dir = self.data_ingestion_config.raw_data_dir

            if os.path.exists(raw_data_dir):
                os.remove(raw_data_dir)

            os.makedirs(raw_data_dir,exist_ok=True)
            logging.info(f"Extracting tgz file: [{tgz_file_path}] into dir: [{raw_data_dir}]")
            with zipfile.ZipFile(tgz_file_path) as myzip:
                myzip.extractall(path=raw_data_dir)
            logging.info(f"Extraction completed")

        except Exception as e:
            raise FlightException(e,sys) from e
    
    def split_data_as_train_test(self) -> DataIngestionArtifact:
        try:
            raw_data_dir = self.data_ingestion_config.raw_data_dir

            file_name = os.listdir(raw_data_dir)[0]

            flight_file_path = os.path.join(raw_data_dir,file_name)


            logging.info(f"Reading csv file: [{flight_file_path}]")
            flight_data_frame = pd.read_csv(flight_file_path)

            flight_data_frame.dropna(inplace=True)

            flight_data_frame["journey_Date"]= pd.to_datetime(flight_data_frame['Date_of_Journey'], format= "%d/%m/%Y").dt.day
            flight_data_frame["journey_Month"]= pd.to_datetime(flight_data_frame['Date_of_Journey'], format= "%d/%m/%Y").dt.month
            flight_data_frame.drop(['Date_of_Journey'],axis=1,inplace=True)

            flight_data_frame['Dep_hour']=pd.to_datetime(flight_data_frame['Dep_Time']).dt.hour  #pd.to_datetime
            flight_data_frame['Dep_min']=pd.to_datetime(flight_data_frame['Dep_Time']).dt.minute
            flight_data_frame.drop(['Dep_Time'],axis=1,inplace=True)

            flight_data_frame['Arrival_hour']=pd.to_datetime(flight_data_frame['Arrival_Time']).dt.hour  #pd.to_datetime
            flight_data_frame['Arrival_min']=pd.to_datetime(flight_data_frame['Arrival_Time']).dt.minute
            flight_data_frame.drop(['Arrival_Time'],axis=1,inplace=True)

            # Assigning and converting Duration column into list to extract hours ans minutes seperately
            duration = list(flight_data_frame["Duration"])
            for i in range(len(duration)):
                if len(duration[i].split()) !=2:  # Check if duration contains only hour or mins
                    if "h" in duration[i]:
                        duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
                    else:
                        duration[i] = "0h " + duration[i]           # Adds 0 hour

            duration_hours = []
            duration_mins = []
            for i in range(len(duration)):
                duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
                duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration

            flight_data_frame["Duration_hours"] = duration_hours
            flight_data_frame["Duration_mins"] = duration_mins
            flight_data_frame.drop(['Duration'],axis=1,inplace=True)
            flight_data_frame.drop(["Route", "Additional_Info"], axis = 1, inplace = True)
            flight_data_frame.replace({'non-stop':0,'1 stop':1,'2 stops':2,'3 stops':3,'4 stops':4},inplace=True)
            
            flight_data_frame.reset_index(inplace=True,drop=True)
            flight_data_frame["binned"] = pd.cut(
                flight_data_frame["Price"],
                bins=[0,1759,5277,8372,12373,np.inf],
                labels=[1,2,3,4,5]
            )

            logging.info(f"Splitting data into train and test")
            strat_train_set = None
            strat_test_set = None

            split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

            for train_index,test_index in split.split(flight_data_frame, flight_data_frame["binned"]):
                strat_train_set = flight_data_frame.loc[train_index].drop(["binned"],axis=1)
                strat_test_set = flight_data_frame.loc[test_index].drop(["binned"],axis=1)

            train_file_path = os.path.join(self.data_ingestion_config.ingested_train_dir,
                                            file_name)

            test_file_path = os.path.join(self.data_ingestion_config.ingested_test_dir,
                                        file_name)
            
            if strat_train_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_train_dir,exist_ok=True)
                logging.info(f"Exporting training datset to file: [{train_file_path}]")
                strat_train_set.to_csv(train_file_path,index=False)

            if strat_test_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_test_dir, exist_ok= True)
                logging.info(f"Exporting test dataset to file: [{test_file_path}]")
                strat_test_set.to_csv(test_file_path,index=False)

            data_ingestion_artifact = DataIngestionArtifact(train_file_path=train_file_path,
                                test_file_path=test_file_path,
                                is_ingested=True,
                                message=f"Data ingestion completed successfully."
                                )
            logging.info(f"Data Ingestion artifact:[{data_ingestion_artifact}]")
            return data_ingestion_artifact

        except Exception as e:
            raise FlightException(e,sys) from e

    def initiate_data_ingestion(self)-> DataIngestionArtifact:
        try:
            tgz_file_path =  self.download_flight_data()
            self.extract_tgz_file(tgz_file_path=tgz_file_path)
            return self.split_data_as_train_test()
        except Exception as e:
            raise FlightException(e,sys) from e
    


    def __del__(self):
        logging.info(f"{'>>'*20}Data Ingestion log completed.{'<<'*20} \n\n")