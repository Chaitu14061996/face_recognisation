
import os
import time
import pickle
import uvicorn
from fastapi import FastAPI 
from fastapi import UploadFile,File
from Predictions import read_image
from Predictions import preprocess
from Predictions import predict
from  src.utils.all_utils import  read_yaml,create_directory


logging_str="[%(asctime)s: %(levelname)s: %(module)s ] : %(message)s"
log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

app = FastAPI()

config=read_yaml('config/config.yaml')
params=read_yaml('params.yaml')

artifacts=config['artifacts']
artifacts_dir=artifacts['artifacts_dir']

# Upload Image_path
upload_img_dir=artifacts['upload_img_dir']
upload_path=os.path.join(artifacts_dir,upload_img_dir)

# pickle formate data paths
pickle_format_data_dir=artifacts['pickle_format_data_dir']
img_pickle_file_name=artifacts['img_pickle_file_name']

raw_local_dir_path=os.path.join(artifacts_dir,pickle_format_data_dir)
pickle_file=os.path.join(raw_local_dir_path,img_pickle_file_name)

# features paths 

feature_extractor_dir=artifacts['features_extraction_dir']
extracted_feature_names=artifacts['extracted_features_name']

feature_extraction_path=os.path.join(artifacts_dir,feature_extractor_dir)
feature_name=os.path.join(feature_extraction_path,extracted_feature_names)

# Params Paths
features_list=pickle.load(open(feature_name,'rb'))


@app.get("/index")

async def index():
    return {"message": "Welcome To Movie acter Prediction"}

@app.post('/api/predict')
async def predict_image(file:UploadFile=File(...)):
    start=time.perf_counter()
    image= await file.read()
    image=read_image(image)
    image=preprocess(image)
    image=predict(image,features_list)
    end=time.perf_counter()
    time_consume=end-start

    return f"Prediction_Face : {image}",f"time_consumption to predicting New Face is {time_consume} Seconds"



if __name__ == "__main__":
    uvicorn.run(
        app,
        port=5500)