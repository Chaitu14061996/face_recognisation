from  src.utils.all_utils import  read_yaml,create_directory
import os
import logging
import pickle
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from keras_vggface.vggface import VGGFace
from mtcnn import MTCNN
from keras.applications.vgg16 import preprocess_input
import cv2
from io import BytesIO

logging_str="[%(asctime)s: %(levelname)s: %(module)s ] : %(message)s"
log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

logging.basicConfig(filename=os.path.join(log_dir,'Running_logging.log'),
level=logging.INFO,format=logging_str,filemode='a')


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
Model_name=params['base']['BASE_MODEL']
include_top=params['base']['include_top']
pooling=params['base']['pooling']


detector=MTCNN()
model=VGGFace(model=Model_name,include_top=include_top,input_shape=(224,224,3),pooling=pooling)
file_names=pickle.load(open(pickle_file, 'rb'))
features_list=pickle.load(open(feature_name,'rb'))

# img_path=r'C:\Users\Admin\Downloads\Ajay-Devgan-Net-Worth.jpeg'

def read_image(image_encoded):
    try:
        pil_img=Image.open(BytesIO(image_encoded))
        pil_img=np.array(pil_img)

        logging.info('Image uploaded successfully')

        return pil_img


    except Exception as e:
        logging.info('Image upload failed')


    


def preprocess(img_path):

    # img=cv2.imread(img_path)
    try:
        image_rgb = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
        image_rgb=np.array(image_rgb)
        result=detector.detect_faces(image_rgb)
        x,y,width,height=result[0]['box']
        face=image_rgb[y:y+height, x:x+width]
        face=cv2.resize(face,(224,224))
        face=np.expand_dims(face,axis=0)
        face=preprocess_input(face)
        result=model.predict(face).flatten()

        logging.info("Image Features Extracted Successfully")

        return result


    except Exception as e:
        logging.info("Image Features Extracted Failed")


# proc_img=preprocess(img_path)

def predict(result,features_list):

    try:
        similarity=[]
        for i in range(len(features_list)):
            similarity.append(cosine_similarity(result.reshape(1,-1),features_list[i].reshape(1,-1))[0][0])
        index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
        consi = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][1]

        logging.info("Cosine similarity values of New Image extracted successfully")

    except Exception as e:
        logging.info("Cosine similarity values of New Image extracted failed")
    
    try:
        predictor=" ".join(file_names[index_pos].split("\\")[1].split("_"))

        if consi>0.5:
            logging.info(f"Predict Successfully : {predictor}, Cosine_similarity: {consi}")
            return predictor
        else:
            logging.info("Predict Failed : Mr/Ms. Not belongs to Indian Acters")
            return "Mr/Ms. Not belongs to Indian Acters"
    except Exception as e:
        logging.info("Predict Failed ")

    # return predictor

# prid=predict(proc_img,features_list)
# print(prid)

