import os

def execute_system():
    bash1=r'C:\PROGRAMMING\Assignment\JTP_Assignments\VGG_Face_Recognition_Project\src\01_generate_image_pkl_file.py'
    bash2=r'C:\PROGRAMMING\Assignment\JTP_Assignments\VGG_Face_Recognition_Project\src\02_Featur_extractor.py'

    os.system(bash1)
    os.system(bash2)


if __name__ == '__main__':
    execute_system()