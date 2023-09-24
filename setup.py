from setuptools import setup

setup(
    name = "src",
    version = "0.0.1",
    author="Naveen_Pilli",
    author_email='naveenpilli1996@gmail.com',
    description="A samll package for Face Recognition",
    install_requirements=[
        'python==3.10.6',
        'mtcnn==0.1.0',
        'scikit-learn==1.1.3',
        'tensorflow==2.10.0',
        'keras==2.10.0',
        'keras-vggface==0.6',
        'Keras-Applications==1.0.8',
        'pyYAML',
        'tqdm'
    ]


)