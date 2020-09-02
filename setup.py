from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'gcsfs==0.6.0',
    'pandas==0.24.2',
    'google-cloud-storage==1.26.0',
    'mlflow==1.8.0',
    'joblib==0.14.1',
    'numpy==1.18.4',
    'psutil==5.7.0',
    'memoized-property==1.0.3',
    'scipy== 1.4.1',
    'tensorflow==2.0.0',
    'tensorflow-hub==0.9.0',
    'tf-models-official==2.2.0',
    'streamlit==0.64.0']

setup(
    name='Resensify',
    version='1.0',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Final Project'
)
