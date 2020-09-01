from setuptools import find_packages
from setuptools import setup
REQUIRED_PACKAGES = [
    # streamlit
    'streamlit==0.64.0',
    # data manipulation
    'numpy==1.18.4',
    'pandas==0.24.2']
setup(
    name='WagonDataStreamlitDemo',
    version='1.0',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Wagon Data Streamlit Demo'
)
