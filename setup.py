from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        [req.replace("\n","") for req in requirements]


setup(
    name="mlproject",
    version='0.0.1',
    author='Praveen Tak',
    author_email='praveentak715@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')

)