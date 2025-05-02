from setuptools import find_packages, setup ## For adding packages
from typing import List


HYPHEN_E_DOT = '-e .'


def get_requirements(file_path:str)->List[str]:
    """
    This function will return the list of requirements and Download the requirements.txt file
    """
    requirements=[]
    with open("requirements.txt") as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n"," ") for req in requirements]

    # removing -e . from requirements because it is a local file
    # and we don't want to install it as a package
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements
   
setup(
    name="mlproject",
    version="0.0.1",
    author="Deepak",
    author_email="deepakleveledge@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    ) 