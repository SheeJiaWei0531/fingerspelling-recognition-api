from setuptools import setup, find_packages
from pip._internal.req import parse_requirements

setup(
    name='fingerspelling',
    version='1.0.0',
    url='https://github.com/pinology/fingerspelling-recognition.git',
    author='Pin C',
    # author_email='author@gmail.com',
    description='A machine learning model to recognize alphabet and numbers in American Sign Language.',
    packages=find_packages(),    
    # install_requires=list(parse_requirements('requirements.txt', session='hack')),
)