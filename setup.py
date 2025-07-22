from setuptools import setup, find_packages

setup(
    name='PyGenAlgo', version='1.7.1', author='Michalis Vrettas',
    author_email='michail.vrettas@gmail.com',
    description='This repository implements a genetic algorithm (GA) '
                'in Python3 programming language using only Numpy and '
                'Joblib as additional libraries.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/vrettasm/PyGeneticAlgorithms',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: GPL-3.0 License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
    install_requires=[
        'numpy',
        'joblib',
    ],
)
