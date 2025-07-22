from setuptools import setup, find_packages

with open("README.md", encoding="utf8") as readme_file:
    readme_txt = readme_file.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name='PyGenAlgo',
    version='1.7.1',
    author='Michalis Vrettas, PhD',
    author_email='michail.vrettas@gmail.com',
    description='Genetic Algorithms toolbox in Python3',
    long_description=readme_txt,
    long_description_content_type='text/markdown',
    url='https://github.com/vrettasm/PyGeneticAlgorithms',
    packages=find_packages(exclude=["docs", "tests"]),
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
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
