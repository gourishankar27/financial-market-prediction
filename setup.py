from setuptools import setup, find_packages

setup(
    name="sp500-tft",
    version="0.0.1",
    packages=find_packages(include=['models', 'src', 'train', 'scripts']),
)
