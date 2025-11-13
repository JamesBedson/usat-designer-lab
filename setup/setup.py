from setuptools import setup, find_packages

setup(
    name="usat_designer",
    version="0.1.0",
    packages=find_packages(include=["usat_designer", "usat_designer.*"]),
    install_requires=[],
    include_package_data=True,
)
