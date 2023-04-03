#!/usr/bin/env python
from setuptools import setup, find_packages

REQUIRES = ["numpy", "scipy", "qutip"]

setup(
    name="SUMO",
    version="0.0.1",
    description="Stochastic Pseudomode solver for open quantum ssytems",
    long_description=open("README.md").read(),
    url="",
    author="Mauro Cirio, Neill Lambert, Si Luo, Pengfei Liang, Harsh Raj ",
    author_email="cirio.mauro@gmail.com",
    packages=find_packages(include=["sumo","sumo.*"]),
    install_requires=REQUIRES,
)