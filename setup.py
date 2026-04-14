from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
import os

ext_modules = [
    Pybind11Extension(
        "spex_tequila",
        ["spex.cpp"],
        cxx_std=17,
        include_dirs=[os.path.abspath("include")]
    ),
]

setup(
    name="spex-tequila",
    version="1.0.0",
    author="Michael Lang",
    author_email="lang-michi@t-online.de",
    url="https://git.rz.uni-augsburg.de/qalg-a/spex",
    description="Expectation value computation module on sparse Pauli states for Tequila, implemented in C++ using Pybind11",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.9",
    install_requires=[
        "pybind11>=2.5.0", 'spex-tequila'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
    ],
    packages=find_packages(),
    include_package_data=True,
)
