from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
import os

ext_modules = [
    Pybind11Extension(
        "spex_tequila",
        ["spex.cpp"],
        cxx_std=17,
        include_dirs=[os.path.abspath("include")],
    ),
]

setup(
    name="spex-fermionic-tequila",
    version="1.0.0",
    author="Julian Bauer",
    author_email="julian.bauer@mbtj.de",
    url="https://git.rz.uni-augsburg.de/qalg-a/spex",
    description=(
        "Fermionic excitation, fSWAP and expectation-value simulator for Tequila, "
        "implemented in C++ using pybind11"
    ),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    license_files=["LICENSE"],
    keywords=["quantum", "fermionic", "expectation value", "tequila", "pybind11"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    py_modules=["spex_expval"],
    packages=find_packages(exclude=["tests", "tests.*"]),
    include_package_data=True,
    zip_safe=False,
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "tequila-basic>=1.9.0",
        "openfermion>=1.7.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
    ],
)
