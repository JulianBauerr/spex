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
    name="spex-tequila",
    version="1.0.0",
    author="Michael Lang, Julian Bauer and spex developer",
    author_email="",
    url="https://github.com/tequilahub/spex",
    description=(
        "Qubit/Fermionic expectation-value and excitation simulator for Tequila, "
        "implemented in C++ using pybind11"
    ),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    license_files=["LICENSE"],
    keywords=["quantum", "fermionic", "expectation value", "tequila", "pybind11"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    packages=find_packages(exclude=["tests", "tests.*"]),
    include_package_data=True,
    zip_safe=False,
    python_requires=">=3.9",
    install_requires=[
        "pybind11>=2.5.0",
    ],
    extras_require={
        "test": [
            "numpy",
            "tequila-basic>=1.9.0",
            "openfermion>=1.7.0",
            "pytest",
            "pyscf",
            "qulacs",
        ],
    },
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
