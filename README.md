# SPEX
spex is an expectation value computation module on sparse Pauli states for [Tequila](https://github.com/tequilahub/tequila), implemented in C++ using Pybind11. It provides computation of expectation values, inner products, and application of exponential Pauli operators on sparse quantum states.

# Install
From PyPI:
```
pip install spex-tequila
```

From source with test dependencies (for development):
```
git clone https://github.com/tequilahub/spex.git
cd spex
pip install -e ".[test]"
```