sayama_consensus
================

Agent based model of group decision making based on work by Sayama et al (2011). Implemented in Python, Cython, and Julia.

Requires Cython (for the Cython version), NumPy, and SciPy (for all three version).
The Julia version also makes use of the PyCall package to use an optimisation routine from SciPy.

Cython implementation

Built thusly:

    python setup.py build_ext --inplace

Run like so:

    python run.py

Experiments should be placed in the run method of sayama_consensus.pyx

Python version

Simply run with python.

Julia version

Install PyCall with -

    Pkg.add(pycall)

Run with -

    julia driver.jl

All three support multiple processes, but default to using 2.

Python version is recommended, as somewhat counter intuitively, it seems to run faster than the other two versions.


References

Sayama, H., Farrell, D. & Dionne, S., 2011. The effects of mental model formation on group decision making: An agent-based simulation. Complexity, 16(3), pp.49–57.
