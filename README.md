sayama_consensus
================

Agent based model of group decision making based on work by Sayama et al (2011).
Requires Cython, NumPy, and SciPy.

Build -
python setup.py build_ext --inplace

Run -
python run.py

Experiments should be placed in the run method of sayama_consensus.pyx

References

Sayama, H., Farrell, D. & Dionne, S., 2011. The effects of mental model formation on group decision making: An agent-based simulation. Complexity, 16(3), pp.49–57.
