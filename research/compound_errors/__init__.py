"""Compound-error study (2026-10-27, stage 1): the rule expert's ceiling on randomly combined errors.

``build_suite`` draws the class suite with the generator's compound
synthesizer, ``rollout`` rolls the expert out on it in parallel and
summarizes per class, ``audit`` runs the decision audit on the failed roots
and a sample of solved ones, and ``report`` assembles the study note's
tables.  Research only; nothing here trains a policy.
"""
