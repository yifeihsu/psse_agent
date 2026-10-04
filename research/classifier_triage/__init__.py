"""WLS-only triage classifiers (2026-10-04): auxiliary models that replace the exhaustive balanced screen.

A triage classifier reads the balanced WLS result of an alarmed state and
answers two questions: does this state need phase-resolved measurements, and
which balanced error family should be investigated first.  The package holds
the IEEE 14 offline benchmark (docs/classifier_triage_plan_20261004.md):
the data layer and truth-derived labels (``data``), graph features computed
from the operator's WLS solve (``features``), the size-agnostic message
passing classifier (``model``, ``train_gnn``), and the comparison against the
physics screen and the gradient-boosted models (``benchmark``).
"""
