"""Pure IEEE-14 ordering constants shared by OpenDSS and lightweight adapters."""

BUS_ORDER = [f"b{i}" for i in range(1, 15)]

# MATPOWER IEEE-14 branch order (20 branches). Used to align Pf/Qf/Pt/Qt
# indices with case14 branch rows.
BRANCH_ORDER = [
    "Line.1-2",
    "Line.1-5",
    "Line.2-3",
    "Line.2-4",
    "Line.2-5",
    "Line.3-4",
    "Line.4-5",
    "Transformer.4-7",
    "Transformer.4-9",
    "Transformer.5-6",
    "Line.6-11",
    "Line.6-12",
    "Line.6-13",
    "Line.7-8",
    "Line.7-9",
    "Line.9-10",
    "Line.9-14",
    "Line.10-11",
    "Line.12-13",
    "Line.13-14",
]
