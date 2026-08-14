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

IEEE14_LOAD_BASE_KW = {
    "b2": 21700.0,
    "b3": 94200.0,
    "b4": 47800.0,
    "b5": 7600.0,
    "b6": 11200.0,
    "b9": 29500.0,
    "b10": 9000.0,
    "b11": 3500.0,
    "b12": 6100.0,
    "b13": 13500.0,
    "b14": 14900.0,
}

IEEE14_GENERATOR_DISPATCH_KW = {
    "b2": 40000.0,
    "b3": 1.0,
    "b6": 1.0,
    "b8": 1.0,
}

IEEE14_GENERATOR_VOLTAGE_PU = {
    "b2": 1.045,
    "b3": 1.010,
    "b6": 1.070,
    "b8": 1.090,
}

IEEE14_SOURCE_VOLTAGE_PU = 1.060

IEEE14_OPERATING_POINT_KEYS = (
    "load_scale",
    "bus_load_scales",
    "generator_dispatch_kw",
    "voltage_setpoints_pu",
    "source_voltage_pu",
)
