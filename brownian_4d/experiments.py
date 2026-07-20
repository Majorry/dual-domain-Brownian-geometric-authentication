"""Experiment definitions for ShallowWater_50km dataset.

Experiment groups:
  A: Same gain, same range, different depth (depth discrimination)
  B: Same gain (12dB), different time (cross-time robustness)
  C: Same gain (8dB), different time (cross-time robustness)
"""


def define_experiments():
    experiments = {}

    # ---- Group A: depth discrimination ----
    experiments["A1"] = {
        "title": "tx35m_8dB vs tx50m_8dB",
        "desc": "Same gain (8dB), depth 35m vs 50m",
        "group_a": ("tx35m", 8, None),
        "group_b": ("tx50m", 8, None),
    }
    experiments["A2"] = {
        "title": "tx35m_12dB vs tx50m_12dB",
        "desc": "Same gain (12dB), depth 35m vs 50m",
        "group_a": ("tx35m", 12, None),
        "group_b": ("tx50m", 12, None),
    }
    experiments["A3"] = {
        "title": "tx20m_12dB vs tx35m_12dB",
        "desc": "Same gain (12dB), depth 20m vs 35m",
        "group_a": ("tx20m", 12, None),
        "group_b": ("tx35m", 12, None),
    }
    experiments["A4"] = {
        "title": "tx20m_12dB vs tx50m_12dB",
        "desc": "Same gain (12dB), depth 20m vs 50m",
        "group_a": ("tx20m", 12, None),
        "group_b": ("tx50m", 12, None),
    }

    # ---- Group B: cross-time robustness (12dB) ----
    experiments["B1"] = {
        "title": "tx35m_12dB(11:38) vs tx20m_12dB",
        "desc": "Same gain (12dB), 35m early vs 20m full",
        "group_a": ("tx35m", 12, "11:38"),
        "group_b": ("tx20m", 12, None),
    }
    experiments["B2"] = {
        "title": "tx35m_12dB(12:55) vs tx20m_12dB",
        "desc": "Same gain (12dB), 35m mid vs 20m full",
        "group_a": ("tx35m", 12, "12:55"),
        "group_b": ("tx20m", 12, None),
    }
    experiments["B3"] = {
        "title": "tx35m_12dB(12:58) vs tx20m_12dB",
        "desc": "Same gain (12dB), 35m late vs 20m full",
        "group_a": ("tx35m", 12, "12:58"),
        "group_b": ("tx20m", 12, None),
    }
    experiments["B4"] = {
        "title": "tx35m_12dB(11:38) vs tx50m_12dB(11:11)",
        "desc": "Same gain (12dB), 35m vs 50m, 27min apart",
        "group_a": ("tx35m", 12, "11:38"),
        "group_b": ("tx50m", 12, "11:11"),
    }
    experiments["B5"] = {
        "title": "tx35m_12dB(12:55) vs tx50m_12dB(11:11)",
        "desc": "Same gain (12dB), 35m vs 50m, 104min apart",
        "group_a": ("tx35m", 12, "12:55"),
        "group_b": ("tx50m", 12, "11:11"),
    }

    # ---- Group C: cross-time robustness (8dB) ----
    experiments["C1"] = {
        "title": "tx35m_8dB(11:41) vs tx50m_8dB(11:17)",
        "desc": "Same gain (8dB), 24min apart",
        "group_a": ("tx35m", 8, "11:41"),
        "group_b": ("tx50m", 8, "11:17"),
    }
    experiments["C2"] = {
        "title": "tx35m_8dB(11:41) vs tx50m_8dB(11:23)",
        "desc": "Same gain (8dB), 18min apart",
        "group_a": ("tx35m", 8, "11:41"),
        "group_b": ("tx50m", 8, "11:23"),
    }
    experiments["C3"] = {
        "title": "tx35m_8dB(13:04) vs tx50m_8dB(11:17)",
        "desc": "Same gain (8dB), 107min apart",
        "group_a": ("tx35m", 8, "13:04"),
        "group_b": ("tx50m", 8, "11:17"),
    }
    experiments["C4"] = {
        "title": "tx35m_8dB(13:04) vs tx50m_8dB(11:23)",
        "desc": "Same gain (8dB), 101min apart",
        "group_a": ("tx35m", 8, "13:04"),
        "group_b": ("tx50m", 8, "11:23"),
    }

    return experiments
