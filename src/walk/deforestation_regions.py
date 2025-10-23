"""
Global Deforestation Hotspots

Based on 2023-2024 data from Global Forest Watch, MAAP, and WWF.
These regions represent active tropical deforestation fronts.

Source: https://www.globalforestwatch.org/
"""

DEFORESTATION_HOTSPOTS = {
    # AMAZON BASIN
    "amazon_para_brazil": {
        "name": "Pará, Brazil (Amazon)",
        "description": "BR-163 road corridor, major soy frontier",
        "bounds": {
            "min_lat": -8.0,
            "max_lat": -3.0,
            "min_lon": -55.0,
            "max_lon": -50.0,
        },
        "region": "Amazon",
        "country": "Brazil",
        "deforestation_rate": "high",  # 28.4% drop from peak, but still active
    },

    "amazon_mato_grosso_brazil": {
        "name": "Mato Grosso, Brazil (Amazon)",
        "description": "Soy frontier, cattle ranching",
        "bounds": {
            "min_lat": -12.0,
            "max_lat": -8.0,
            "min_lon": -60.0,
            "max_lon": -54.0,
        },
        "region": "Amazon",
        "country": "Brazil",
        "deforestation_rate": "high",  # 45.1% drop from peak
    },

    "amazon_rondonia_brazil": {
        "name": "Rondônia, Brazil (Amazon)",
        "description": "Historic deforestation arc",
        "bounds": {
            "min_lat": -13.0,
            "max_lat": -8.0,
            "min_lon": -66.0,
            "max_lon": -60.0,
        },
        "region": "Amazon",
        "country": "Brazil",
        "deforestation_rate": "very_high",  # 62.5% drop, but from very high baseline
    },

    "amazon_santa_cruz_bolivia": {
        "name": "Santa Cruz, Bolivia (Amazon)",
        "description": "Soy frontier, record 2024 loss",
        "bounds": {
            "min_lat": -18.0,
            "max_lat": -14.0,
            "min_lon": -64.0,
            "max_lon": -60.0,
        },
        "region": "Amazon",
        "country": "Bolivia",
        "deforestation_rate": "very_high",  # Record 476k ha in 2024
    },

    "amazon_madre_de_dios_peru": {
        "name": "Madre de Dios, Peru (Amazon)",
        "description": "Gold mining frontier",
        "bounds": {
            "min_lat": -13.5,
            "max_lat": -10.0,
            "min_lon": -72.0,
            "max_lon": -68.0,
        },
        "region": "Amazon",
        "country": "Peru",
        "deforestation_rate": "high",
    },

    # CONGO BASIN
    "congo_basin_drc_east": {
        "name": "Eastern DRC (Congo Basin)",
        "description": "75% of Congo Basin deforestation",
        "bounds": {
            "min_lat": -5.0,
            "max_lat": 0.0,
            "min_lon": 20.0,
            "max_lon": 25.0,
        },
        "region": "Congo",
        "country": "DRC",
        "deforestation_rate": "very_high",  # Record 526k ha in 2023
    },

    "congo_basin_drc_central": {
        "name": "Central DRC (Congo Basin)",
        "description": "Expanding deforestation front",
        "bounds": {
            "min_lat": -3.0,
            "max_lat": 2.0,
            "min_lon": 22.0,
            "max_lon": 27.0,
        },
        "region": "Congo",
        "country": "DRC",
        "deforestation_rate": "high",
    },

    "congo_basin_republic_congo": {
        "name": "Republic of Congo (Congo Basin)",
        "description": "150% increase in 2024",
        "bounds": {
            "min_lat": -5.0,
            "max_lat": -1.0,
            "min_lon": 12.0,
            "max_lon": 16.0,
        },
        "region": "Congo",
        "country": "Republic of Congo",
        "deforestation_rate": "very_high",  # Fire-driven
    },

    "congo_basin_cameroon": {
        "name": "Cameroon (Congo Basin)",
        "description": "Uptick in recent years",
        "bounds": {
            "min_lat": 2.0,
            "max_lat": 6.0,
            "min_lon": 10.0,
            "max_lon": 14.0,
        },
        "region": "Congo",
        "country": "Cameroon",
        "deforestation_rate": "medium",
    },

    # SOUTHEAST ASIA
    "sumatra_riau_indonesia": {
        "name": "Riau, Sumatra (Indonesia)",
        "description": "Palm oil expansion, peatland conversion",
        "bounds": {
            "min_lat": -1.0,
            "max_lat": 2.0,
            "min_lon": 100.0,
            "max_lon": 104.0,
        },
        "region": "Southeast Asia",
        "country": "Indonesia",
        "deforestation_rate": "high",
    },

    "sumatra_aceh_indonesia": {
        "name": "Aceh, Sumatra (Indonesia)",
        "description": "Leuser ecosystem encroachment",
        "bounds": {
            "min_lat": 3.0,
            "max_lat": 6.0,
            "min_lon": 96.0,
            "max_lon": 99.0,
        },
        "region": "Southeast Asia",
        "country": "Indonesia",
        "deforestation_rate": "medium",
    },

    "papua_indonesia": {
        "name": "Papua, Indonesia",
        "description": "New frontier, protected area encroachment",
        "bounds": {
            "min_lat": -6.0,
            "max_lat": -2.0,
            "min_lon": 136.0,
            "max_lon": 141.0,
        },
        "region": "Southeast Asia",
        "country": "Indonesia",
        "deforestation_rate": "high",
    },

    "borneo_sabah_malaysia": {
        "name": "Sabah, Borneo (Malaysia)",
        "description": "Palm oil, logging",
        "bounds": {
            "min_lat": 4.0,
            "max_lat": 7.0,
            "min_lon": 115.0,
            "max_lon": 119.0,
        },
        "region": "Southeast Asia",
        "country": "Malaysia",
        "deforestation_rate": "medium",  # Declining but still active
    },
}


def get_regions_by_rate(rate="high"):
    """Get regions filtered by deforestation rate."""
    return {
        name: info
        for name, info in DEFORESTATION_HOTSPOTS.items()
        if (rate == "high" and info["deforestation_rate"] in [rate, "very_high"])
        or (rate != "high" and info["deforestation_rate"] == rate)
    }


def get_regions_by_continent(continent):
    """Get regions filtered by continent/basin."""
    continent_map = {
        "amazon": "Amazon",
        "congo": "Congo",
        "asia": "Southeast Asia",
    }
    target = continent_map.get(continent.lower(), continent)
    return {
        name: info
        for name, info in DEFORESTATION_HOTSPOTS.items()
        if info["region"] == target
    }


# Intact Forest Bastions - Stable, protected cores for intact sampling
INTACT_FOREST_BASTIONS = {
    # AMAZON CORE
    "amazon_core_brazil": {
        "name": "Amazon Core, Brazil",
        "description": "Deep Amazon, protected reserves",
        "bounds": {
            "min_lat": -5.0,
            "max_lat": 0.0,
            "min_lon": -70.0,
            "max_lon": -65.0,
        },
        "region": "Amazon",
        "country": "Brazil",
    },
    "amazon_guiana_shield": {
        "name": "Guiana Shield (Suriname/Guyana)",
        "description": "Low deforestation, high forest cover",
        "bounds": {
            "min_lat": 2.0,
            "max_lat": 6.0,
            "min_lon": -58.0,
            "max_lon": -54.0,
        },
        "region": "Amazon",
        "country": "Guyana/Suriname",
    },

    # CONGO CORE
    "congo_core_gabon": {
        "name": "Gabon Central",
        "description": "High forest cover, protected",
        "bounds": {
            "min_lat": -2.0,
            "max_lat": 1.0,
            "min_lon": 10.0,
            "max_lon": 14.0,
        },
        "region": "Congo",
        "country": "Gabon",
    },
    "congo_core_drc_northwest": {
        "name": "Northwest DRC Core",
        "description": "Remote, low-access forest",
        "bounds": {
            "min_lat": 0.0,
            "max_lat": 3.0,
            "min_lon": 20.0,
            "max_lon": 24.0,
        },
        "region": "Congo",
        "country": "DRC",
    },

    # SOUTHEAST ASIA PROTECTED
    "borneo_core_kalimantan": {
        "name": "Central Kalimantan Protected",
        "description": "National parks, lower deforestation",
        "bounds": {
            "min_lat": -2.0,
            "max_lat": 1.0,
            "min_lon": 112.0,
            "max_lon": 116.0,
        },
        "region": "Southeast Asia",
        "country": "Indonesia",
    },
}


def get_diverse_sample(n_regions=5):
    """
    Get a diverse sample of regions across continents.

    Returns regions balanced across Amazon, Congo, and Asia.
    """
    amazon = [k for k, v in DEFORESTATION_HOTSPOTS.items() if v["region"] == "Amazon"]
    congo = [k for k, v in DEFORESTATION_HOTSPOTS.items() if v["region"] == "Congo"]
    asia = [k for k, v in DEFORESTATION_HOTSPOTS.items() if v["region"] == "Southeast Asia"]

    # Balance across continents
    import random
    random.seed(42)

    selected = []
    selected.extend(random.sample(amazon, min(2, len(amazon))))
    selected.extend(random.sample(congo, min(2, len(congo))))
    selected.extend(random.sample(asia, min(1, len(asia))))

    return {k: DEFORESTATION_HOTSPOTS[k] for k in selected[:n_regions]}


def get_intact_bastions(n_regions=5):
    """
    Get intact forest bastions for negative class sampling.

    Returns protected/stable regions with high forest cover.
    """
    amazon = [k for k, v in INTACT_FOREST_BASTIONS.items() if v["region"] == "Amazon"]
    congo = [k for k, v in INTACT_FOREST_BASTIONS.items() if v["region"] == "Congo"]
    asia = [k for k, v in INTACT_FOREST_BASTIONS.items() if v["region"] == "Southeast Asia"]

    # Balance across continents
    import random
    random.seed(42)

    selected = []
    selected.extend(random.sample(amazon, min(2, len(amazon))))
    selected.extend(random.sample(congo, min(2, len(congo))))
    selected.extend(random.sample(asia, min(1, len(asia))))

    return {k: INTACT_FOREST_BASTIONS[k] for k in selected[:n_regions]}
