from src.algorithms.variants.simple_ac import SimpleActorCritic

# Maps string names from JSON to actual Python classes
ALGO_REGISTRY = {
    "SimpleActorCritic": SimpleActorCritic
}