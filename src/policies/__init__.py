from src.policies.variants.separated_mlp import SeparatedMlpPolicy

# The Registry: Only contains custom code
POLICY_REGISTRY = {
    "SeparatedMlpPolicy": SeparatedMlpPolicy,
    # Add new custom policies here later...
}