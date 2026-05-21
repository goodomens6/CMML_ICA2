from __future__ import annotations


TRUE_TO_HARMONIZED = {
    "activated_stellate": "stellate",
    "quiescent_stellate": "stellate",
}

SCIMILARITY_EXACT_MAP = {
    "pancreatic A cell": "alpha",
    "type B pancreatic cell": "beta",
    "pancreatic D cell": "delta",
    "pancreatic acinar cell": "acinar",
    "pancreatic ductal cell": "ductal",
    "pancreatic stellate cell": "stellate",
    "mast cell": "mast",
    "macrophage": "macrophage",
}


def harmonize_true_label(label: str) -> str:
    return TRUE_TO_HARMONIZED.get(label, label)


def harmonize_dataset_prediction(label: str) -> str:
    return harmonize_true_label(label)


def harmonize_scimilarity_prediction(label: str) -> str:
    if label in SCIMILARITY_EXACT_MAP:
        return SCIMILARITY_EXACT_MAP[label]
    if "endothelial cell" in label:
        return "endothelial"
    if label == "T cell" or label.endswith(" T cell"):
        return "t_cell"
    return "other"
