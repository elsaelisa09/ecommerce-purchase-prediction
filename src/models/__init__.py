from .logistic_regression import (
    create_model as create_logistic_regression,
)
from .random_forest import (
    create_model as create_random_forest,
)
from .xgboost_model import (
    create_model as create_xgboost,
)
from .dummy_classifier import (
    create_model as create_dummy_classifier,
)

from .catboost_model import (
    create_model as create_catboost,
)

MODEL_BUILDERS = {
    "logistic_regression": create_logistic_regression,
    "random_forest": create_random_forest,
    "xgboost": create_xgboost,
    "dummy": create_dummy_classifier,
    "catboost": create_catboost,
}


def build_model(model_config, seed):
    """Memilih model berdasarkan nama di konfigurasi."""

    model_name = model_config["name"]

    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"Model belum terdaftar: {model_name}")

    params = model_config.get("params", {})

    return MODEL_BUILDERS[model_name](
        params=params,
        seed=seed,
    )