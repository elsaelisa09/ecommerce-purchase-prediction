from catboost import CatBoostClassifier


def create_model(params, seed):
    """Membuat model CatBoost berdasarkan konfigurasi."""

    model_params = params.copy()

    model_params.setdefault("random_seed", seed)
    model_params.setdefault("task_type", "CPU")
    model_params.setdefault("verbose", False)
    model_params.setdefault("allow_writing_files", False)

    return CatBoostClassifier(**model_params)