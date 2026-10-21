from xgboost import XGBClassifier


def create_model(params, seed):
    """Membuat model XGBoost dari konfigurasi."""

    model_params = params.copy()
    model_params.setdefault("random_state", seed)

    return XGBClassifier(**model_params)