from sklearn.ensemble import RandomForestClassifier


def create_model(params, seed):
    """Membuat model Random Forest dari konfigurasi."""

    model_params = params.copy()
    model_params.setdefault("random_state", seed)

    return RandomForestClassifier(**model_params)