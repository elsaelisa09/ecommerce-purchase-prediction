from sklearn.dummy import DummyClassifier


def create_model(params, seed):
    """Membuat model referensi sederhana."""

    model_params = params.copy()
    model_params.setdefault("random_state", seed)

    return DummyClassifier(**model_params)