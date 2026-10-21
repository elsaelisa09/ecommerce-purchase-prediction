from hashlib import sha256
from pathlib import Path

import joblib
import wandb
import yaml
import matplotlib.pyplot as plt
from src.evaluation import create_confusion_matrix
from src.data_loader import load_raw_data, validate_manifest, get_split
from src.training import train_model
from src.evaluation import evaluate_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    # Membaca konfigurasi.
    config_path = PROJECT_ROOT / "configs" / "experiment.yaml"

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # Membaca dan memeriksa data.
    df, manifest = load_raw_data(config["data"])
    validate_manifest(df, manifest)

    # Mengambil training dan validation sesuai manifest.
    datasets = {}

    for split_name in ["train", "validation"]:
        datasets[split_name] = get_split(
            df,
            manifest,
            split_name=split_name,
            target_column=config["data"]["target"],
            exclude_columns=config["features"]["exclude_columns"],
        )

    X_train, y_train = datasets["train"]
    X_val, y_val = datasets["validation"]

    # Membuka satu run eksperimen di W&B.
    with wandb.init(
        project=config["wandb"]["project"],
        name=config["experiment_name"],
        config=config,
        mode=config["wandb"]["mode"],
        job_type="train",
    ) as run:
        # Mencatat fitur dan identitas isi file sumber.
        run.config.update({
            "feature_columns": X_train.columns.tolist(),
            "train_sessions": len(X_train),
            "validation_sessions": len(X_val),
            "dataset_sha256": sha256(
                (PROJECT_ROOT / config["data"]["path"]).read_bytes()
            ).hexdigest(),
            "split_sha256": sha256(
                (PROJECT_ROOT / config["data"]["split_path"]).read_bytes()
            ).hexdigest(),
        })

        print("Memulai training:", config["experiment_name"])
        pipeline = train_model(X_train, y_train, config)

        # Mengevaluasi validation set.
        metrics, predictions, probabilities = evaluate_model(
            pipeline,
            X_val,
            y_val,
            threshold=config["evaluation"]["threshold"],
        )

        # Mencatat metrik ke W&B.
        run.log({
            f"validation/{name}": float(value)
            for name, value in metrics.items()
        })

        # Menyimpan pipeline ke folder unik untuk setiap run.
        output_dir = PROJECT_ROOT / config["output_dir"] / run.id
        output_dir.mkdir(parents=True, exist_ok=True)

        model_path = output_dir / "pipeline.joblib"
        joblib.dump(pipeline, model_path)

        # Mengunggah pipeline, konfigurasi, dan manifest ke W&B.
        artifact = wandb.Artifact(
            name=f"{config['model']['name']}-pipeline",
            type="model",
        )
        artifact.add_file(str(model_path))
        artifact.add_file(str(config_path))
        artifact.add_file(
            str(PROJECT_ROOT / config["data"]["split_path"])
        )
        run.log_artifact(artifact)

        print("\nHasil validation:")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

        print("\nPipeline disimpan:", model_path)
        print("Link W&B:", run.url)

def document_saved_run(run_path):
    """Membuat confusion matrix dari pipeline yang sudah tersimpan."""

    # Membaca konfigurasi asli eksperimen dari W&B.
    entity, project, run_id = run_path.split("/")
    config = dict(wandb.Api().run(run_path).config)

    df, manifest = load_raw_data(config["data"])
    validate_manifest(df, manifest)

    # Memastikan dataset dan pembagiannya sama seperti saat training.
    for path_key, hash_key in [
        ("path", "dataset_sha256"),
        ("split_path", "split_sha256"),
    ]:
        file_path = PROJECT_ROOT / config["data"][path_key]
        current_hash = sha256(file_path.read_bytes()).hexdigest()

        if current_hash != config[hash_key]:
            raise ValueError(
                f"File {file_path.name} berbeda dari saat training."
            )

    # Mengambil data validation.
    X_val, y_val = get_split(
        df,
        manifest,
        split_name="validation",
        target_column=config["data"]["target"],
        exclude_columns=config["features"]["exclude_columns"],
    )

    # Memuat pipeline tanpa melakukan training ulang.
    output_dir = PROJECT_ROOT / config["output_dir"] / run_id
    pipeline = joblib.load(output_dir / "pipeline.joblib")

    _, predictions, _ = evaluate_model(
        pipeline,
        X_val,
        y_val,
        threshold=config["evaluation"]["threshold"],
    )

    # Membuat dan menyimpan grafik di laptop.
    fig = create_confusion_matrix(y_val, predictions)
    figure_path = output_dir / "validation_confusion_matrix.png"

    fig.savefig(figure_path, dpi=150)
    plt.close(fig)

    # Menambahkan grafik ke run W&B yang sudah ada.
    with wandb.init(
        entity=entity,
        project=project,
        id=run_id,
        resume="must",
        mode="online",
    ) as run:
        run.log({
            "validation/confusion_matrix": wandb.Image(
                str(figure_path)
            )
        })

        print("Grafik disimpan:", figure_path)
        print("Link W&B:", run.url)

def evaluate_final_selection():
    """Mengevaluasi model terpilih pada test tanpa training ulang."""

    selection_path = PROJECT_ROOT / "configs" / "final_selection.yaml"

    with selection_path.open("r", encoding="utf-8") as file:
        selection = yaml.safe_load(file)

    # Membaca konfigurasi asli model terpilih.
    source_path = selection["source_run_path"]
    entity, project, source_run_id = source_path.split("/")

    source_run = wandb.Api().run(source_path)
    source_config = dict(source_run.config)

    df, manifest = load_raw_data(source_config["data"])
    validate_manifest(df, manifest)

    # Memastikan file data sama seperti saat training.
    for path_key, hash_key in [
        ("path", "dataset_sha256"),
        ("split_path", "split_sha256"),
    ]:
        file_path = PROJECT_ROOT / source_config["data"][path_key]
        current_hash = sha256(file_path.read_bytes()).hexdigest()

        if current_hash != source_config[hash_key]:
            raise ValueError(
                f"File {file_path.name} berbeda dari saat training."
            )

    X_test, y_test = get_split(
        df,
        manifest,
        split_name="test",
        target_column=source_config["data"]["target"],
        exclude_columns=source_config["features"]["exclude_columns"],
    )

    pipeline_path = (
        PROJECT_ROOT
        / source_config["output_dir"]
        / source_run_id
        / "pipeline.joblib"
    )

    pipeline = joblib.load(pipeline_path)

    with wandb.init(
        entity=entity,
        project=project,
        name=f"final_test_{source_run.name}",
        job_type="test",
        mode="online",
        config={
            "final_selection": selection,
            "source_training_config": source_config,
            "evaluation_split": "test",
            "test_sessions": len(X_test),
        },
    ) as run:

        metrics, predictions, _ = evaluate_model(
            pipeline,
            X_test,
            y_test,
            threshold=selection["threshold"],
        )

        metrics = {
            name: float(value)
            for name, value in metrics.items()
        }

        run.log({
            f"test/{name}": value
            for name, value in metrics.items()
        })

        output_dir = PROJECT_ROOT / source_config["output_dir"] / run.id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Menyimpan metrik di laptop.
        metrics_path = output_dir / "test_metrics.yaml"
        metrics_path.write_text(
            yaml.safe_dump(metrics, sort_keys=False),
            encoding="utf-8",
        )

        # Membuat confusion matrix khusus test.
        fig = create_confusion_matrix(y_test, predictions)
        fig.axes[0].set_title("Confusion Matrix — Test")

        figure_path = output_dir / "test_confusion_matrix.png"
        fig.savefig(figure_path, dpi=150)
        plt.close(fig)

        run.log({
            "test/confusion_matrix": wandb.Image(str(figure_path))
        })

        artifact = wandb.Artifact(
            name="final-test-evaluation",
            type="evaluation",
        )
        artifact.add_file(str(selection_path))
        artifact.add_file(str(metrics_path))
        artifact.add_file(str(figure_path))
        run.log_artifact(artifact)

        print("\nHasil evaluasi test:")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

        print("\nThreshold:", selection["threshold"])
        print("Jumlah sesi test:", len(X_test))
        print("Hasil disimpan:", output_dir)
        print("Link W&B:", run.url)

if __name__ == "__main__":
    main()