"""Script to train downstream audio classifiers (Logistic Regression & MLP) on ESC-50 embeddings."""
from pathlib import Path
import json
from ainak.training.audio_downstream import extract_esc50_embeddings, train_downstream_models

def main():
    dataset_dir = Path("datasets/ESC-50")
    panns_checkpoint = Path("artifacts/audio/Cnn14_mAP=0.431.pth")
    cache_path = Path("artifacts/audio/esc50_embeddings_cnn14.npz")
    output_checkpoint = Path("artifacts/audio/downstream_mlp.pt")
    output_report = Path("reports/audio-downstream-eval.json")

    embeddings, targets, folds, classes = extract_esc50_embeddings(
        dataset_dir=dataset_dir,
        panns_checkpoint=panns_checkpoint,
        cache_path=cache_path,
        device="cpu",
    )

    report = train_downstream_models(
        embeddings=embeddings,
        targets=targets,
        folds=folds,
        classes=classes,
        output_checkpoint=output_checkpoint,
        output_report=output_report,
        test_fold=5,
        epochs=35,
        batch_size=32,
        lr=1e-3,
        device="cpu",
        seed=42,
    )
    print("Downstream training summary:")
    print(f"  LR Accuracy:  {report['models_compared']['logistic_regression']['accuracy']:.4f}")
    print(f"  MLP Accuracy: {report['models_compared']['downstream_mlp']['accuracy']:.4f}")
    print(f"  Report: {output_report}")

if __name__ == "__main__":
    main()
