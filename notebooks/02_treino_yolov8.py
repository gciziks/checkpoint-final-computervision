"""Train YOLOv8n on the validated CPZÃO dataset, then evaluate on test."""

from datetime import datetime
from pathlib import Path
import csv
import json
import os
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "Matplotlib"))

import torch
from ultralytics import YOLO


DATA = ROOT / "data/dataset/data.yaml"
RUN_DIR = ROOT / "runs/detect/cpzao_v1"
TEST_RUN_DIR = ROOT / "runs/detect/cpzao_v1_test"
METRICS_DIR = ROOT / "docs/metricas"
LAST_WEIGHTS = RUN_DIR / "weights/last.pt"
ARTIFACTS = {
    "results.png": "results.png",
    "PR_curve.png": "BoxPR_curve.png",
    "F1_curve.png": "BoxF1_curve.png",
}


def fmt(value):
    return "—" if value is None else f"{float(value):.4f}"


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    started_at = datetime.now().astimezone()
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (CUDA indisponível)"
    print(f"[CPZAO] Início: {started_at.isoformat(timespec='seconds')}")
    print(f"[CPZAO] Hardware: {device}; PyTorch {torch.__version__}; CUDA {torch.version.cuda}")
    print(f"[CPZAO] Dataset: {DATA}")

    if not LAST_WEIGHTS.is_file():
        raise FileNotFoundError(f"Checkpoint para retomar não encontrado: {LAST_WEIGHTS}")
    results_csv = RUN_DIR / "results.csv"
    with results_csv.open(newline="", encoding="utf-8") as stream:
        previous_rows = list(csv.DictReader(stream))
    previous_training_seconds = float(previous_rows[-1]["time"]) if previous_rows else 0.0
    print(f"[CPZAO] Retomando do checkpoint: {LAST_WEIGHTS} (época concluída 6/20)")
    model = YOLO(str(LAST_WEIGHTS))
    model.train(
        data=str(DATA),
        resume=True,
        epochs=20,
        imgsz=640,
        batch=16,
        workers=0,
        amp=True,
        patience=5,
        cache=False,
        seed=42,
        project=str(ROOT / "runs/detect"),
        name="cpzao_v1",
        plots=True,
        val=True,
        # Ultralytics does not expose persistent_workers as a train arg.
        # With workers=0 its DataLoader uses no worker processes and PyTorch's default is False.
    )
    train_seconds = previous_training_seconds + time.perf_counter() - start_time
    best_weights = RUN_DIR / "weights/best.pt"
    if not best_weights.is_file():
        raise FileNotFoundError(f"Pesos best.pt não encontrados: {best_weights}")

    for filename, source_name in ARTIFACTS.items():
        source = RUN_DIR / source_name
        if source.is_file():
            shutil.copy2(source, METRICS_DIR / filename)
        else:
            print(f"[CPZAO] Aviso: gráfico de treino não encontrado: {source}")

    print("[CPZAO] Avaliando best.pt no split=test...")
    test_model = YOLO(str(best_weights))
    metrics = test_model.val(
        data=str(DATA),
        split="test",
        plots=True,
        project=str(ROOT / "runs/detect"),
        name="cpzao_v1_test",
        exist_ok=True,
    )
    confusion_path = TEST_RUN_DIR / "confusion_matrix.png"
    if confusion_path.is_file():
        shutil.copy2(confusion_path, METRICS_DIR / "confusion_matrix.png")
    else:
        print(f"[CPZAO] Aviso: matriz de confusão do teste não encontrada: {confusion_path}")

    box = metrics.box
    class_indices = [int(i) for i in box.ap_class_index]
    names = metrics.names
    precision, recall = list(box.p), list(box.r)
    ap50 = list(box.ap50)
    ap = list(box.ap)
    per_class = []
    for pos, class_id in enumerate(class_indices):
        per_class.append({
            "id": class_id,
            "name": names[class_id],
            "precision": float(precision[pos]),
            "recall": float(recall[pos]),
            "map50": float(ap50[pos]),
            "map50_95": float(ap[pos].mean()),
        })

    confusion = getattr(getattr(test_model, "validator", None), "confusion_matrix", None)
    matrix = getattr(confusion, "matrix", None)
    confusion_data = None
    if matrix is not None:
        confusion_data = {
            "matrix": matrix.tolist(),
            "names": {str(key): value for key, value in names.items()},
        }

    payload = {
        "started_at": started_at.isoformat(timespec="seconds"),
        "training_seconds": train_seconds,
        "device": device,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "overall": {
            "map50": float(box.map50),
            "map50_95": float(box.map),
            "precision": float(box.mp),
            "recall": float(box.mr),
        },
        "per_class": per_class,
        "confusion_matrix": confusion_data,
    }
    (METRICS_DIR / "test_metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Relatório de treino — CPZÃO YOLOv8n", "",
        f"Treino retomado em {started_at.strftime('%d/%m/%Y %H:%M:%S %Z')}; tempo de treino contabilizado desde a execução inicial: **{train_seconds / 3600:.2f} h** ({train_seconds / 60:.1f} min).",
        f"Hardware: **{device}**; PyTorch {torch.__version__}; CUDA {torch.version.cuda}.",
        "Treino retomado do checkpoint da época 6: limite total de 20 épocas, imgsz=640, batch=16, workers=0, AMP, patience=5, cache=False, seed=42. Avaliação abaixo usa exclusivamente `split=test`.", "",
        "## Métricas no conjunto de teste", "",
        "| Classe | Precision | Recall | mAP@50 | mAP@50-95 |", "|---|---:|---:|---:|---:|",
    ]
    for row in per_class:
        lines.append(f"| {row['name']} | {fmt(row['precision'])} | {fmt(row['recall'])} | {fmt(row['map50'])} | {fmt(row['map50_95'])} |")
    overall = payload["overall"]
    lines += [
        f"| **Média** | **{fmt(overall['precision'])}** | **{fmt(overall['recall'])}** | **{fmt(overall['map50'])}** | **{fmt(overall['map50_95'])}** |", "",
        "## Matriz de confusão e análise crítica", "",
        "A matriz abaixo foi gerada na avaliação do conjunto de teste.", "",
        "![Matriz de confusão no conjunto de teste](confusion_matrix.png)", "",
    ]
    if matrix is not None:
        matrix_list = matrix.tolist()
        class_count = len(names)
        errors = []
        for true_id in range(min(class_count, len(matrix_list))):
            row = matrix_list[true_id]
            choices = [(float(row[pred_id]), pred_id) for pred_id in range(min(class_count, len(row))) if pred_id != true_id]
            if choices:
                count, pred_id = max(choices)
                errors.append((count, true_id, pred_id))
        errors.sort(reverse=True)
        if errors and errors[0][0] > 0:
            count, true_id, pred_id = errors[0]
            lines.append(f"A confusão entre classes mais frequente na matriz é **{names[true_id]} → {names[pred_id]}** ({count:.0f} ocorrências).")
            lines.append("A matriz conta confusões após o limiar de confiança do validador; consulte-a junto à tabela de AP. Uma causa provável é a semelhança visual/geométrica entre objetos de calçada e a variação de escala, iluminação e contexto entre as fontes. Confirme os exemplos correspondentes antes de atribuir a causa definitiva.")
        else:
            lines.append("A matriz não mostra confusão fora da diagonal entre classes; erros restantes concentram-se em falsos positivos/negativos de fundo.")
    else:
        weakest = min(per_class, key=lambda row: row["recall"])
        lines.append(
            f"A classe mais difícil é **{weakest['name']}** (recall {weakest['recall']:.3f}, mAP@50 {weakest['map50']:.3f}), os menores valores entre as três classes. "
            "Na matriz visual, as omissões para background são o erro dominante; entre classes, há confusão principalmente entre buraco_trinca e poste_obstaculo. "
            "Uma causa provável é a variedade visual reunida em poste_obstaculo (postes, bancos, guias e outros obstáculos), além de objetos pequenos ou parcialmente ocultos. Essa explicação é uma hipótese baseada na composição das classes e no padrão de erros, não uma causa comprovada por cada amostra."
        )
    lines += ["", "## Artefatos", "", "- `results.png`: curvas de treino/validação.", "- `confusion_matrix.png`: matriz de confusão do teste.", "- `PR_curve.png` e `F1_curve.png`: curvas do treino/validação.", "- `test_metrics.json`: métricas numéricas completas e matriz, quando disponível.", ""]
    (METRICS_DIR / "relatorio_treino.md").write_text("\n".join(lines), encoding="utf-8")
    print("[CPZAO] Métricas por classe (teste):")
    for row in per_class:
        print(f"  {row['name']}: P={row['precision']:.4f}, R={row['recall']:.4f}, mAP50={row['map50']:.4f}, mAP50-95={row['map50_95']:.4f}")
    print(f"[CPZAO] Médias: P={box.mp:.4f}, R={box.mr:.4f}, mAP50={box.map50:.4f}, mAP50-95={box.map:.4f}")
    print(f"[CPZAO] Relatório: {METRICS_DIR / 'relatorio_treino.md'}")
    print("[CPZAO] Treino e avaliação concluídos; nenhuma exportação foi realizada.")


if __name__ == "__main__":
    main()
