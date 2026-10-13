"""Fine-tune the validated CPZÃO model and compare it with the baseline."""

from datetime import datetime
from pathlib import Path
import json
import os
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "Matplotlib"))

import torch
from ultralytics import YOLO


DATA = ROOT / "data/dataset/data.yaml"
BASELINE = ROOT / "runs/detect/cpzao_v1/weights/best.pt"
PROJECT = ROOT / "runs/detect"
RUN_NAME = "cpzao_v2_adamw"
RUN_DIR = PROJECT / RUN_NAME
TEST_RUN_NAME = "cpzao_v2_test"
METRICS_DIR = ROOT / "docs/metricas"
BASELINE_METRICS = METRICS_DIR / "test_metrics.json"


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA indisponível; treino cancelado para evitar execução acidental em CPU.")
    if not DATA.is_file() or not BASELINE.is_file():
        raise FileNotFoundError(f"Dataset ou checkpoint ausente: {DATA} / {BASELINE}")
    if RUN_DIR.exists():
        raise FileExistsError(f"A pasta de execução já existe; escolha outro RUN_NAME: {RUN_DIR}")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    baseline = json.loads(BASELINE_METRICS.read_text(encoding="utf-8"))
    started_at = datetime.now().astimezone()
    device = torch.cuda.get_device_name(0)
    print(f"[CPZAO] Início: {started_at.isoformat(timespec='seconds')}", flush=True)
    print(f"[CPZAO] GPU: {device}; PyTorch {torch.__version__}; CUDA {torch.version.cuda}", flush=True)
    print(f"[CPZAO] Checkpoint inicial: {BASELINE}", flush=True)
    print("[CPZAO] Baseline test mAP50={:.4f}, mAP50-95={:.4f}".format(
        baseline["overall"]["map50"], baseline["overall"]["map50_95"]
    ), flush=True)

    model = YOLO(str(BASELINE))
    model.train(
        data=str(DATA),
        epochs=30,
        imgsz=640,
        batch=8,
        workers=0,
        amp=True,
        patience=8,
        cache=False,
        seed=42,
        device=0,
        optimizer="AdamW",
        project=str(PROJECT),
        name=RUN_NAME,
        plots=True,
        val=True,
        lr0=0.0005,
        lrf=0.01,
        warmup_epochs=1.0,
        close_mosaic=5,
    )
    training_seconds = time.perf_counter() - start
    best = RUN_DIR / "weights/best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"Novo best.pt não encontrado: {best}")

    print("[CPZAO] Treino concluído; avaliando o melhor checkpoint uma vez no split de teste.", flush=True)
    candidate = YOLO(str(best))
    metrics = candidate.val(
        data=str(DATA), split="test", plots=True, device=0,
        project=str(PROJECT), name=TEST_RUN_NAME, exist_ok=True,
    )
    box = metrics.box
    class_ids = [int(index) for index in box.ap_class_index]
    rows = []
    for position, class_id in enumerate(class_ids):
        rows.append({
            "id": class_id,
            "name": metrics.names[class_id],
            "precision": float(box.p[position]),
            "recall": float(box.r[position]),
            "map50": float(box.ap50[position]),
            "map50_95": float(box.ap[position].mean()),
        })
    result = {
        "started_at": started_at.isoformat(timespec="seconds"),
        "training_seconds": training_seconds,
        "device": device,
        "checkpoint": str(best.relative_to(ROOT)),
        "baseline": baseline["overall"],
        "candidate": {
            "map50": float(box.map50),
            "map50_95": float(box.map),
            "precision": float(box.mp),
            "recall": float(box.mr),
            "per_class": rows,
        },
    }
    report_path = METRICS_DIR / "finetune_comparison.json"
    report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n[CPZAO] Comparação de teste: baseline → fine-tune", flush=True)
    for metric in ("precision", "recall", "map50", "map50_95"):
        old, new = baseline["overall"][metric], result["candidate"][metric]
        print(f"  {metric}: {old:.4f} → {new:.4f} ({new-old:+.4f})", flush=True)
    print("\n[CPZAO] Métricas por classe do fine-tune (teste)", flush=True)
    for row in rows:
        print("  {name}: P={precision:.4f}, R={recall:.4f}, mAP50={map50:.4f}, mAP50-95={map50_95:.4f}".format(**row), flush=True)
    print(f"\n[CPZAO] Tempo: {training_seconds/60:.1f} min; GPU: {device}", flush=True)
    print(f"[CPZAO] Comparação salva em: {report_path}", flush=True)
    print("[CPZAO] Os pesos originais continuam intactos; revise a comparação antes de substituir/exportar modelos.", flush=True)


if __name__ == "__main__":
    start = time.perf_counter()
    main()
