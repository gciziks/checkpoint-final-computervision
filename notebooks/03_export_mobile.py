"""Export trained YOLOv8n weights to TFLite and ONNX, then smoke-test TFLite."""

from pathlib import Path
import os
import shutil

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "Ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "Matplotlib"))

from ultralytics import YOLO


WEIGHTS = ROOT / "runs/detect/cpzao_v1/weights/best.pt"
TEST_IMAGES = ROOT / "data/dataset/images/test"
MODELS = ROOT / "models"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def locate_export(returned_path, suffix):
    """Resolve the exported artifact across Ultralytics versions/platforms."""
    candidate = Path(returned_path) if returned_path else None
    if candidate and candidate.is_file() and candidate.suffix.lower() == suffix:
        return candidate
    search_roots = [candidate] if candidate and candidate.is_dir() else []
    search_roots.extend([WEIGHTS.parent, WEIGHTS.parent.parent])
    matches = []
    for root in search_roots:
        if root.exists():
            matches.extend(root.rglob(f"*{suffix}"))
    if not matches:
        raise FileNotFoundError(f"Ultralytics não gerou o arquivo {suffix} esperado.")
    return max(matches, key=lambda path: path.stat().st_mtime)


def main():
    if not WEIGHTS.is_file():
        raise FileNotFoundError(f"Pesos treinados não encontrados: {WEIGHTS}")
    images = sorted(p for p in TEST_IMAGES.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError(f"Nenhuma imagem de teste em {TEST_IMAGES}")
    MODELS.mkdir(parents=True, exist_ok=True)

    print(f"[EXPORT] Pesos de origem: {WEIGHTS}")
    model = YOLO(str(WEIGHTS))

    tflite_target = MODELS / "best.tflite"
    print("[EXPORT] Exportando TFLite (imgsz=640)...")
    try:
        tflite_result = model.export(format="tflite", imgsz=640)
        tflite_source = locate_export(tflite_result, ".tflite")
        shutil.copy2(tflite_source, tflite_target)
        print(f"[EXPORT] TFLite: {tflite_source} -> {tflite_target}")
    except AssertionError as exc:
        if "only supported on Linux x86 and macOS" not in str(exc):
            raise
        print("[EXPORT] LiteRT direto não suporta Windows nesta versão. Tentando exportação TensorFlow SavedModel, que também produz TFLite...")
        try:
            saved_model = model.export(format="saved_model", imgsz=640)
        except Exception as saved_model_error:
            # Some Ultralytics Windows builds emit valid TFLite files, then fail
            # their post-export validation because the SavedModel path was removed.
            candidates = list(WEIGHTS.parent.rglob("*.tflite"))
            if not candidates:
                raise saved_model_error
            saved_model = max(candidates, key=lambda path: path.stat().st_mtime).parent
            print(f"[EXPORT] Validação interna falhou após gerar TFLite ({saved_model_error}); usando arquivo exportado em disco.")
        tflite_source = locate_export(saved_model, ".tflite")
        shutil.copy2(tflite_source, tflite_target)
        print(f"[EXPORT] TFLite via SavedModel: {tflite_source} -> {tflite_target}")

    print("[EXPORT] Exportando ONNX (imgsz=640)...")
    onnx_result = YOLO(str(WEIGHTS)).export(format="onnx", imgsz=640)
    onnx_source = locate_export(onnx_result, ".onnx")
    onnx_target = MODELS / "best.onnx"
    shutil.copy2(onnx_source, onnx_target)
    print(f"[EXPORT] ONNX: {onnx_source} -> {onnx_target}")

    if tflite_target:
        print(f"[EXPORT] Inferência smoke test TFLite: {images[0]}")
        import cv2
        import numpy as np
        import tensorflow as tf

        # model_content avoids a Windows LiteRT path issue with accented OneDrive paths.
        interpreter = tf.lite.Interpreter(model_content=tflite_target.read_bytes())
        interpreter.allocate_tensors()
        input_detail = interpreter.get_input_details()[0]
        output_detail = interpreter.get_output_details()[0]
        frame = cv2.imread(str(images[0]))
        if frame is None:
            raise RuntimeError(f"Não foi possível ler a imagem de teste: {images[0]}")
        tensor = cv2.cvtColor(cv2.resize(frame, (640, 640)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        interpreter.set_tensor(input_detail["index"], tensor[None, ...])
        interpreter.invoke()
        output = interpreter.get_tensor(output_detail["index"])
        if not np.isfinite(output).all():
            raise RuntimeError("A saída TFLite contém valores não finitos.")
        print(f"[EXPORT] TFLite inferiu sem erro: input={input_detail['shape']}; output={output.shape}; imagem={images[0].name}")
    else:
        print(f"[EXPORT] Inferência smoke test ONNX (TFLite indisponível): {images[0]}")
        inference_model = YOLO(str(onnx_target), task="detect")
        predictions = inference_model.predict(source=str(images[0]), imgsz=640, device="cpu", verbose=False)
        if len(predictions) != 1:
            raise RuntimeError(f"Esperada 1 predição, recebidas {len(predictions)}")
        print(f"[EXPORT] ONNX inferiu sem erro: {len(predictions[0].boxes)} detecção(ões) na imagem de teste.")
    print(f"[EXPORT] Concluído. TFLite={tflite_target or 'não gerado nesta plataforma'}; ONNX={onnx_target}")


if __name__ == "__main__":
    main()
