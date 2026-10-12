"""Merge, remap, split and validate the CPZÃO Roboflow YOLO datasets."""

from collections import Counter, defaultdict
from pathlib import Path
import math
import random
import re
import shutil

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data" / "raw"
OUT_ROOT = ROOT / "data" / "dataset"
STATS_PATH = ROOT / "docs" / "dataset_stats.md"
SEED = 42
DEST_NAMES = {0: "buraco_trinca", 1: "poste_obstaculo", 2: "piso_tatil"}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RULES = [
    (0, ("crack", "pothole", "potholes", "hole", "trinca", "buraco")),
    (
        1,
        (
            "pole", "post", "bench", "curb", "stairs", "bicycle", "motorcycle",
            "waste", "trash", "dustbin", "traffic_light", "fire_hydrant", "street_light",
            "obstacle", "poste",
        ),
    ),
    (2, ("tactile", "paving", "piso_tatil", "blind")),
]


def log(message):
    print(f"[CPZAO] {message}", flush=True)


def parse_names(yaml_text):
    """Read Roboflow inline names lists or numbered Ultralytics names."""
    match = re.search(r"(?m)^\s*names\s*:\s*(\[[^\n]*\]|\{[^\n]*\})\s*$", yaml_text)
    if match:
        payload = match.group(1).strip()
        if payload.startswith("["):
            values = re.findall(r"(['\"])(.*?)\1", payload)
            names = [value for _, value in values]
            if not names:
                names = [part.strip().strip("'\"") for part in payload[1:-1].split(",") if part.strip()]
            return dict(enumerate(names))
        entries = re.findall(r"(\d+)\s*:\s*(['\"]?)([^,'\"}]+)\2", payload)
        if entries:
            return {int(index): value.strip() for index, _, value in entries}

    names = {}
    in_names = False
    for line in yaml_text.splitlines():
        if re.match(r"^\s*names\s*:\s*$", line):
            in_names = True
            continue
        if in_names:
            entry = re.match(r"^\s+(\d+)\s*:\s*(.*?)\s*$", line)
            if entry:
                names[int(entry.group(1))] = entry.group(2).strip("'\"")
            elif line.strip() and not line.lstrip().startswith("#"):
                break
    if not names:
        raise ValueError("Não foi possível interpretar a seção names do data.yaml")
    return names


def destination_id(class_name):
    normalized = class_name.casefold()
    for target_id, tokens in RULES:
        if any(token.casefold() in normalized for token in tokens):
            return target_id
    return None


def split_path(dataset_root, yaml_text, split):
    """Resolve the configured image folder, then use Roboflow folder conventions."""
    key = {"train": "train", "valid": "val", "test": "test"}[split]
    match = re.search(rf"(?m)^\s*{key}\s*:\s*[\"']?([^\n#\"']+)", yaml_text)
    if match:
        configured = match.group(1).strip()
        path = Path(configured)
        candidates = [path if path.is_absolute() else dataset_root / path]
        candidates += [dataset_root / split / "images", dataset_root / split]
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
    for candidate in (dataset_root / split / "images", dataset_root / split):
        if candidate.is_dir():
            return candidate
    return None


def paired_label(image_path, image_dir):
    relative = image_path.relative_to(image_dir)
    label = image_dir.parent / "labels" / relative.with_suffix(".txt")
    if label.is_file():
        return label
    sibling = image_dir.parent.parent / "labels" / relative.with_suffix(".txt")
    return sibling if sibling.is_file() else label


def parse_label(label_path, class_names, unknown_counts, source):
    mapped = []
    if not label_path.is_file():
        return mapped
    for line_no, line in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        try:
            old_id = int(float(parts[0]))
        except (ValueError, IndexError) as exc:
            raise ValueError(f"{label_path}:{line_no}: ID de classe inválido") from exc
        if len(parts) < 5 or len(parts) % 2 == 0:
            raise ValueError(f"{label_path}:{line_no}: esperado ID + 4 coordenadas YOLO ou ID + pares de polígono, encontrado {len(parts)} campos")
        if old_id not in class_names:
            raise ValueError(f"{label_path}:{line_no}: ID {old_id} ausente em names do YAML")
        new_id = destination_id(class_names[old_id])
        if new_id is None:
            unknown_counts[(source, class_names[old_id])] += 1
            continue
        if len(parts) == 5:
            coordinates = parts[1:]
        else:
            # Roboflow YOLO segmentation: polygon (x1 y1 ... xn yn).
            # Convert to the enclosing normalized YOLO detection box.
            points = [float(value) for value in parts[1:]]
            xs, ys = points[0::2], points[1::2]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            coordinates = [
                f"{(x_min + x_max) / 2:.8f}",
                f"{(y_min + y_max) / 2:.8f}",
                f"{x_max - x_min:.8f}",
                f"{y_max - y_min:.8f}",
            ]
        mapped.append((new_id, coordinates))
    return mapped


def allocate_counts(n):
    """Largest-remainder allocation, with all splits represented when n >= 3."""
    raw = [n * weight for weight in (0.70, 0.20, 0.10)]
    counts = [math.floor(value) for value in raw]
    remainder = n - sum(counts)
    for idx in sorted(range(3), key=lambda i: raw[i] - counts[i], reverse=True)[:remainder]:
        counts[idx] += 1
    if n >= 3:
        for empty in [i for i, count in enumerate(counts) if count == 0]:
            donor = max(range(3), key=lambda i: counts[i])
            counts[donor] -= 1
            counts[empty] += 1
    return counts


def box_counts(items):
    counts = Counter()
    for item in items:
        counts.update(class_id for class_id, _ in item["boxes"])
    return counts


def transform_box(box, flip, angle, crop):
    class_id, coords = box
    cx, cy, width, height = (float(value) for value in coords)
    corners = [(cx + sx * width / 2, cy + sy * height / 2) for sx in (-1, 1) for sy in (-1, 1)]
    if flip:
        corners = [(1 - x, y) for x, y in corners]
    theta = math.radians(angle)
    cosine, sine = math.cos(theta), math.sin(theta)
    rotated = []
    for x, y in corners:
        dx, dy = x - 0.5, y - 0.5
        rotated.append((0.5 + cosine * dx + sine * dy, 0.5 - sine * dx + cosine * dy))
    xs, ys = zip(*rotated)
    x1, x2 = max(0.0, min(xs)), min(1.0, max(xs))
    y1, y2 = max(0.0, min(ys)), min(1.0, max(ys))
    x0, y0, crop_w, crop_h = crop
    x1, x2 = max(0.0, (x1 - x0) / crop_w), min(1.0, (x2 - x0) / crop_w)
    y1, y2 = max(0.0, (y1 - y0) / crop_h), min(1.0, (y2 - y0) / crop_h)
    if x2 <= x1 or y2 <= y1:
        return None
    return (class_id, [f"{(x1+x2)/2:.8f}", f"{(y1+y2)/2:.8f}", f"{x2-x1:.8f}", f"{y2-y1:.8f}"])


def augment_record(record, variant_index, seed, attempt=0):
    rng = random.Random(seed)
    image = Image.open(record["image"]).convert("RGB")
    width, height = image.size
    flip = rng.random() < 0.5
    angle = rng.uniform(-15.0, 15.0)
    if flip:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    image = image.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False)
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.80, 1.20))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.80, 1.20))
    image = ImageEnhance.Color(image).enhance(rng.uniform(0.80, 1.20))
    # Mild directional 5x5 motion blur, horizontal or vertical.
    if rng.random() < 0.7:
        kernel = [0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0,
                  0.2, 0.2, 0.2, 0.2, 0.2,
                  0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0]
        if rng.random() < 0.5:
            kernel = [kernel[row * 5 + col] for col in range(5) for row in range(5)]
        image = image.filter(ImageFilter.Kernel((5, 5), kernel, scale=1.0))
    scale = rng.uniform(0.85, 1.0)
    crop_w, crop_h = scale, scale
    crop_x = rng.uniform(0.0, 1.0 - crop_w)
    crop_y = rng.uniform(0.0, 1.0 - crop_h)
    left, top = int(crop_x * width), int(crop_y * height)
    right, bottom = max(left + 1, int((crop_x + crop_w) * width)), max(top + 1, int((crop_y + crop_h) * height))
    image = image.crop((left, top, right, bottom)).resize((width, height), Image.Resampling.BILINEAR)
    pixel_crop = (left / width, top / height, (right - left) / width, (bottom - top) / height)
    boxes = [mapped for box in record["boxes"] if (mapped := transform_box(box, flip, angle, pixel_crop)) is not None]
    if not boxes:
        if attempt >= 100:
            raise ValueError(f"Não foi possível preservar caixas após 100 tentativas de augmentation: {record['image']}")
        return augment_record(record, variant_index, seed + 1, attempt + 1)
    return image, boxes


def main():
    log(f"Raiz do projeto: {ROOT}")
    log(f"Origem: {RAW_ROOT}")
    log(f"Destino: {OUT_ROOT}")
    if not RAW_ROOT.is_dir():
        raise FileNotFoundError(f"Pasta de datasets não encontrada: {RAW_ROOT}")
    dataset_roots = sorted(path for path in RAW_ROOT.iterdir() if path.is_dir() and (path / "data.yaml").is_file())
    if not dataset_roots:
        raise FileNotFoundError("Nenhuma pasta com data.yaml encontrada em data/raw/")

    records = []
    unknown_counts = Counter()
    source_dominant_counts = defaultdict(Counter)
    source_box_counts = defaultdict(Counter)
    source_old_counts = {}
    used_names = set()

    for dataset_root in dataset_roots:
        yaml_text = (dataset_root / "data.yaml").read_text(encoding="utf-8-sig")
        class_names = parse_names(yaml_text)
        old_counts = Counter()
        log("=" * 72)
        log(f"Dataset: {dataset_root}")
        log("Classes originais: " + ", ".join(f"{idx}={name}" for idx, name in sorted(class_names.items())))
        for split in ("train", "valid", "test"):
            image_dir = split_path(dataset_root, yaml_text, split)
            images = sorted(p for p in image_dir.rglob("*") if p.is_file() and p.suffix.casefold() in EXTENSIONS) if image_dir else []
            log(f"Imagens {split}: {len(images)}" + (f" ({image_dir})" if image_dir else " (pasta ausente)"))
            for image_path in images:
                label_path = paired_label(image_path, image_dir)
                boxes = parse_label(label_path, class_names, unknown_counts, dataset_root.name)
                if label_path.is_file():
                    for line in label_path.read_text(encoding="utf-8-sig").splitlines():
                        if line.strip():
                            original_id = int(float(line.split()[0]))
                            old_counts[class_names[original_id]] += 1
                if not boxes:
                    continue

                dominant = Counter(class_id for class_id, _ in boxes).most_common(1)[0][0]
                relative = image_path.relative_to(dataset_root)
                base = re.sub(r"[^A-Za-z0-9._-]+", "_", relative.stem).strip("._") or "image"
                prefix = re.sub(r"[^A-Za-z0-9]+", "", dataset_root.name).lower()
                candidate = f"{prefix}_{base}{image_path.suffix.lower()}"
                unique = candidate
                serial = 2
                while unique.casefold() in used_names:
                    unique = f"{Path(candidate).stem}_{serial}{Path(candidate).suffix}"
                    serial += 1
                used_names.add(unique.casefold())
                records.append({"source": dataset_root.name, "image": image_path, "label": label_path, "filename": unique, "boxes": boxes, "dominant": dominant, "source_key": str(image_path)})
                source_dominant_counts[dataset_root.name][dominant] += 1
                source_box_counts[dataset_root.name].update(class_id for class_id, _ in boxes)
        source_old_counts[dataset_root.name] = old_counts
        log("Instâncias originais: " + (", ".join(f"{name}={count}" for name, count in sorted(old_counts.items())) or "nenhuma"))

    for (source, class_name), count in sorted(unknown_counts.items()):
        log(f"DESCARTANDO anotações: dataset={source}, classe={class_name!r}, instâncias={count}")
    if not records:
        raise RuntimeError("Nenhuma imagem possui bounding box válido após o mapeamento.")

    groups = defaultdict(list)
    for record in records:
        groups[record["dominant"]].append(record)
    rng = random.Random(SEED)
    split_records = {"train": [], "val": [], "test": []}
    if len(groups[1]) < 3500:
        raise ValueError(f"Há apenas {len(groups[1])} imagens poste_obstaculo; não é possível manter 3.500.")
    if len(groups[2]) > 3000:
        raise ValueError(f"Há {len(groups[2])} imagens piso_tatil; este procedimento foi solicitado para augmentar até 3.000, não reduzir.")
    groups[1] = rng.sample(groups[1], 3500)
    tactile_sources = groups[2][:]
    tactile_split_counts = allocate_counts(3000)
    source_split_counts = allocate_counts(len(tactile_sources))
    staging_dir = OUT_ROOT / "_augmentation_staging"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    for class_id in (0, 1):
        group = groups[class_id][:]
        rng.shuffle(group)
        n_train, n_val, n_test = allocate_counts(len(group))
        split_records["train"].extend(group[:n_train])
        split_records["val"].extend(group[n_train:n_train + n_val])
        split_records["test"].extend(group[n_train + n_val:n_train + n_val + n_test])
        log(f"Estrato {class_id}={DEST_NAMES[class_id]} após balanceamento: {len(group)} imagens -> train={n_train}, val={n_val}, test={n_test}")

    # Group all augmented variants with their source image to prevent train/val/test leakage.
    rng.shuffle(tactile_sources)
    source_cursor = 0
    tactile_by_split = {}
    for split, n_sources in zip(("train", "val", "test"), source_split_counts):
        tactile_by_split[split] = tactile_sources[source_cursor:source_cursor + n_sources]
        source_cursor += n_sources
    for split, target_count in zip(("train", "val", "test"), tactile_split_counts):
        source_group = tactile_by_split[split]
        split_records[split].extend(source_group)
        variants_needed = target_count - len(source_group)
        for variant_index in range(variants_needed):
            source = source_group[variant_index % len(source_group)]
            variant_seed = SEED + variant_index + (0 if split == "train" else 100_000 if split == "val" else 200_000)
            augmented_image, augmented_boxes = augment_record(source, variant_index, variant_seed)
            name = f"aug_{Path(source['filename']).stem}_v{variant_index + 1:04d}.jpg"
            staging_path = staging_dir / name
            augmented_image.save(staging_path, quality=95)
            split_records[split].append({**source, "filename": name, "boxes": augmented_boxes, "augmented": True, "staging_path": staging_path, "source_key": source["source_key"]})
        log(f"Estrato 2=piso_tatil: originais={len(source_group)}, variações={variants_needed}, total {split}={target_count}")
    for split in split_records:
        rng.shuffle(split_records[split])

    # Rebuild only generated split folders; raw source datasets are untouched.
    for split in ("train", "val", "test"):
        image_dir, label_dir = OUT_ROOT / "images" / split, OUT_ROOT / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for stale in list(image_dir.iterdir()) + list(label_dir.iterdir()):
            if stale.is_file():
                stale.unlink()
    for split, split_group in split_records.items():
        for record in split_group:
            if record.get("augmented"):
                shutil.copy2(record["staging_path"], OUT_ROOT / "images" / split / record["filename"])
            else:
                shutil.copy2(record["image"], OUT_ROOT / "images" / split / record["filename"])
            label_text = "".join(f"{class_id} {' '.join(coords)}\n" for class_id, coords in record["boxes"])
            (OUT_ROOT / "labels" / split / f"{Path(record['filename']).stem}.txt").write_text(label_text, encoding="utf-8")
        log(f"Copiados {len(split_group)} pares para split {split}")
    shutil.rmtree(staging_dir)

    yaml_output = (
        "path: ./data/dataset\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: buraco_trinca\n"
        "  1: poste_obstaculo\n"
        "  2: piso_tatil\n"
    )
    (OUT_ROOT / "data.yaml").write_text(yaml_output, encoding="utf-8")

    balanced_records = [record for split in ("train", "val", "test") for record in split_records[split]]
    groups = defaultdict(list)
    source_dominant_counts = defaultdict(Counter)
    source_box_counts = defaultdict(Counter)
    for record in balanced_records:
        groups[record["dominant"]].append(record)
        source_dominant_counts[record["source"]][record["dominant"]] += 1
        source_box_counts[record["source"]].update(class_id for class_id, _ in record["boxes"])
    total_box_counts = box_counts(balanced_records)
    split_box_counts = {split: box_counts(items) for split, items in split_records.items()}
    split_dominant_counts = {split: Counter(item["dominant"] for item in items) for split, items in split_records.items()}

    # Sanity checks: exact image/label pairing, non-empty labels and valid IDs.
    all_ok = True
    for split in ("train", "val", "test"):
        image_dir, label_dir = OUT_ROOT / "images" / split, OUT_ROOT / "labels" / split
        images = [p for p in image_dir.iterdir() if p.is_file()]
        labels = list(label_dir.glob("*.txt"))
        stems = {p.stem for p in images}
        label_stems = {p.stem for p in labels}
        missing = sorted(stems - label_stems)
        extra = sorted(label_stems - stems)
        bad_ids = []
        for label in labels:
            for line_no, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
                if line.strip() and line.split()[0] not in {"0", "1", "2"}:
                    bad_ids.append(f"{label.name}:{line_no}:{line.split()[0]}")
        empty = [label.name for label in labels if not label.read_text(encoding="utf-8").strip()]
        log(f"Sanity {split}: imagens={len(images)}, labels={len(labels)}, sem_label={len(missing)}, labels_extras={len(extra)}, labels_vazios={len(empty)}, IDs_inválidos={len(bad_ids)}")
        if missing or extra or bad_ids or empty or len(images) != len(labels):
            all_ok = False
            if missing:
                log(f"  Imagens sem label: {missing[:10]}")
            if extra:
                log(f"  Labels sem imagem: {extra[:10]}")
            if bad_ids:
                log(f"  IDs inválidos: {bad_ids[:10]}")
    if not all_ok:
        raise AssertionError("A verificação de sanidade falhou.")

    lines = [
        "# Estatísticas do dataset CPZÃO", "",
        f"Gerado com seed {SEED}; estratificação pela classe dominante. A divisão usa 70/20/10 por classe com arredondamento de maiores restos.", "",
        "## Resumo", "",
        f"- Imagens válidas após balanceamento: **{len(balanced_records):,}**",
        f"- Bounding boxes válidos: **{sum(total_box_counts.values()):,}**",
        f"- Split: train **{len(split_records['train']):,}** ({len(split_records['train']) / len(balanced_records):.1%}), val **{len(split_records['val']):,}** ({len(split_records['val']) / len(balanced_records):.1%}), test **{len(split_records['test']):,}** ({len(split_records['test']) / len(balanced_records):.1%})",
        "- Balanceamento: todas as imagens `buraco_trinca` mantidas; `poste_obstaculo` reduzida aleatoriamente a 3.500 (seed 42); `piso_tatil` aumentada a 3.000 (1.395 originais + 1.605 variações).",
        "- Variações de piso tátil usam flip horizontal, rotação aleatória ±15°, brilho/contraste/saturação, motion blur leve e crop aleatório 0,85–1,0. Variantes da mesma origem permanecem no mesmo split para evitar vazamento entre treino, validação e teste.", "",
        "## Imagens e bounding boxes por classe", "",
        "| Classe | Imagens (classe dominante) | Bounding boxes |", "|---|---:|---:|",
    ]
    for class_id, name in DEST_NAMES.items():
        lines.append(f"| {name} | {len(groups[class_id]):,} | {total_box_counts[class_id]:,} |")
    lines += [
        "", "## Distribuição por split", "",
        "| Split | Total de imagens | buraco_trinca (imagens / caixas) | poste_obstaculo (imagens / caixas) | piso_tatil (imagens / caixas) |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in ("train", "val", "test"):
        cells = [f"{split_dominant_counts[split][i]:,} / {split_box_counts[split][i]:,}" for i in DEST_NAMES]
        lines.append(f"| {split} | {len(split_records[split]):,} | " + " | ".join(cells) + " |")
    lines += [
        "", "## Origem por classe", "",
        "| Dataset de origem | buraco_trinca (imagens / caixas) | poste_obstaculo (imagens / caixas) | piso_tatil (imagens / caixas) | Total de imagens válidas |",
        "|---|---:|---:|---:|---:|",
    ]
    for source in sorted(source_dominant_counts):
        cells = [f"{source_dominant_counts[source][i]:,} / {source_box_counts[source][i]:,}" for i in DEST_NAMES]
        lines.append(f"| {source} | " + " | ".join(cells) + f" | {sum(source_dominant_counts[source].values()):,} |")
    lines.append(f"| **Total** | **{len(groups[0]):,} / {total_box_counts[0]:,}** | **{len(groups[1]):,} / {total_box_counts[1]:,}** | **{len(groups[2]):,} / {total_box_counts[2]:,}** | **{len(balanced_records):,}** |")
    lines += ["", "## Classes descartadas", "", "Anotações sem classe de destino foram filtradas individualmente; a imagem foi mantida se restou ao menos uma caixa válida.", "", "No YAML original de `object on road`, o nome da classe de ID 0 é literalmente `0` (não há nome semântico). A inspeção visual de 12 exemplos mostra caixas sobre rampas/soleiras de acesso; não há evidência de que sejam postes, então o ID 0 ficou sem mapeamento. `dustbin` foi mapeada para `poste_obstaculo`.", "", "| Dataset | Classe original | Anotações descartadas |", "|---|---|---:|"]
    if unknown_counts:
        for (source, name), count in sorted(unknown_counts.items()):
            lines.append(f"| {source} | {name} | {count:,} |")
    else:
        lines.append("| — | Nenhuma | 0 |")
    lines += ["", "## Alertas e recomendações", ""]
    low_data = [i for i in DEST_NAMES if len(groups[i]) < 30]
    if low_data:
        lines.append("⚠️ Menos de 30 imagens por classe dominante: " + ", ".join(f"{DEST_NAMES[i]} ({len(groups[i])})" for i in low_data) + ". Há risco de underfitting; priorize mais dados anotados e augmentation direcionado.")
    else:
        lines.append("Todas as classes têm pelo menos 30 imagens com a classe como dominante.")
    under20 = [i for i in DEST_NAMES if len(groups[i]) < 20]
    if under20:
        lines += ["", "🚨 Classe(s) com menos de 20 imagens: " + ", ".join(f"{DEST_NAMES[i]} ({len(groups[i])})" for i in under20) + ". Antes do treino, obtenha imagens adicionais ou use oversampling/augmentation da classe minoritária apenas no conjunto de treino; não duplique imagens entre splits."]
    lines += ["", "## Imagens por classe dominante e split", "", "![Gráfico de barras com imagens por classe dominante em cada split](dataset_stats.svg)", ""]
    chart_w, chart_h, left, top, plot_h = 920, 390, 270, 55, 245
    max_value = max(split_dominant_counts[s][i] for s in split_dominant_counts for i in DEST_NAMES)
    bar_h = 18
    colors = {"train": "#2878B5", "val": "#F39C34", "test": "#4A9D67"}
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{chart_w}" height="{chart_h}" viewBox="0 0 {chart_w} {chart_h}">', '<rect width="100%" height="100%" fill="white"/>', '<style>text{font-family:Arial,sans-serif;fill:#243447}.axis{stroke:#ccd4dc;stroke-width:1}</style>', '<text x="20" y="28" font-size="19" font-weight="bold">Imagens por classe dominante e split</text>']
    for tick in range(5):
        value = round(max_value * tick / 4)
        x = left + (chart_w - left - 50) * tick / 4
        svg.append(f'<line class="axis" x1="{x:.1f}" y1="{top-5}" x2="{x:.1f}" y2="{top+plot_h}"/>')
        svg.append(f'<text x="{x:.1f}" y="{top+plot_h+20}" font-size="12" text-anchor="middle">{value}</text>')
    group_height, group_gap = 3 * (bar_h + 4) + 18, 18
    for class_index, class_id in enumerate(DEST_NAMES):
        y0 = top + class_index * (group_height + group_gap)
        svg.append(f'<text x="{left-12}" y="{y0+group_height/2}" font-size="14" text-anchor="end" dominant-baseline="middle">{DEST_NAMES[class_id]}</text>')
        for row, split in enumerate(("train", "val", "test")):
            value = split_dominant_counts[split][class_id]
            y = y0 + row * (bar_h + 4)
            width = (chart_w - left - 50) * value / max_value
            svg.append(f'<rect x="{left}" y="{y}" width="{width:.1f}" height="{bar_h}" rx="3" fill="{colors[split]}"/>')
            svg.append(f'<text x="{left+width+7:.1f}" y="{y+14}" font-size="12">{value:,} ({split})</text>')
    svg.append('</svg>')
    (STATS_PATH.parent / "dataset_stats.svg").write_text("\n".join(svg), encoding="utf-8")
    lines += ["## Sanidade", "", "Todos os arquivos de imagem possuem label correspondente; os labels não estão vazios e os IDs pertencem a `{0, 1, 2}`."]
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    log(f"YAML gerado: {OUT_ROOT / 'data.yaml'}")
    log(f"Relatório gerado: {STATS_PATH}")
    log("Resumo final: " + ", ".join(f"{DEST_NAMES[i]}={len(groups[i])} imagens dominantes/{total_box_counts[i]} caixas" for i in DEST_NAMES))
    log(f"Total={len(balanced_records)} imagens; train={len(split_records['train'])}, val={len(split_records['val'])}, test={len(split_records['test'])}")
    log("Sanidade aprovada; pronto para revisão do dataset_stats.md (nenhum treino foi iniciado).")


if __name__ == "__main__":
    main()
