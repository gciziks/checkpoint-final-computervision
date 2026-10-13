# Checkpoint Visão Computacional — Auditoria de Calçadas

Pipeline de visão computacional para detectar problemas e elementos de acessibilidade em calçadas urbanas, com YOLOv8n e inferência móvel.

O objetivo é ajudar a identificar defeitos e obstáculos que dificultam a circulação a pé, além de reconhecer piso tátil. O protótipo classifica imagens em três categorias; ele não substitui uma vistoria técnica.

## Integrantes

| RM | Nome |
|---|---|
| RM551059 | Cassio Valezzi |
| RM98215 | Gabriel Antony Cadima Ciziks |
| RM98169 | Lucca Sabatini Tambellini |
| RM98209 | Victor Nuzzi |
| RM551272 | João Arthur |

## Classes

| ID | Classe | Descrição |
|---:|---|---|
| 0 | `buraco_trinca` | Buracos e trincas no piso. |
| 1 | `poste_obstaculo` | Postes e outros obstáculos urbanos, como bancos, guias e lixeiras. |
| 2 | `piso_tatil` | Piso tátil e elementos de acessibilidade. |

## Métricas no conjunto de teste

YOLOv8n (`yolov8n.pt`), treinado por até 20 épocas e avaliado no split `test`.

| Classe | Precision | Recall | mAP@50 | mAP@50–95 |
|---|---:|---:|---:|---:|
| `buraco_trinca` | 0,7464 | 0,6176 | 0,6893 | 0,3992 |
| `poste_obstaculo` | 0,7614 | 0,4621 | 0,5705 | 0,3638 |
| `piso_tatil` | 0,7894 | 0,7799 | 0,8128 | 0,5693 |
| **Média** | **0,7657** | **0,6199** | **0,6909** | **0,4441** |

### Fine-tuning do modelo

O checkpoint inicial foi continuado com AdamW a partir do melhor peso anterior. O treino foi encerrado após a época 24, e o checkpoint resultante foi avaliado no mesmo conjunto de teste:

| Classe | Precision | Recall | mAP@50 | mAP@50–95 |
|---|---:|---:|---:|---:|
| `buraco_trinca` | 0,8461 | 0,6049 | 0,7154 | 0,4241 |
| `poste_obstaculo` | 0,7869 | 0,4802 | 0,5896 | 0,3762 |
| `piso_tatil` | 0,8435 | 0,7786 | 0,8395 | 0,6054 |
| **Média** | **0,8255** | **0,6212** | **0,7148** | **0,4686** |

Em relação ao modelo inicial, a média de mAP@50 aumentou de **0,6909 para 0,7148** (+0,0240) e a mAP@50–95 de **0,4441 para 0,4686** (+0,0245). O recall médio ficou praticamente estável. Os resultados completos estão em [`docs/metricas/finetune_test_metrics.json`](docs/metricas/finetune_test_metrics.json).

## Dataset

O conjunto combinado contém **10.284 imagens rotuladas** e **21.477 bounding boxes** de três classes. A divisão estratificada por classe dominante usa seed 42:

| Split | Imagens |
|---|---:|
| Treino | 7.199 |
| Validação | 2.057 |
| Teste | 1.028 |

As imagens e labels originais foram obtidas das três versões abaixo no Roboflow Universe. `data/raw/` e `data/dataset/` são ignoradas pelo Git, portanto baixe as versões indicadas e extraia cada export YOLOv7 em sua pasta correspondente antes de executar o notebook. O resumo de balanceamento, caixas por classe e origem está em [`docs/dataset_stats.md`](docs/dataset_stats.md).

| Dataset | Versão usada | Link da versão/exportação | Licença indicada pela fonte |
|---|---:|---|---|
| [road potholes and cracks](https://universe.roboflow.com/pothole-dataset-hhozt/road-potholes-and-cracks-jmxtp-kasjb-ef2as/dataset/1) | 1 | [Projeto no Roboflow Universe](https://universe.roboflow.com/pothole-dataset-hhozt/road-potholes-and-cracks-jmxtp-kasjb-ef2as) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| [object on road](https://universe.roboflow.com/ali-gjxr7/object-on-road-1qv79/dataset/1) | 1 | [Projeto no Roboflow Universe](https://universe.roboflow.com/ali-gjxr7/object-on-road-1qv79) | CC BY 4.0 (conforme `README.dataset.txt` do export local) |
| [BlindPath&Zebra_Dataset](https://universe.roboflow.com/experimentlab/blindpath-zebra_dataset/dataset/4) | 4 | [Projeto no Roboflow Universe](https://universe.roboflow.com/experimentlab/blindpath-zebra_dataset) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

Os arquivos `data.yaml` dos exports registram as versões 1, 1 e 4 e a licença CC BY 4.0. A redistribuição e adaptação sob CC BY 4.0 exigem atribuição, indicação de mudanças e link para a licença; consulte os termos na página da Creative Commons. O merge renomeia classes, filtra anotações e aplica balanceamento/augmentation a `piso_tatil`. Isso descreve a licença declarada nos metadados dos exports; verifique a página de cada fonte e preserve a atribuição ao reutilizar ou redistribuir dados e imagens.

Para reproduzir: baixe cada versão acima no formato YOLOv7 PyTorch e extraia para `data/raw/road potholes and cracks.v1i.yolov7pytorch/`, `data/raw/object on road.v1i.yolov7pytorch/` e `data/raw/BlindPath-Zebra_Dataset.v4i.yolov7pytorch/`. Em seguida execute `notebooks/01_merge_datasets.ipynb`.

### Limitação conhecida

`poste_obstaculo` continua sendo a classe mais difícil no teste após o fine-tuning: recall 0,4802 e mAP@50 0,5896, os menores resultados entre as classes. Objetos não detectados e confusões com `buraco_trinca` permanecem relevantes. A classe reúne objetos visualmente variados (postes, bancos, guias e lixeiras), e objetos pequenos ou parcialmente ocultos podem contribuir para as omissões. Essa explicação é uma hipótese baseada na composição da classe e no padrão das matrizes.

## Artefatos

- Pesos do treino inicial: [`models/best.pt`](models/best.pt); pesos e export ONNX do fine-tuning avaliado: [`models/best_finetuned.pt`](models/best_finetuned.pt) e [`models/best_finetuned.onnx`](models/best_finetuned.onnx). O APK usa esse ONNX; a PWA usa uma cópia em `app/Code/public/models/best.onnx`. Os exports TFLite e ONNX anteriores são do modelo inicial.
- Métricas detalhadas, curvas e matriz de confusão: [`docs/metricas/relatorio_treino.md`](docs/metricas/relatorio_treino.md).
- Dataset Ultralytics: `data/dataset/data.yaml`.
- Notebook de preparação e treino: `notebooks/01_merge_datasets.ipynb`, `notebooks/02_treino_yolov8.ipynb`; script de fine-tuning: `notebooks/04_finetune_score.py`.
- Notebook de exportação móvel: `notebooks/03_export_mobile.ipynb`.
- App Android nativo (Kotlin + ONNX Runtime): `app/android/`; instruções de build em [`app/android/README.md`](app/android/README.md). O APK atualizado já foi instalado no celular conectado.

### Instalar e usar o app

Com Android SDK e JDK 17 configurados, gere o APK debug com `app/android/gradlew.bat assembleDebug`. O arquivo fica em `app/android/app/build/outputs/apk/debug/app-debug.apk`. Instale-o abrindo o APK no Android ou via USB com `adb install -r app/android/app/build/outputs/apk/debug/app-debug.apk`. No app, escolha uma foto da galeria; as detecções aparecem com caixas e score. O limiar atual é 10% para facilitar a demonstração, então scores baixos podem incluir falsos positivos.

### Câmera traseira em tempo real (PWA)

O clone adaptado de `yoloinferApp` está em `app/Code/`: usa o ONNX do fine-tuning, os nomes das três classes e `getUserMedia` com `facingMode: environment` para inferência contínua pela câmera traseira. O limiar inicial da PWA é 40%. Para usar a câmera no celular, a PWA precisa ser publicada em HTTPS (ou servida em um contexto seguro) e receber permissão de câmera; o projeto tem configuração Netlify, mas ainda não há URL pública documentada nem confirmação de teste presencial dessa versão. O APK Kotlin instalado é um caminho nativo separado e processa fotos escolhidas da galeria.

Para a demonstração de 13/out com câmera ao vivo, falta publicar a PWA em HTTPS e validar no celular a permissão, o desempenho contínuo e as caixas. Para elevar scores em condições reais, a prioridade é coletar e rotular fotos do mesmo tipo de calçada, distância e iluminação observados no celular. Baixar o threshold só mostra detecções mais incertas; não aumenta a confiança do modelo. `poste_obstaculo` merece atenção especial (recall 0,480 e mAP@50 0,590 no teste), pois reúne objetos visualmente variados.

## Pendências para entrega pública

- Publicar `app/Code/` em HTTPS e confirmar a inferência ao vivo pela câmera traseira no celular.

## Estrutura

```text
data/
  raw/
  dataset/
    images/{train,val,test}/
    labels/{train,val,test}/
notebooks/
models/
app/
  android/                 # aplicativo Android nativo
docs/metricas/
runs/detect/cpzao_v1/weights/
```

GPU usada no treino: NVIDIA GeForce RTX 3060 com 12 GB de VRAM.

## Referências

- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [Auditoria de Calçadas](https://falzoni.com.br/auditoriacalcadas/)
- [Mjrovai/yoloinferApp](https://github.com/Mjrovai/yoloinferApp)
