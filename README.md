# CPZÃO — Auditoria de Calçadas

Pipeline de visão computacional para detectar problemas e elementos de acessibilidade em calçadas urbanas, com YOLOv8n e inferência móvel.

O objetivo é ajudar a identificar defeitos e obstáculos que dificultam a circulação a pé, além de reconhecer piso tátil. O protótipo classifica imagens em três categorias; ele não substitui uma vistoria técnica.

## Integrantes

| RM | Nome |
|---|---|
| RM551059 | Cassio Valezzi |
| RM98215 | Gabriel Antony Cadima Ciziks |
| RM98169 | Lucca Sabatini Tambellini |
| RM98209 | Victor Nuzzi |

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

## Dataset

O conjunto combinado contém **10.284 imagens rotuladas** e **21.477 bounding boxes** de três classes. A divisão estratificada por classe dominante usa seed 42:

| Split | Imagens |
|---|---:|
| Treino | 7.199 |
| Validação | 2.057 |
| Teste | 1.028 |

As origens são os datasets de buracos/trincas, obstáculos em vias e BlindPath-Zebra; o resumo de balanceamento, caixas por classe e origem está em [`docs/dataset_stats.md`](docs/dataset_stats.md). As imagens e labels não estão incluídas no repositório: `data/raw/` e `data/dataset/` são ignoradas pelo Git.

### Limitação conhecida

`poste_obstaculo` é a classe mais difícil no teste: obteve recall 0,4621 e mAP@50 0,5705, os menores resultados entre as classes. A matriz de confusão indica que o erro mais comum são objetos não detectados (classificados como background); há também confusão entre `buraco_trinca` e `poste_obstaculo`. A classe agrega objetos visualmente variados (postes, bancos, guias e lixeiras), e objetos pequenos ou parcialmente ocultos podem contribuir para as omissões. Essa explicação é uma hipótese baseada na composição da classe e no padrão da matriz.

## Artefatos

- Pesos treinados: [`models/best.pt`](models/best.pt); exports mobile: `models/best.tflite` e `models/best.onnx`.
- Métricas detalhadas, curvas e matriz de confusão: [`docs/metricas/relatorio_treino.md`](docs/metricas/relatorio_treino.md).
- Dataset Ultralytics: `data/dataset/data.yaml`.
- Notebook de preparação e treino: `notebooks/01_merge_datasets.ipynb` e `notebooks/02_treino_yolov8.ipynb`.
- Notebook de exportação móvel: `notebooks/03_export_mobile.ipynb`.
- App Android nativo (Kotlin + TFLite): `app/android/`; instruções de build em [`app/android/README.md`](app/android/README.md).

### Instalar e usar o app

Com Android SDK e JDK 17 configurados, gere o APK debug com `app/android/gradlew.bat assembleDebug`. O arquivo fica em `app/android/app/build/outputs/apk/debug/app-debug.apk`. Instale-o abrindo o APK no Android ou via USB com `adb install -r app/android/app/build/outputs/apk/debug/app-debug.apk`. No app, escolha uma foto da galeria; as detecções aparecem com caixas e score. O limiar atual é 10% para facilitar a demonstração, então scores baixos podem incluir falsos positivos.

### Câmera traseira em tempo real (PWA)

O clone adaptado de `yoloinferApp` está em `app/Code/`: usa `models/best.onnx`, os nomes das três classes e `getUserMedia` com `facingMode: environment` para inferência contínua pela câmera traseira. O limiar inicial da PWA é 40%. Para usar a câmera no celular, a PWA precisa ser publicada em HTTPS (ou servida em um contexto seguro) e receber permissão de câmera; o projeto tem configuração Netlify, mas ainda não há URL pública documentada nem confirmação de teste presencial dessa versão. O APK Kotlin instalado é um caminho nativo separado e, por enquanto, processa fotos escolhidas da galeria.

Para a demonstração de 13/out com câmera ao vivo, falta publicar a PWA em HTTPS e validar no celular a permissão, o desempenho contínuo e as caixas. Para melhorar scores baixos, a prioridade é coletar e rotular fotos do mesmo tipo de calçada, distância e iluminação observados no celular e continuar o treino: a mAP de validação ainda subiu de 0,651 na época 16 para 0,685 na época 20. Baixar o threshold só mostra detecções mais incertas; não aumenta a confiança do modelo. `poste_obstaculo` merece atenção especial (recall 0,462 e mAP@50 0,571 no teste), pois reúne objetos visualmente variados.

## Pendências para entrega pública

- Criar/conectar o repositório GitHub e publicar o projeto; esta pasta ainda não contém `.git`.
- Publicar `app/Code/` em HTTPS e confirmar a inferência ao vivo pela câmera traseira no celular.
- Registrar no README os links e licenças exatos dos datasets de origem para que outra pessoa consiga baixar os dados ignorados por Git e reproduzir o merge.

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
