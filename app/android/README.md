# Checkpoint Visão (Android nativo)

Aplicativo Kotlin para selecionar uma imagem do dispositivo e executar o modelo YOLO pelo ONNX Runtime para Android.

- Modelo: `app/src/main/assets/best.onnx` (float32, entrada `[1, 3, 640, 640]`, saída `[1, 7, 8400]`), executado com ONNX Runtime para Android.
- Labels: `app/src/main/assets/labels.txt`, na ordem do modelo.
- Threshold atual: `0.10`; NMS por classe com IoU `0.45`.
- `best.onnx` é o checkpoint de fine-tuning exportado para ONNX. A saída do modelo tem formato `[1, 7, 8400]`; o app lê cada um dos 7 canais (4 coordenadas e 3 scores de classe).

## Build debug

Requisitos: JDK 17, Android SDK Platform 35 e Build-Tools 34. Aceite a licença do Android SDK no SDK Manager e defina `ANDROID_HOME`/`ANDROID_SDK_ROOT` para o SDK instalado. Em seguida:

```powershell
cd app/android
.\gradlew.bat assembleDebug
```

O APK será criado em `app/build/outputs/apk/debug/app-debug.apk`.
