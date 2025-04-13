# Video Upscaler using CoreML on Apple Silicon

This project is a video upscaler that uses CoreML on Apple Silicon to leverage the Apple Neural Engine on M-series chips.

This is a work in progress! Pushing now so that it's out there for others to use and modify.

## Installation

1. Install MacOS requirements

```bash
brew install ffmpeg
```

2. Install Python requirements

```bash
pip install -r requirements.txt
```

3. (Optional) Download additional models from [CoreML-Models](https://github.com/john-rocky/CoreML-Models) and place them in the `models` directory.

## Usage

Most models only support 4x upscaling. The most comprehensive model is the realcugan models.

RealCUGAN models are only supported for 240p input due to bugs in flexible input shapes export in CoreML.

Modify the `inference_video.py` script to use the model you want to use. TODO: make model selection configurable.

```bash
python inference_video.py -i inputs/ -o results/ -s 4
```

## Development notes

RealCUGAN source code is stored and modified to allow for export to CoreML. There were a bunch of lines that were using negative padding, which is not supported in CoreML. I've converted them to just slicing the tensor to get the model to export properly.

## Credits

- [Fast-SRGAN](https://github.com/HasnainRaz/Fast-SRGAN)
- [Real-CUGAN](https://github.com/bilibili/ailab/blob/main/Real-CUGAN/README_EN.md)
- [RealESRGAN](https://github.com/xinntao/Real-ESRGAN)
- [Video2X](https://github.com/k4yt3x/video2x)
- [CoreML-Models](https://github.com/john-rocky/CoreML-Models)
