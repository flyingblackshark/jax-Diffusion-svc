English | [中文教程](README_zh_cn.md)
# DIFFUSION SVC IN JAX
The following tutorials are for Google TPU v2-8/v3-8

## Prepare Environment
	pip install -r requirements.txt
	pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
	sudo apt update && sudo apt install -y libsndfile1 ffmpeg
## Prepare Dataset
Dwonload pretrained models using download.sh

Dataset Folder Strucutre:
```
data
├─ train
│    ├─ audio
│    │    ├─ 1
│    │    │   ├─ aaa.wav
│    │    │   ├─ bbb.wav
│    │    │   └─ ....wav
│    │    ├─ 2
│    │    │   ├─ ccc.wav
│    │    │   ├─ ddd.wav
│    │    │   └─ ....wav
│    │    └─ ...
|
├─ val
|    ├─ audio
│    │    ├─ 1
│    │    │   ├─ eee.wav
│    │    │   ├─ fff.wav
│    │    │   └─ ....wav
│    │    ├─ 2
│    │    │   ├─ ggg.wav
│    │    │   ├─ hhh.wav
│    │    │   └─ ....wav
│    │    └─ ...
```
and
```
python preprocess.py -c configs/base.yaml
```
## Train Your Model
	python3 svc_trainer_combine.py
## Inference
	python infer.py -i .\ourola.wav -o output9.wav -k 0 -id 1 -speedup 10 -method "dpm-solver" -kstep 200 // k id speedup method kstep arguments is not functional
### [Discord Channel](https://discord.gg/mrGUhMVWUM)