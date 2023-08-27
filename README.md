English | [中文教程](README_zh_cn.md)
# SO-VITS-SVC 5.0 IN JAX
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
	python3 svc_inference.py --config configs/base.yaml --spk xxx.spk.npy --wave test.wav