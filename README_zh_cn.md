[English](README.md) | 中文教程
# DIFFUSION-SVC IN JAX
以下教程针对谷歌TPU v2-8/v3-8

## 配置环境
	pip install -r requirements.txt
	pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
	sudo apt update && sudo apt install -y libsndfile1 ffmpeg
## 制作数据集
运行 download.sh

数据集按以下形式存放
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
随后运行
```
python preprocess.py -c configs/base.yaml
```
## 开始训练
	python3 svc_trainer_combine.py
## 推理
	python infer.py -i .\ourola.wav -o output9.wav -k 0 -id 1 -speedup 10 -method "dpm-solver" -kstep 200 // k id speedup method kstep参数仅占位，不起作用

### 讨论QQ群 771728973 [Discord](https://discord.gg/mrGUhMVWUM)