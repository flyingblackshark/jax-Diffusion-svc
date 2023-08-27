wget -O pretrain/contentvec/checkpoint_best_legacy_500.pt https://huggingface.co/OOPPEENN/encoder_model/resolve/main/hubert_base.pt
wget -O pretrain/nsf_hifigan_20221211.zip https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-v1/nsf_hifigan_20221211.zip
unzip -o -d pretrain pretrain/nsf_hifigan_20221211.zip