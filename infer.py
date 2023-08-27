import os
import torch
import librosa
import argparse
import numpy as np
import soundfile as sf
from ast import literal_eval
from tools.infer_tools import DiffusionSVC
import jax.numpy as jnp
import jax

import optax
from flax.training.train_state import TrainState
from omegaconf import OmegaConf
from flax.training import orbax_utils
import orbax
from functools import partial
from diffusion.naive import Unit2MelNaive
from diffusion.gaussian import Gaussian
from diffusion.wavenet import WaveNet

jax.config.update('jax_platform_name', 'gpu')
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "-model",
    #     "--model",
    #     type=str,
    #     required=True,
    #     help="path to the diffusion model checkpoint",
    # )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=None,
        required=False,
        help="cpu or cuda, auto if not set")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="path to the input audio file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="path to the output audio file",
    )
    parser.add_argument(
        "-id",
        "--spk_id",
        type=str,
        required=False,
        default=1,
        help="speaker id (for multi-speaker model) | default: 1",
    )
    parser.add_argument(
        "-mix",
        "--spk_mix_dict",
        type=str,
        required=False,
        default="None",
        help="mix-speaker dictionary (for multi-speaker model) | default: None",
    )
    parser.add_argument(
        "-k",
        "--key",
        type=str,
        required=False,
        default=0,
        help="key changed (number of semitones) | default: 0",
    )
    parser.add_argument(
        "-f",
        "--formant_shift_key",
        type=str,
        required=False,
        default=0,
        help="formant changed (number of semitones) , only for pitch-augmented model| default: 0",
    )
    parser.add_argument(
        "-pe",
        "--pitch_extractor",
        type=str,
        required=False,
        default='crepe',
        help="pitch extrator type: parselmouth, dio, harvest, crepe (default) or rmvpe",
    )
    parser.add_argument(
        "-fmin",
        "--f0_min",
        type=str,
        required=False,
        default=50,
        help="min f0 (Hz) | default: 50",
    )
    parser.add_argument(
        "-fmax",
        "--f0_max",
        type=str,
        required=False,
        default=1100,
        help="max f0 (Hz) | default: 1100",
    )
    parser.add_argument(
        "-th",
        "--threhold",
        type=str,
        required=False,
        default=-60,
        help="response threhold (dB) | default: -60",
    )
    parser.add_argument(
        "-th4sli",
        "--threhold_for_split",
        type=str,
        required=False,
        default=-40,
        help="threhold for split (dB) | default: -40",
    )
    parser.add_argument(
        "-min_len",
        "--min_len",
        type=str,
        required=False,
        default=5000,
        help="min split len | default: 5000",
    )
    parser.add_argument(
        "-speedup",
        "--speedup",
        type=str,
        required=False,
        default=10,
        help="speed up | default: 10",
    )
    parser.add_argument(
        "-method",
        "--method",
        type=str,
        required=False,
        default='dpm-solver',
        help="ddim, pndm, dpm-solver or unipc | default: dpm-solver",
    )
    parser.add_argument(
        "-kstep",
        "--k_step",
        type=str,
        required=False,
        default=None,
        help="shallow diffusion steps | default: None",
    )
    # parser.add_argument(
    #     "-nmodel",
    #     "--naive_model",
    #     type=str,
    #     required=False,
    #     default=None,
    #     help="path to the naive model, shallow diffusion if not None and k_step not None",
    # )
    parser.add_argument(
        "-ir",
        "--index_ratio",
        type=str,
        required=False,
        default=0,
        help="index_ratio, if > 0 will use index | default: 0",
    )
    return parser.parse_args(args=args, namespace=namespace)

from omegaconf import OmegaConf
if __name__ == '__main__':
    # parse commands
    cmd = parse_args()

    device = cmd.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    diffusion_svc = DiffusionSVC(device=device)  # 加载模型
    diffusion_svc.load_model(f0_model=cmd.pitch_extractor, f0_max=cmd.f0_max, f0_min=cmd.f0_min)

    spk_mix_dict = literal_eval(cmd.spk_mix_dict)

    hp = OmegaConf.load("configs/base.yaml")
    def create_naive_state(): 
        r"""Create the training state given a model class. """ 
        rng = jax.random.PRNGKey(1234)
        model = Unit2MelNaive(input_channel=hp.data.encoder_out_channels, 
                    n_spk=hp.model_naive.n_spk,
                    use_pitch_aug=hp.model_naive.use_pitch_aug,
                    out_dims=128,
                    n_layers=hp.model_naive.n_layers,
                    n_chans=hp.model_naive.n_chans,
                    n_hidden=hp.model_naive.n_hidden,
                    use_speaker_encoder=hp.model_naive.use_speaker_encoder,
                    speaker_encoder_out_channels=hp.data.speaker_encoder_out_channels)
        tx =  optax.lion(learning_rate=0.01, b1=hp.train.betas[0],b2=hp.train.betas[1])
        fake_ppg = jnp.ones((1,400,768))
        fake_vol = jnp.ones((1,400,256))
        fake_pit = jnp.ones((1,400,256))
        params_key,r_key,dropout_key,rng = jax.random.split(rng,4)
        init_rngs = {'params': params_key, 'dropout': dropout_key,'rnorms':r_key}
        variables = model.init(init_rngs, ppg=fake_ppg, f0=fake_pit,volume=fake_vol)
        state = TrainState.create(apply_fn=model.apply, tx=tx,params=variables['params'])
        
        return state
    def create_wavenet_state(): 
        r"""Create the training state given a model class. """ 
        rng = jax.random.PRNGKey(1234)
        model = WaveNet(in_dims=128,
                            n_layers=hp.model_diff.n_layers,
                            n_chans=hp.model_diff.n_chans,
                            n_hidden=hp.model_diff.n_hidden)
        input_shape = (1, 128, 250)
        input_shapes = (input_shape, input_shape[0], input_shape)
        inputs = list(map(lambda shape: jnp.empty(shape), input_shapes))
        tx = optax.lion(learning_rate=0.01,b1=hp.train.betas[0],b2=hp.train.betas[1])
        variables = model.init(rng, *inputs)
        state = TrainState.create(apply_fn=model.apply, tx=tx,params=variables['params'])
        
        return state
    naive_state = create_naive_state()
    wavenet_state = create_wavenet_state()
    gaussian_config = hp['Gaussian']
    diff = Gaussian(**gaussian_config)
    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=2, create=True)
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        'chkpt/combine/', orbax_checkpointer, options)
    if checkpoint_manager.latest_step() is not None:
        target = {'model_naive': naive_state, 'model_wavenet': wavenet_state}
        step = checkpoint_manager.latest_step()  # step = 4
        states=checkpoint_manager.restore(step,items=target)
        naive_state=states['model_naive']
        wavenet_state=states['model_wavenet']
    del states
    # naive_model_path = cmd.naive_model
    # if naive_model_path is not None:
    #     if cmd.k_step is None:
    #         naive_model_path = None
    #         print(" [WARN] Could not shallow diffusion without k_step value when Only set naive_model path")
    #     else:
    #         diffusion_svc.load_naive_model(naive_model_path=naive_model_path)

    spk_emb = None

    # load wav
    in_wav, in_sr = librosa.load(cmd.input, sr=None)
    if len(in_wav.shape) > 1:
        in_wav = librosa.to_mono(in_wav)
    # infer
    out_wav, out_sr = diffusion_svc.infer_from_long_audio(
        in_wav, sr=in_sr,
        key=float(cmd.key),
        spk_id=int(cmd.spk_id),
        spk_mix_dict=spk_mix_dict,
        aug_shift=int(cmd.formant_shift_key),
        infer_speedup=int(cmd.speedup),
        method=cmd.method,
        k_step=cmd.k_step,
        use_tqdm=True,
        spk_emb=spk_emb,
        threhold=float(cmd.threhold),
        threhold_for_split=float(cmd.threhold_for_split),
        min_len=int(cmd.min_len),
        index_ratio=float(cmd.index_ratio),
        naive_state=naive_state,
        wavenet_state=wavenet_state,
        gaussian=diff
    )
    # save
    sf.write(cmd.output, out_wav, out_sr)
