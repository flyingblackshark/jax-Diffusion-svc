import os
import time
import logging
import tqdm
import flax
import jax
import optax
import numpy as np
import orbax
from flax import linen as nn
from diff_extend.writer import MyWriter
import jax.numpy as jnp
import orbax.checkpoint
from functools import partial
from flax.training import train_state
from flax.training.common_utils import shard, shard_prng_key
from flax.training import orbax_utils
from diffusion.naive import Unit2MelNaive
from diffusion.gaussian import Gaussian
from diffusion.wavenet import WaveNet
from typing import Any

class EMATrainState(train_state.TrainState):
    ema_params: Any = None

@partial(jax.pmap, )
def update_ema(state, ema_decay=0.999, ):
    new_ema_params = jax.tree_map(lambda ema, normal: ema * ema_decay + (1 - ema_decay) * normal, state.ema_params , state.params)
    state = state.replace(ema_params=new_ema_params)
    return state

PRNGKey = jnp.ndarray
def create_naive_state(rng, model_cls,hp,trainloader): 
    r"""Create the training state given a model class. """ 
    model = model_cls(input_channel=hp.data.encoder_out_channels, 
                    n_spk=hp.model_naive.n_spk,
                    use_pitch_aug=hp.model_naive.use_pitch_aug,
                    out_dims=128,
                    n_layers=hp.model_naive.n_layers,
                    n_chans=hp.model_naive.n_chans,
                    n_hidden=hp.model_naive.n_hidden,
                    use_speaker_encoder=hp.model_naive.use_speaker_encoder,
                    speaker_encoder_out_channels=hp.data.speaker_encoder_out_channels)
    
    #exponential_decay_scheduler = optax.exponential_decay(init_value=hp.train.learning_rate, transition_steps=hp.train.total_steps,decay_rate=hp.train.lr_decay)
    tx = optax.lion(learning_rate=hp.train.learning_rate, b1=hp.train.betas[0],b2=hp.train.betas[1])
        

    
    params_key,r_key,dropout_key,rng = jax.random.split(rng,4)
    init_rngs = {'params': params_key, 'dropout': dropout_key,'rnorms':r_key}
    data = next(iter(trainloader))
    i1 = jnp.asarray(data['units'])
    i2 = jnp.asarray(data['f0'])
    i3 = jnp.asarray(data['volume'])
    inputs = (i1,i2,i3)
    variables = model.init(init_rngs, *inputs)

    state = EMATrainState.create(apply_fn=model.apply, tx=tx, params=variables['params'], ema_params=variables['params'])
    
    return state
def create_wavenet_state(rng, model_cls,hp,trainloader): 
    r"""Create the training state given a model class. """ 

    model = model_cls(in_dims=128,
                        n_layers=hp.model_diff.n_layers,
                        n_chans=hp.model_diff.n_chans,
                        n_hidden=hp.model_diff.n_hidden)

    input_shape = (1, 128, 172)
    input_shapes = (input_shape, input_shape[0], input_shape)
    inputs = list(map(lambda shape: jnp.empty(shape), input_shapes))

    #exponential_decay_scheduler = optax.exponential_decay(init_value=hp.train.learning_rate, transition_steps=hp.train.total_steps, decay_rate=hp.train.lr_decay)
    tx = optax.lion(learning_rate=hp.train.learning_rate, b1=hp.train.betas[0],b2=hp.train.betas[1])

    variables = model.init(rng, *inputs)

    state = EMATrainState.create(apply_fn=model.apply, tx=tx,params=variables['params'], ema_params=variables['params'])
    
    return state
def train(args,chkpt_path, hp):
    num_devices = jax.device_count()
    gaussian_config = hp['Gaussian']
    diff = Gaussian(**gaussian_config)
    @partial(jax.pmap, axis_name='num_devices')
    def combine_step(naive_state: EMATrainState,wavenet_state: EMATrainState, ppg : jnp.ndarray , pit : jnp.ndarray,spec : jnp.ndarray, vol:jnp.ndarray,rng_e:PRNGKey):
        ppg = jnp.asarray(ppg)
        pit = jnp.asarray(pit)
        spec = jnp.asarray(spec)
        
        
        def naive_loss_fn(params):
            fake_mel = naive_state.apply_fn({'params': params},ppg=ppg,f0=pit,volume=vol,gt_spec=spec,infer=False,rngs={'rnorms':rng_e})
            fake_mel_ema = naive_state.apply_fn({'params': naive_state.ema_params},ppg=ppg,f0=pit,volume=vol,gt_spec=spec,infer=False,rngs={'rnorms':rng_e})
            loss_naive = optax.squared_error(fake_mel, spec).mean()
            loss_naive_ema = optax.squared_error(fake_mel_ema, spec).mean()
            return loss_naive,(fake_mel,loss_naive_ema)

        grad_fn = jax.value_and_grad(naive_loss_fn, has_aux=True)
        (loss_naive,(fake_mel,loss_naive_ema)), grads_naive = grad_fn(naive_state.params)
        grads_naive = jax.lax.pmean(grads_naive, axis_name='num_devices')
        loss_naive = jax.lax.pmean(loss_naive, axis_name='num_devices')
        loss_naive_ema = jax.lax.pmean(loss_naive_ema, axis_name='num_devices')
        new_naive_state = naive_state.apply_gradients(grads=grads_naive)
        fake_mel = fake_mel.transpose(0,2,1)
        spec = spec.transpose(0,2,1)
        def diff_loss_fn(params):
            loss_diff = diff(rng_e, wavenet_state, params, spec, fake_mel,hp.shallow.k_step_max)
            return loss_diff
        
        grad_fn = jax.value_and_grad(diff_loss_fn, has_aux=False)
        loss_diff, grads_diff = grad_fn(wavenet_state.params)
        grads_diff = jax.lax.pmean(grads_diff, axis_name='num_devices')
        loss_diff = jax.lax.pmean(loss_diff, axis_name='num_devices')

        new_wavenet_state = wavenet_state.apply_gradients(grads=grads_diff)
        return new_naive_state,new_wavenet_state,loss_naive,loss_diff,loss_naive_ema
    @partial(jax.pmap, axis_name='num_devices')         
    def generate_hidden(naive_state: EMATrainState,ppg_val:jnp.ndarray,pit_val:jnp.ndarray,vol_val:jnp.ndarray):
        predict_key = jax.random.PRNGKey(1234)
        hidden = naive_state.apply_fn({'params': naive_state.params},ppg=ppg_val, f0=pit_val,volume=vol_val,infer=True, mutable=False,rngs={'rnorms':predict_key})
        return hidden
    def do_validate(wavenet_state:EMATrainState,hidden:jnp.ndarray,spec_val:jnp.ndarray):   
        spec_val=spec_val.transpose(0,2,1)
        mel_fake = diff.sample(key, wavenet_state,hidden )
        mel_loss_val = jnp.mean(jnp.abs(mel_fake - spec_val))
        spec_fake = mel_fake[0]
        spec_real = spec_val[0]
        return mel_loss_val,spec_fake, spec_real
    def validate(naive_state,wavenet_state):
        loader = tqdm.tqdm(valloader, desc='Validation loop')
       
     
        mel_loss = 0.0
        for data in loader:
            val_ppg = shard(jnp.asarray(data['units']))
            val_pit = shard(jnp.asarray(data['f0']))
            val_vol = shard(jnp.asarray(data['volume']))
            val_spec = jnp.asarray(data['mel'])
            hidden = generate_hidden(naive_state,val_ppg,val_pit,val_vol)
            hidden=hidden.squeeze(1)
            hidden=hidden.transpose(0,2,1)
            mel_loss_val,spec_fake,spec_real=do_validate(wavenet_state,hidden,val_spec)
            mel_loss += mel_loss_val.mean()
            writer.log_fig_audio(np.asarray(spec_fake), np.asarray(spec_real),np.asarray(hidden[0]), 0, step)

        mel_loss = mel_loss / len(valloader.dataset)
        mel_loss = np.asarray(mel_loss)
       
        writer.log_validation(mel_loss, step)

    key = jax.random.PRNGKey(seed=hp.train.seed)
    combine_step_key,key_generator, key_discriminator, key = jax.random.split(key, 4)
    
    init_epoch = 1
    step = 0
    pth_dir = os.path.join(hp.log.pth_dir, args.name)
    log_dir = os.path.join(hp.log.log_dir, args.name)
    os.makedirs(pth_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, '%s-%d.log' % (args.name, time.time()))),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()
    writer = MyWriter(hp, log_dir)
    from diffusion.data_loaders import get_data_loaders
    trainloader, valloader = get_data_loaders(hp, whole_audio=False)

    naive_state = create_naive_state(key_discriminator,Unit2MelNaive ,hp,trainloader)
    wavenet_state = create_wavenet_state(key_generator, WaveNet,hp,trainloader)

    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=hp.log.max_to_keep, create=True)
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        'chkpt/combine/', orbax_checkpointer, options)
    if checkpoint_manager.latest_step() is not None:
        target = {'model_naive': naive_state, 'model_wavenet': wavenet_state}
        step = checkpoint_manager.latest_step()  # step = 4
        states=checkpoint_manager.restore(step,items=target)
        naive_state=states['model_naive']
        wavenet_state=states['model_wavenet']

    naive_state = flax.jax_utils.replicate(naive_state)
    wavenet_state = flax.jax_utils.replicate(wavenet_state)

    for epoch in range(init_epoch, hp.train.epochs):

        loader = tqdm.tqdm(trainloader, desc='Loading train data')
        for data in loader:
            step_key,combine_step_key=jax.random.split(combine_step_key)
            step_key = shard_prng_key(step_key)
            ppg = shard(jnp.asarray(data['units']))
            pit = shard(jnp.asarray(data['f0']))
            spec = shard(jnp.asarray(data['mel']))
            vol = shard(jnp.asarray(data['volume']))
            naive_state,wavenet_state,loss_naive,loss_diff,loss_ema=combine_step(naive_state,wavenet_state,ppg=ppg,pit=pit, spec=spec,vol=vol,rng_e=step_key)
            # EMA
            decay = min(0.9999, (1 + step) / (10 + step))
            decay = flax.jax_utils.replicate(jnp.array([decay]))
            naive_state = update_ema(naive_state, decay)
            wavenet_state = update_ema(wavenet_state, decay)
            # Update Step
            step += 1

            loss_naive,loss_diff,loss_ema = jax.device_get([loss_naive[0],loss_diff[0],loss_ema[0]])
            
            if step % hp.log.info_interval == 0:
                # writer.log_training(
                #     loss_g, loss_d, loss_m, loss_s, loss_k, loss_r, score_loss,step)
                logger.info("loss_naive %.04f loss_diff %.04f loss_ema %.04f | step %d" % (loss_naive,loss_diff,loss_ema, step))
                
        if epoch % hp.log.eval_interval == 0:
            validate(naive_state,wavenet_state)
        if epoch % hp.log.save_interval == 0:
            naive_state_s = flax.jax_utils.unreplicate(naive_state)
            wavenet_state_s = flax.jax_utils.unreplicate(wavenet_state)
            ckpt = {'model_naive': naive_state_s, 'model_wavenet': wavenet_state_s}
            save_args = orbax_utils.save_args_from_target(ckpt)
            checkpoint_manager.save(step, ckpt, save_kwargs={'save_args': save_args})

