"""
Code adapted from:
https://github.com/BarqueroGerman/BeLFusion
"""
import torch
import torch.nn as nn
import os
import model as module_arch
from model.diffusion.mlp_diffae import MLPSkipNet, Activation
from model.diffusion.diffusion_prior.transformer_prior import DiffusionPriorNetwork
from model.diffusion.diffusion_decoder.transformer_denoiser import TransformerDenoiser
from model.diffusion.gaussian_diffusion import PriorLatentDiffusion, DecoderLatentDiffusion
from model.diffusion.resample import UniformSampler
from utils.util import load_config_from_file, checkpoint_load, checkpoint_resume
from einops import rearrange


class BaseLatentModel(nn.Module):
    def __init__(self, device, cfg, emb_preprocessing=False, freeze_encoder=True):
        super(BaseLatentModel, self).__init__()
        self.emb_preprocessing = emb_preprocessing
        self.freeze_encoder = freeze_encoder
        def_dtype = torch.get_default_dtype()

        # RNN_VAE embedder
        self.latent_embedder = getattr(module_arch, cfg.latent_embedder.type)(cfg.latent_embedder.args)
        model_path = cfg.latent_embedder.checkpoint_path
        assert os.path.exists(model_path), (
            "Miss checkpoint for latent embedder: {}.".format(model_path))
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        self.latent_embedder.load_state_dict(state_dict)

        # Speaker audio auto-encoder
        self.audio_encoder = getattr(module_arch, cfg.audio_encoder.type)(**cfg.audio_encoder.args)
        model_path = cfg.audio_encoder.get("checkpoint_path", None)
        if model_path is not None:
            checkpoint = torch.load(model_path, map_location='cpu')
            state_dict = checkpoint['state_dict']
            self.audio_encoder.load_state_dict(state_dict)

        # Listener personal-specific encoder
        self.person_encoder = getattr(module_arch, cfg.person_specific.type)(device, **cfg.person_specific.args)
        model_path = cfg.person_specific.checkpoint_path
        assert os.path.exists(model_path), (
            "Miss checkpoint for audio encoder: {}.".format(model_path))
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        self.person_encoder.load_state_dict(state_dict)

        # ====== TODO: (optional) RNN_VAE embedder for 3dmm feature encodings
        self.latent_3dmm_embedder = getattr(module_arch, cfg.latent_3dmm_embedder.type)(cfg.latent_3dmm_embedder.args)
        model_path = cfg.latent_3dmm_embedder.checkpoint_path
        assert os.path.exists(model_path), (
            "Miss checkpoint for latent embedder: {}.".format(model_path))
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        self.latent_3dmm_embedder.load_state_dict(state_dict)
        # ======

        # assert "statistics" in checkpoint or emb_preprocessing.lower() == "none", \
        #     "Model statistics are not available in its checkpoint. Can't apply embeddings preprocessing."
        # self.embed_emotion_stats = checkpoint["statistics"] if "statistics" in checkpoint else None

        if self.freeze_encoder:  # freeze models
            for para in self.latent_embedder.parameters():
                para.requires_grad = False
            for para in self.latent_3dmm_embedder.parameters():
                para.requires_grad = False
            for para in self.person_encoder.parameters():
                para.requires_grad = False

        torch.set_default_dtype(def_dtype)  # config loader changes this

        self.init_params = None

    def deepcopy(self):
        assert self.init_params is not None, "Cannot deepcopy LatentUNetMatcher if init_params is None."
        # I can't deep copy this class. I need to do this trick to make the deepcopy of everything
        model_copy = self.__class__(**self.init_params)
        weights_path = f'weights_temp_{id(model_copy)}.pt'
        torch.save(self.state_dict(), weights_path)
        model_copy.load_state_dict(torch.load(weights_path))
        os.remove(weights_path)
        return model_copy

    def preprocess(self, emb):
        stats = self.embed_emotion_stats
        if stats is None:
            return emb  # when no checkpoint was loaded, there is no stats.

        if "standardize" in self.emb_preprocessing:
            return (emb - stats["mean"]) / torch.sqrt(stats["var"])
        elif "normalize" in self.emb_preprocessing:
            return 2 * (emb - stats["min"]) / (stats["max"] - stats["min"]) - 1
        elif "none" in self.emb_preprocessing.lower():
            return emb
        else:
            raise NotImplementedError(f"Error on the embedding preprocessing value: '{self.emb_preprocessing}'")

    def undo_preprocess(self, emb):
        stats = self.embed_emotion_stats
        if stats is None:
            return emb  # when no checkpoint was loaded, there is no stats.

        if "standardize" in self.emb_preprocessing:
            return torch.sqrt(stats["var"]) * emb + stats["mean"]
        elif "normalize" in self.emb_preprocessing:
            return (emb + 1) * (stats["max"] - stats["min"]) / 2 + stats["min"]
        elif "none" in self.emb_preprocessing.lower():
            return emb
        else:
            raise NotImplementedError(f"Error on the embedding preprocessing value: '{self.emb_preprocessing}'")

    def forward(self, pred, timesteps, seq_em):
        raise NotImplementedError("This is an abstract class.")

    # override checkpointing
    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def cuda(self):
        return self.to(torch.device("cuda"))

    # override eval and train
    def train(self, mode=True):
        self.model.train(mode)

    def eval(self):
        self.model.eval()


class PriorLatentMatcher(BaseLatentModel):
    """ The Prior diffusion model
    """

    def __init__(self, conf, device):
        cfg = conf.diffusion_prior.args
        super(PriorLatentMatcher, self).__init__(
            device,
            conf,
            emb_preprocessing=cfg.emb_preprocessing,
            freeze_encoder=cfg.freeze_encoder,
        )

        self.window_size = cfg.window_size
        self.token_len = cfg.token_len
        self.init_params = {
            "audio_dim": cfg.get("audio_dim", 78),
            "window_size": cfg.get("window_size", 50),
            "_3dmm_dim": cfg.get("_3dmm_dim", 58),
            "speaker_emb_dim": cfg.get("speaker_emb_dim", 512),
            "latent_dim": cfg.get("latent_dim", 512),
            "depth": cfg.get("depth", 6),
            "num_time_layers": cfg.get("num_time_layers", 2),
            "num_time_embeds": cfg.get("num_time_embeds", 1),
            "num_time_emb_channels": cfg.get("num_time_emb_channels", 64),
            "time_last_act": cfg.get("time_last_act", False),
            "use_learned_query": cfg.get("use_learned_query=True", True),
            # for classifier free guidance.
            "s_audio_cond_drop_prob": cfg.get("s_audio_cond_drop_prob", 0.5),
            "s_latentemb_cond_drop_prob": cfg.get("s_latentemb_cond_drop_prob", 0.5),
            "s_3dmm_cond_drop_prob": cfg.get("s_3dmm_cond_drop_prob", 0.5),
            "guidance_scale": cfg.get("guidance_scale", 7.5),
            "dim_head": cfg.get("dim_head", 64),
            "ff_mult": cfg.get("ff_mult", 4),
            "norm_in": cfg.get("norm_in", False),
            "norm_out": cfg.get("norm_out", True),
            "attn_dropout": cfg.get("attn_dropout", 0.0),
            "ff_dropout": cfg.get("ff_dropout", 0.0),
            "final_proj": cfg.get("final_proj", True),
            "normformer": cfg.get("normformer", False),
            "rotary_emb": cfg.get("rotary_emb", True),
        }
        self.model = DiffusionPriorNetwork(**self.init_params)

        self.mode = conf.mode
        # if self.mode == "test":  # load checkpoints
        #     checkpoint_load(conf, self.model, device)

        resume = conf.trainer.get("resume", None)
        if resume is not None:  # during training
            checkpoint_resume(conf, self.model, device)

        # diffusion forward / denoise
        self.prior_diffusion = PriorLatentDiffusion(
            conf.diffusion_prior.args,
            conf.diffusion_prior.scheduler,
            conf.diffusion_prior.scheduler.num_train_timesteps,
            conf.diffusion_prior.scheduler.num_inference_timesteps,
        )

        self.schedule_sampler = UniformSampler(self.prior_diffusion)

        self.window_sliding = conf.diffusion_prior.args.window_sliding
        self.k = conf.diffusion_prior.scheduler.k  # k appropriate generations

    def _forward(
            self,
            speaker_audio=None,  # optional condition, for training.
            speaker_emotion=None,  # optional condition, emotion, for training.
            speaker_3dmm=None,  # optional condition, 3dmm, for training.
            listener_emotion=None,  # condition, emotion, k appropriate, for training.
            listener_3dmm=None,  # condition, 3dmm, k appropriate, for training.
            **kwargs,
    ):

        batch_size, seq_len, d = speaker_audio.shape
        emo_dim = listener_emotion.shape[-1]
        _3dmm_dim = listener_3dmm.shape[-1]

        if self.mode in ["train", "val"]:  # sample random sub-window for each batch element
            # sample random window size for all batch elements
            window_start = torch.randint(0, seq_len - self.window_size, (1,), device=speaker_audio.device)
            window_end = window_start + self.window_size

            # shape: (batch_size, window_size, ...)
            s_audio_selected = speaker_audio[:, window_start:window_end]
            s_emotion_selected = speaker_emotion[:, window_start:window_end]
            s_3dmm_selected = speaker_3dmm[:, window_start:window_end]
            # shape: (batch_size * k, window_size, ...)
            l_emotion_selected = listener_emotion[:, window_start:window_end]
            # l_3dmm_selected = listener_3dmm[:, window_start:window_end]

            with torch.no_grad():
                s_audio_encodings = self.audio_encoder._encode(s_audio_selected)
                # s_audio_encodings.shape: (batch_size, window_size, 78)
                # we repeat the obs 'k' times for k appropriate reactions
                s_audio_encodings = s_audio_encodings.repeat_interleave(self.k, dim=0)

                # listener latent embedding (x_0 in diffusion) to be predicted (and forward diffused)
                x_start = self.latent_embedder.encode(l_emotion_selected).unsqueeze(1)  # (..., l_3dmm_selected)

                # freeze latent RNN_VAE embedder to extract speaker latent embedding
                s_latent_embed = self.latent_embedder.encode(s_emotion_selected).unsqueeze(1)  # (..., s_3dmm_selected)
                s_latent_embed = s_latent_embed.repeat_interleave(self.k, dim=0)

                # speaker 3dmm feature, optional condition
                s_3dmm_encodings = s_3dmm_selected.repeat_interleave(self.k, dim=0)

                model_kwargs = {"speaker_audio_encodings": s_audio_encodings,
                                "speaker_latent_emb": s_latent_embed,
                                "speaker_3dmm_encodings": s_3dmm_encodings}

            t, weights = self.schedule_sampler.sample(batch_size, speaker_audio.device)
            # timestep, with shape: (batch_size,)
            # weights for each timestep: with shape: (batch_size,)

            output_pred = self.prior_diffusion.denoise(self.model, x_start, t, model_kwargs=model_kwargs)

            return output_pred

        else:  # iterate over all windows
            # TODO: we modify the online inference strategy. we use sliding window.
            # assert seq_len % self.window_size == 0, "seq_len must be divisible by window_size"
            # diff_batch = batch_size * (seq_len // self.window_size)
            # s_audio = speaker_audio.reshape(diff_batch, self.window_size, d)
            # s_emotion = speaker_emotion.reshape(diff_batch, self.window_size, emo_dim)
            # s_3dmm = speaker_3dmm.reshape(diff_batch, self.window_size, _3dmm_dim)
            # listener_emotion = listener_emotion.reshape(-1, self.k, (seq_len // self.window_size),
            #                                             self.window_size, emo_dim)
            # listener_emotion = listener_emotion.transpose(1, 2).contiguous()
            # l_emotion = listener_emotion.reshape(-1, self.window_size, emo_dim)

            assert (seq_len - self.window_size) % self.window_sliding == 0, \
                "must be divisible when using sliding window."
            num_windows = (seq_len - self.window_size) // self.window_sliding + 1
            s_audio = (torch.zeros(size=(batch_size, num_windows, self.window_size, speaker_audio.shape[-1]))
                       .to(speaker_audio.device))
            s_emotion = (torch.zeros(size=(batch_size, num_windows, self.window_size, speaker_emotion.shape[-1]))
                         .to(speaker_emotion.device))
            s_3dmm = (torch.zeros(size=(batch_size, num_windows, self.window_size, speaker_3dmm.shape[-1]))
                      .to(speaker_3dmm.device))

            for i in range(num_windows):
                start = (i * self.window_sliding)
                end = self.window_size + (i * self.window_sliding)
                s_audio[:, i, :, :] = speaker_audio[:, start:end, :]
                s_emotion[:, i, :, :] = speaker_emotion[:, start:end, :]
                s_3dmm[:, i, :, :] = speaker_3dmm[:, start:end, :]
            # reshape [bs, num_windows, window_size, dim] ==> [bs * num_windows, window_size, dim]
            s_audio = s_audio.reshape(-1, self.window_size, speaker_audio.shape[-1])
            s_emotion = s_emotion.reshape(-1, self.window_size, speaker_emotion.shape[-1])
            s_3dmm = s_3dmm.reshape(-1, self.window_size, speaker_3dmm.shape[-1])

            with torch.no_grad():
                s_audio_encodings = self.audio_encoder._encode(s_audio)
                s_audio_encodings = s_audio_encodings.repeat_interleave(self.k, dim=0)

                s_latent_embed = self.latent_embedder.encode(s_emotion).unsqueeze(1)  # (..., s_3dmm)
                s_latent_embed = s_latent_embed.repeat_interleave(self.k, dim=0)

                # speaker 3dmm feature, optional condition
                speaker_3dmm_encodings = s_3dmm.repeat_interleave(self.k, dim=0)

                # speaker emotion feature, only for return
                speaker_emotion_encodings = s_emotion.repeat_interleave(self.k, dim=0)

                # listener latent embedding to be predicted (and forward diffused)
                # TODO: we only need the shape of listener_gt
                # listener_gt = self.latent_embedder.encode(l_emotion).unsqueeze(1)
                listener_gt = torch.zeros(s_latent_embed.size())

                model_kwargs = {"speaker_audio_encodings": s_audio_encodings,
                                "speaker_latent_emb": s_latent_embed,
                                "speaker_3dmm_encodings": speaker_3dmm_encodings}

            output = [output for output in self.prior_diffusion.ddim_sample_loop_progressive(
                matcher=self,
                model=self.model,
                model_kwargs=model_kwargs,
                gt=listener_gt,
            )][-1]  # get last output

            output_prior = output["sample_enc"]

            # add speaker emotion encodings to model_kwargs only for next stage
            model_kwargs = model_kwargs.copy()
            model_kwargs["speaker_emotion_encodings"] = speaker_emotion_encodings

            return output_prior, model_kwargs

    def forward_prior(self,
                      speaker_audio=None,  # optional condition, for training.
                      speaker_emotion_input=None,  # optional condition, emotion, for training.
                      speaker_3dmm_input=None,  # optional condition, 3dmm, for training.
                      listener_emotion_input=None,  # condition, emotion, k appropriate, for training.
                      listener_3dmm_input=None,  # condition, 3dmm, k appropriate, for training.
                      **kwargs):

        if self.mode in ["train", "val"]:
            # conditions
            speaker_audio_shifted = speaker_audio[:, :-self.window_size]  # (batch_size, seq_len, ...)
            speaker_emotion_shifted = speaker_emotion_input[:, :-self.window_size]  # emotion
            speaker_3dmm_shifted = speaker_3dmm_input[:, :-self.window_size]  # 3dmm

            # condition, we got k appropriate reacted listeners for each speaker
            # shape: (batch_size * k, seq_len, ...)
            listener_emotion_shifted = listener_emotion_input[:, self.window_size:]  # emotion
            listener_3dmm_shifted = listener_3dmm_input[:, self.window_size:]  # 3dmm

        else:
            # shift speaker by window size and fill with zeros on the left
            # TODO: an alternative strategy might be filling it with the most common speaker emotion,
            #  or use other strategy for online.
            # speaker_audio_shifted = torch.cat([torch.zeros_like(speaker_audio[:, :self.window_size]),
            #                                    speaker_audio[:, :-self.window_size]], dim=1)
            # speaker_emotion_shifted = torch.cat([torch.zeros_like(speaker_emotion_input[:, :self.window_size]),
            #                                      speaker_emotion_input[:, :-self.window_size]], dim=1)
            # speaker_3dmm_shifted = torch.cat([torch.zeros_like(speaker_3dmm_input[:, :self.window_size]),
            #                                   speaker_3dmm_input[:, :-self.window_size]], dim=1)
            # listener_emotion_shifted = listener_emotion_input
            # listener_3dmm_shifted = listener_3dmm_input

            # TODO: we use sliding window. Note that we need to create 'self.window_sliding'.
            speaker_audio_shifted = torch.cat((torch.zeros_like(speaker_audio[:, :self.window_sliding]),
                                               speaker_audio[:, :-self.window_sliding]), dim=1)
            speaker_emotion_shifted = torch.cat([torch.zeros_like(speaker_emotion_input[:, :self.window_sliding]),
                                                 speaker_emotion_input[:, :-self.window_sliding]], dim=1)
            speaker_3dmm_shifted = torch.cat([torch.zeros_like(speaker_3dmm_input[:, :self.window_sliding]),
                                              speaker_3dmm_input[:, :-self.window_sliding]], dim=1)
            listener_emotion_shifted = listener_emotion_input
            listener_3dmm_shifted = listener_3dmm_input

        # for the same listener window to be predicted, the speaker will correspond to the past
        return self._forward(speaker_audio_shifted,
                             speaker_emotion_shifted,
                             speaker_3dmm_shifted,
                             listener_emotion_shifted,
                             listener_3dmm_shifted,
                             **kwargs)

    def forward(self, **kwargs):
        return self.forward_prior(**kwargs)


class DecoderLatentMatcher(BaseLatentModel):
    """ The decoder diffusion model
    """

    def __init__(self, conf, device):
        cfg = conf.diffusion_decoder.args
        super(DecoderLatentMatcher, self).__init__(
            device,
            conf,
            emb_preprocessing=cfg.emb_preprocessing,
            freeze_encoder=cfg.freeze_encoder,
        )

        self.reaction_delay = cfg.reaction_delay  # we set reaction delay to 8 frames.
        self.window_size = cfg.window_size
        self.token_len = cfg.token_len
        self._3dmm_dim = cfg.get("nfeats", 58)  # 3dmm feat dimension
        # (optional) whether encode raw emotion or 3dmm features to the encodings.
        self.encode_emotion = cfg.get("encode_emotion", False)
        self.encode_3dmm = cfg.get("encode_3dmm", False)
        self.use_past_frames = cfg.get("use_past_frames", False)

        self.init_params = {
            "reaction_delay": self.reaction_delay,
            "token_len": self.token_len,
            "encode_emotion": self.encode_emotion,
            "encode_3dmm": self.encode_3dmm,
            "ablation_skip_connection": cfg.get("ablation_skip_connection", True),
            "nfeats": cfg.get("nfeats", 58),
            "latent_dim": cfg.get("latent_dim", 512),
            "ff_size": cfg.get("ff_size", 1024),
            "num_layers": cfg.get("num_layers", 6),
            "num_heads": cfg.get("num_heads", 8),
            "dropout": cfg.get("dropout", 0.1),
            "normalize_before": cfg.get("normalize_before", False),
            "activation": cfg.get("activation", "gelu"),
            "flip_sin_to_cos": cfg.get("flip_sin_to_cos", True),
            "return_intermediate_dec": cfg.get("return_intermediate_dec", False),
            "position_embedding": cfg.get("position_embedding", "learned"),
            "arch": cfg.get("arch", "trans_enc"),
            "use_attn_mask": cfg.get("use_attn_mask", True),
            "period": cfg.get("period", 50),  # period in biased_mask
            "freq_shift": cfg.get("freq_shift", 0),
            "time_encoded_dim": cfg.get("time_encoded_dim", 64),
            "s_audio_dim": cfg.get("s_audio_dim", 78),
            "s_audio_scale": cfg.get("s_audio_scale", cfg.get("latent_dim", 512) ** -0.5),  # scale the audio MFCC feat
            "s_emotion_dim": cfg.get("s_emotion_dim", 25),
            "l_embed_dim": cfg.get("l_embed_dim", 512),  # defined in diffusion prior
            "s_embed_dim": cfg.get("s_embed_dim", 512),  # defined in diffusion prior
            "personal_emb_dim": cfg.get("personal_emb_dim", 512),
            "s_3dmm_dim": cfg.get("s_3dmm_dim", 58),
            "concat": cfg.get("concat", "concat_first"),  # "concat_last"
            "condition_concat": cfg.get("condition_concat", "token_concat"),  # feat_concat | token_concat | cascade
            "guidance_scale": cfg.get("guidance_scale", 7.5),
            "l_latent_embed_drop_prob": cfg.get("l_latent_embed_drop_prob", 0.2),
            "l_personal_embed_drop_prob": cfg.get("l_personal_embed_drop_prob", 0.2),
            "s_audio_enc_drop_prob": cfg.get("s_audio_enc_drop_prob", 0.2),
            "s_latent_embed_drop_prob": cfg.get("s_latent_embed_drop_prob", 0.2),
            "s_3dmm_enc_drop_prob": cfg.get("s_3dmm_enc_drop_prob", 0.2),
            "s_emotion_enc_drop_prob": cfg.get("s_emotion_enc_drop_prob", 0.2),
            "past_l_3dmm_drop_prob": cfg.get("past_l_3dmm_drop_prob", 0.2),
            "use_past_frames": self.use_past_frames,
        }

        self.model = TransformerDenoiser(**self.init_params)
        self.mode = conf.mode
        if self.mode == "test":
            checkpoint_load(conf, self.model, device)
        resume = conf.trainer.get("resume", None)
        if resume is not None:  # during training
            checkpoint_resume(conf, self.model, device)

        self.decoder_diffusion = DecoderLatentDiffusion(
            conf.diffusion_decoder.scheduler,
            conf.diffusion_decoder.scheduler.num_train_timesteps,
            conf.diffusion_decoder.scheduler.num_inference_timesteps,
        )
        self.schedule_sampler = UniformSampler(self.decoder_diffusion)

        self.window_sliding = conf.diffusion_decoder.args.window_sliding
        self.k = conf.diffusion_decoder.scheduler.k  # k appropriate generations
        # whether use generated past listener 3dmm during training
        self.use_generated_pl = conf.diffusion_decoder.args.use_generated_pl
        self.same_latent_noise = conf.diffusion_decoder.scheduler.get("same_latent_noise", True)
        self.eta = conf.diffusion_decoder.scheduler.eta  # for DDIM sampling (inference)

    def _forward(
            self,
            speaker_audio=None,  # optional condition,
            speaker_emotion_input=None,  # optional condition, emotion,
            speaker_3dmm_input=None,  # optional condition, 3dmm,
            listener_emotion_input=None,  # condition, emotion, k appropriate,
            listener_3dmm_input=None,  # 1. condition, 3dmm, k appropriate,
            # 2. listener 3dmm feature, x_0 to be predicted, k appropriate,
            listener_personal_input=None,  # 3dmm or emotion, input to personal-specific encoder,
            speaker_audio_encodings=None,  # optional condition, speaker's audio encodings.
            speaker_latent_emb=None,  # optional condition, output from diffusion prior network,
            listener_latent_embed=None,  # k appropriate, output from prior diffusion.
    ):

        if self.mode in ["train", "val"]:
            speaker_audio_shifted = speaker_audio[:, :-self.window_size]  # (batch_size, seq_len, ...)
            batch_size, seq_len, d = speaker_audio_shifted.shape
            speaker_emotion_shifted = speaker_emotion_input[:, :-self.window_size]  # emotion
            speaker_3dmm_shifted = speaker_3dmm_input[:, :-self.window_size]  # 3dmm

            # condition, we got k appropriate reacted listeners for each speaker
            # shape: (batch_size * k, seq_len, ...)

            listener_emotion_shifted = listener_emotion_input[:,
                                       self.reaction_delay:(self.token_len - self.window_size + self.reaction_delay)]
            listener_3dmm_shifted = listener_3dmm_input[:,
                                    self.reaction_delay:(self.token_len - self.window_size + self.reaction_delay)]

            l_batch, _, _3dmm_dim = listener_3dmm_shifted.shape
            past_listener_3dmm = torch.cat((
                torch.zeros(size=(l_batch, (self.window_size - self.reaction_delay), _3dmm_dim)).to(
                    listener_3dmm_input.device),
                listener_3dmm_input[:, :(self.token_len + self.reaction_delay - 2 * self.window_size)],
            ), dim=1)  # past frames of listener's 3dmm coefficient

            # listener_emotion_shifted = listener_emotion_input[:, self.window_size:]  # emotion
            # listener_3dmm_shifted = listener_3dmm_input[:, self.window_size:]  # 3dmm
            # past_listener_3dmm = listener_3dmm_input[:, :-self.window_size]  # past 3dmm (shifted)

            # sample random window size for all batch elements
            window_start = torch.randint(0, seq_len - self.window_size, (1,), device=speaker_audio.device)
            if self.use_generated_pl:
                # we use generated past listener 3dmm as condition
                if window_start > (seq_len - 2 * self.window_size):
                    window_start = (seq_len - 2 * self.window_size)
            window_end = window_start + self.window_size

            # shape: (batch_size, window_size, ...)
            s_audio_selected = speaker_audio_shifted[:, window_start:window_end]
            s_emotion_selected = speaker_emotion_shifted[:, window_start:window_end]
            s_3dmm_selected = speaker_3dmm_shifted[:, window_start:window_end]

            # shape: (batch_size * k, window_size, ...)
            l_emotion_selected = listener_emotion_shifted[:, window_start:window_end]
            l_3dmm_selected = listener_3dmm_shifted[:, window_start:window_end]
            l_personal_input = listener_personal_input
            # past frames of 3dmm for listener
            past_listener_3dmmenc = past_listener_3dmm[:, window_start:window_end]

            x_start_selected = l_3dmm_selected  # (batch_size * k, window_size, 58)

            with torch.no_grad():
                # freeze personal-specific encoder to extract the personal embedding.
                personal_embed = self.person_encoder.forward(l_personal_input)[0].unsqueeze(1)
                # (batch_size * k, 1, ...)

                # freeze latent RNN_VAE embedder to extract listener latent embedding
                l_latent_embed = self.latent_embedder.encode(l_emotion_selected).unsqueeze(1)

                s_audio_encodings = self.audio_encoder._encode(s_audio_selected)
                s_audio_encodings = s_audio_encodings.repeat_interleave(self.k, dim=0)

                # freeze latent RNN_VAE embedder to extract speaker latent embedding
                s_latent_embed = self.latent_embedder.encode(s_emotion_selected).unsqueeze(1)
                # we repeat the obs 'k' times for k appropriate reactions
                s_latent_embed = s_latent_embed.repeat_interleave(self.k, dim=0)  # shape: (batch_size * k, 1, dim)

                # TODO: optional condition, speaker 3dmm encodings
                if self.encode_3dmm:  # whether encode speaker 3dmm features (encodings)
                    s_3dmm_selected = self.latent_3dmm_embedder.get_encodings(s_3dmm_selected)
                s_3dmm_encodings = s_3dmm_selected.repeat_interleave(self.k, dim=0)
                # shape: (batch_size * k, window_size, ...)

                # TODO: optional condition, speaker emotion encodings
                if self.encode_emotion:  # whether encode speaker emotion features (encodings)
                    s_emotion_selected = self.latent_embedder.get_encodings(s_emotion_selected)
                s_emotion_encodings = s_emotion_selected.repeat_interleave(self.k, dim=0)
                # shape: (batch_size * k, window_size, ...)

                model_kwargs = {"listener_latent_embed": l_latent_embed,
                                "listener_personal_embed": personal_embed,
                                "speaker_audio_encodings": s_audio_encodings,
                                "speaker_latent_embed": s_latent_embed,
                                "speaker_3dmm_encodings": s_3dmm_encodings,
                                "speaker_emotion_encodings": s_emotion_encodings,
                                "past_listener_3dmmenc": past_listener_3dmmenc}

            t, _ = self.schedule_sampler.sample(batch_size, x_start_selected.device)
            # timestep, with shape: (batch_size * k,)
            timesteps = t.long()

            output_whole = self.decoder_diffusion.denoise(self.model, x_start_selected, timesteps,
                                                          model_kwargs=model_kwargs)

            if self.use_generated_pl:  # we do feed-forward again, but based on conditions after a period (window_size).
                window_start = window_start + self.window_size
                window_end = window_start + self.window_size

                # shape: (batch_size, window_size, ...)
                s_audio_selected = speaker_audio_shifted[:, window_start:window_end]
                s_emotion_selected = speaker_emotion_shifted[:, window_start:window_end]
                s_3dmm_selected = speaker_3dmm_shifted[:, window_start:window_end]
                # shape: (batch_size * k, window_size, ...)
                l_emotion_selected = listener_emotion_shifted[:, window_start:window_end]
                l_3dmm_selected = listener_3dmm_shifted[:, window_start:window_end]

                # past frames of 3dmm for listener
                past_listener_3dmmenc = output_whole['prediction_3dmm'].detach()  # detach
                _, _, l, d = past_listener_3dmmenc.shape
                past_listener_3dmmenc = past_listener_3dmmenc.reshape(-1, l, d)

                x_start_selected = l_3dmm_selected  # (batch_size * k, window_size, 58)

                with torch.no_grad():
                    # freeze latent RNN_VAE embedder to extract listener latent embedding
                    l_latent_embed = self.latent_embedder.encode(l_emotion_selected).unsqueeze(1)

                    s_audio_encodings = self.audio_encoder._encode(s_audio_selected)
                    s_audio_encodings = s_audio_encodings.repeat_interleave(self.k, dim=0)

                    # freeze latent RNN_VAE embedder to extract speaker latent embedding
                    s_latent_embed = self.latent_embedder.encode(s_emotion_selected).unsqueeze(1)
                    # we repeat the obs 'k' times for k appropriate reactions
                    s_latent_embed = s_latent_embed.repeat_interleave(self.k, dim=0)  # shape: (batch_size * k, 1, dim)

                    # TODO: optional condition, speaker 3dmm encodings
                    if self.encode_3dmm:  # whether encode speaker 3dmm features (encodings)
                        s_3dmm_selected = self.latent_3dmm_embedder.get_encodings(s_3dmm_selected)
                    s_3dmm_encodings = s_3dmm_selected.repeat_interleave(self.k, dim=0)
                    # shape: (batch_size * k, window_size, ...)

                    # TODO: optional condition, speaker emotion encodings
                    if self.encode_emotion:  # whether encode speaker emotion features (encodings)
                        s_emotion_selected = self.latent_embedder.get_encodings(s_emotion_selected)
                    s_emotion_encodings = s_emotion_selected.repeat_interleave(self.k, dim=0)
                    # shape: (batch_size * k, window_size, ...)

                    model_kwargs = {"listener_latent_embed": l_latent_embed,
                                    "listener_personal_embed": personal_embed,
                                    "speaker_audio_encodings": s_audio_encodings,
                                    "speaker_latent_embed": s_latent_embed,
                                    "speaker_3dmm_encodings": s_3dmm_encodings,
                                    "speaker_emotion_encodings": s_emotion_encodings,
                                    "past_listener_3dmmenc": past_listener_3dmmenc}

                    t, _ = self.schedule_sampler.sample(batch_size, x_start_selected.device)
                    # timestep, with shape: (batch_size * k,)
                    timesteps = t.long()

                    output = self.decoder_diffusion.denoise(self.model, x_start_selected, timesteps,
                                                            model_kwargs=model_kwargs)

                    prediction_3dmm = torch.cat((
                        output_whole['prediction_3dmm'], output['prediction_3dmm']
                    ), dim=-2)  # (B, k, 2 * window_size, d)
                    target_3dmm = torch.cat((
                        output_whole['target_3dmm'], output['target_3dmm']
                    ), dim=-2)

                    output_whole = {'prediction_3dmm': prediction_3dmm,
                                    'target_3dmm': target_3dmm}

            return output_whole

        else:
            diff_batch = speaker_latent_emb.shape[0]
            seq_len = (self.token_len - self.window_size) // self.window_sliding + 1  # num_windows
            batch_size = diff_batch // (seq_len * self.k)

            with torch.no_grad():
                # TODO: (optional) speaker emotion encodings (and/or speaker 3dmm encodings)
                if self.encode_3dmm:
                    speaker_3dmm_input = self.latent_3dmm_embedder.get_encodings(speaker_3dmm_input)
                if self.encode_emotion:
                    speaker_emotion_input = self.latent_embedder.get_encodings(speaker_emotion_input)

                # freeze personal-specific encoder to extract the personal embedding.
                personal_embed, _ = self.person_encoder.forward(listener_personal_input)  # (batch_size * k, ...)

            dim = personal_embed.shape[-1]
            personal_embed = personal_embed.reshape(-1, self.k, dim)
            personal_embed = personal_embed.repeat(1, seq_len, 1)
            personal_embed = personal_embed.reshape(-1, dim)  # (batch_size * n * k, ...)

            if self.use_past_frames:  # past listener 3dmm (or emotion)
                listener_latent_embed = listener_latent_embed.reshape(batch_size, seq_len, self.k, 1, -1)
                personal_embed = personal_embed.reshape(batch_size, seq_len, self.k, -1)
                speaker_latent_emb = speaker_latent_emb.reshape(batch_size, seq_len, self.k, 1, -1)
                speaker_audio_encodings = speaker_audio_encodings.reshape(
                    batch_size, seq_len, self.khape(
                    batch_size, seq_len, self.k, self.window_size, -1), self.window_size, -1)
                speaker_3dmm_encodings = speaker_3dmm_input.reshape(
                    batch_size, seq_len, self.k, self.window_size, -1)
                speaker_emotion_encodings = speaker_emotion_input.res

                past_listener_3dmmenc = torch.zeros(
                    size=(batch_size * self.k, self.window_size, self._3dmm_dim)
                ).to(device=listener_latent_embed.device)
                output_listener_3dmm = torch.zeros(
                    size=(seq_len, batch_size, self.k, self.window_size, self._3dmm_dim)
                ).to(device=listener_latent_embed.device)

                # TODO: we utilize same initial noise to generate all reaction window clip (50 frames)
                #   for whole reaction sequence (750 frames).
                if self.same_latent_noise:
                    noise = torch.randn_like(past_listener_3dmmenc).to(past_listener_3dmmenc.device)
                else:
                    noise = None

                for i in range(seq_len):
                    model_kwargs = {
                        # condition, generated from diffusion prior network; shape: (batch_size * k, 1, dim)
                        "listener_latent_embed": listener_latent_embed[:, i].reshape(batch_size * self.k, 1, -1),
                        # condition, generated from personal-specific encoder; shape: (batch_size * k, 1, dim)
                        "listener_personal_embed": personal_embed[:, i].reshape(batch_size * self.k, 1, -1),
                        # optional condition, obtained from model_kwargs; shape: (batch_size * k, window_size, 78)
                        "speaker_audio_encodings": speaker_audio_encodings[:, i].reshape(
                            batch_size * self.k, self.window_size, -1),
                        # optional condition, obtained from model_kwargs; shape: (batch_size * k, 1, dim)
                        "speaker_latent_embed": speaker_latent_emb[:, i].reshape(batch_size * self.k, 1, -1),
                        "speaker_3dmm_encodings": speaker_3dmm_encodings[:, i].reshape(
                            batch_size * self.k, self.window_size, -1),
                        "speaker_emotion_encodings": speaker_emotion_encodings[:, i].reshape(
                            batch_size * self.k, self.window_size, -1),
                        "past_listener_3dmmenc": past_listener_3dmmenc,
                    }

                    with torch.no_grad():
                        output = [output for output in self.decoder_diffusion.ddim_sample_loop_progressive(
                            matcher=self,
                            model=self.model,
                            noise=noise,
                            model_kwargs=model_kwargs,
                            shape=past_listener_3dmmenc.shape,  # (batch_size * k, window_size, 58)
                        )][-1]  # get last output, and used as past_listener_3dmmenc

                    past_listener_3dmmenc = output["sample_enc"]
                    output_listener_3dmm[i, :, :, :, :] = output["sample_enc"].reshape(
                        batch_size, self.k, self.window_size, self._3dmm_dim
                    )
                output_listener_3dmm = output_listener_3dmm.permute(1, 0, 2, 3, 4).contiguous()

            else:
                model_kwargs = {
                    # condition, generated from diffusion prior network; shape: (batch_size * n * k, 1, dim)
                    "listener_latent_embed": listener_latent_embed,
                    # condition, generated from personal-specific encoder; shape: (batch_size * n * k, 1, dim)
                    "listener_personal_embed": personal_embed.unsqueeze(-2),
                    # optional condition, obtained from model_kwargs; shape: (batch_size * n * k, window_size, 78)
                    "speaker_audio_encodings": speaker_audio_encodings,
                    # optional condition, obtained from model_kwargs; shape: (batch_size * n * k, 1, dim)
                    "speaker_latent_embed": speaker_latent_emb,
                    # optional condition, obtained from model_kwargs; shape: (batch_size * n * k, window_size, 58)
                    "speaker_3dmm_encodings": speaker_3dmm_input,
                    # optional condition, shape: (batch_size * n * k, window_size, 25)
                    "speaker_emotion_encodings": speaker_emotion_input,
                }

                # TODO: debug: we utilize same initial noise to generate all reaction window clip (50 frames)
                #   for whole reaction sequence (750 frames).
                if self.same_latent_noise:
                    noise = (torch.randn(size=(batch_size, 1, self.k, self.window_size, self._3dmm_dim))
                             .to(speaker_3dmm_input.device))
                    noise = noise.expand(-1, seq_len, -1, -1, -1).reshape(-1, self.window_size, self._3dmm_dim)
                else:
                    noise = None

                with torch.no_grad():
                    output = [output for output in self.decoder_diffusion.ddim_sample_loop_progressive(
                        matcher=self,
                        model=self.model,
                        noise=noise,
                        model_kwargs=model_kwargs,
                        eta=self.eta,
                        shape=(diff_batch, self.window_size, self._3dmm_dim),
                    )][-1]  # get last output
                output_listener_3dmm = output["sample_enc"]

            # ==> (batch_size, n, k, window_size, dim=58)
            output_listener_3dmm = output_listener_3dmm.reshape(
                batch_size, seq_len, self.k, self.window_size, self._3dmm_dim)
            # ==> (batch_size, k, n, window_size, dim=58)
            output_listener_3dmm = output_listener_3dmm.transpose(1, 2).contiguous()

            # TODO: 'undo' the window sliding done before
            _output_listener_3dmm = (torch.randn(size=(batch_size, self.k, 0, self._3dmm_dim))
                                     .to(output_listener_3dmm.device))

            for i in range(seq_len):
                if i == 0:
                    _output_listener_3dmm = torch.cat((_output_listener_3dmm,
                                                       output_listener_3dmm[:, :, i, :, :]), dim=-2)
                else:
                    _output_listener_3dmm = torch.cat((_output_listener_3dmm,
                                                       output_listener_3dmm[:, :, i, -self.window_sliding:, :]), dim=-2)

            # ==> (batch_size, k, (n*window_size)==seq_len, dim=58)
            # output_listener_3dmm = _output_listener_3dmm.reshape(batch_size, self.k, -1, self._3dmm_dim)
            output_listener_3dmm = _output_listener_3dmm
            output_whole = {"prediction_3dmm": output_listener_3dmm}

            return output_whole

    def forward(self, **kwargs):
        return self._forward(**kwargs)


class LatentMatcher(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.mode = cfg.mode  # train (val) or test
        self.diffusion_prior = PriorLatentMatcher(cfg, device=device)
        self.diffusion_decoder = DecoderLatentMatcher(cfg, device=device)

    def forward(
            self,
            speaker_audio=None,  # optional condition, for training.
            speaker_emotion_input=None,  # optional condition, emotion, for training.
            speaker_3dmm_input=None,  # optional condition, 3dmm, for training.
            listener_emotion_input=None,  # condition, emotion, k appropriate, for training.
            listener_3dmm_input=None,  # 1. condition, 3dmm, k appropriate, for training.
            # 2. listener 3dmm feature, x_0 (ground-truth) to be predicted, k appropriate.
            listener_personal_input=None,  # 3dmm or emotion, input to personal-specific encoder, for training.
    ):
        if self.mode in ["train", "val"]:  # train | val
            # diffusion prior
            output_prior = self.diffusion_prior.forward(
                speaker_audio=speaker_audio,  # shape: (batch_size, seq_len, audio_dim=78)
                speaker_emotion_input=speaker_emotion_input,  # shape: (batch_size, seq_len, emotion_dim=25)
                speaker_3dmm_input=speaker_3dmm_input,  # shape: (batch_size, seq_len, 3dmm_dim=58)
                listener_emotion_input=listener_emotion_input,  # shape: (batch_size * k, seq_len, emotion_dim=25)
                listener_3dmm_input=listener_3dmm_input  # shape: (batch_size * k, seq_len, 3dmm_dim=58)
            )
            # output_prior: type dict
            # output_prior["encoded_prediction"], shape: (batch_size, k, 1, encoded_dim)
            # output_prior["encoded_target"], shape: (batch_size, k, 1, encoded_dim)

            # diffusion decoder
            output_decoder = self.diffusion_decoder.forward(
                speaker_audio=speaker_audio,  # shape: (batch_size, seq_len, audio_dim=78)
                speaker_emotion_input=speaker_emotion_input,  # shape: (batch_size, seq_len, emotion_dim=25)
                speaker_3dmm_input=speaker_3dmm_input,  # shape: (batch_size, seq_len, 3dmm_dim=58)
                listener_emotion_input=listener_emotion_input,  # shape: (batch_size * k, seq_len, emotion_dim=25)
                listener_3dmm_input=listener_3dmm_input,  # shape: (batch_size * k, seq_len, 3dmm_dim=58)
                listener_personal_input=listener_personal_input,  # shape: (batch_size * k, seq_len, 3dmm_dim=58)
            )
            # output_decoder: type dict
            # output_decoder["prediction_3dmm"], shape: (batch_size, k, seq_len, 3dmm_dim==58)
            # output_decoder["target_3dmm"], shape: (batch_size , k, seq_len, 3dmm_dim==58)

        else:  # test
            # diffusion prior

            output_prior, model_kwargs = self.diffusion_prior.forward(
                speaker_audio=speaker_audio,  # shape: (batch_size, seq_len, audio_dim=78)
                speaker_emotion_input=speaker_emotion_input,  # shape: (batch_size, seq_len, emotion_dim=25)
                speaker_3dmm_input=speaker_3dmm_input,  # shape: (batch_size, seq_len, 3dmm_dim=58)
                listener_emotion_input=listener_emotion_input,  # shape: (batch_size * k, seq_len, emotion_dim=25)
                listener_3dmm_input=listener_3dmm_input  # shape: (batch_size * k, seq_len, 3dmm_dim=58)
            )

            # output_prior.shape: (batch_size * n * k, 1, encoded_dim) (e.g. listener_latent_embed)
            # model_kwargs:
            speaker_latent_embed = model_kwargs["speaker_latent_emb"]  # (batch_size * n * k, 1, encoded_dim==512)
            speaker_audio_encodings = model_kwargs["speaker_audio_encodings"]  # (batch_size * n * k, window_size, 78)
            speaker_3dmm_encodings = model_kwargs["speaker_3dmm_encodings"]  # (batch_size * n * k, window_size, 58)
            speaker_emotion_encodings = model_kwargs[
                "speaker_emotion_encodings"]  # (batch_size * n * k, window_size, 25)

            # diffusion decoder
            output_decoder = self.diffusion_decoder.forward(
                # listener_3dmm_input=listener_3dmm_input,  # shape: (batch_size * k, seq_len, 3dmm_dim=58)
                speaker_emotion_input=speaker_emotion_encodings,  # shape: (batch_size * n * k, window_size, emo_dim=25)
                speaker_3dmm_input=speaker_3dmm_encodings,  # shape: (batch_size * n * k, window_size, 3dmm_dim=58)
                listener_personal_input=listener_personal_input,  # shape: (batch_size * k, seq_len, 3dmm_dim=58)
                speaker_audio_encodings=speaker_audio_encodings,  # shape: (batch_size * n * k, window_size, 78)
                speaker_latent_emb=speaker_latent_embed,  # shape: (batch_size * n * k, 1, encoded_dim)
                listener_latent_embed=output_prior,  # shape: (batch_size * n * k, 1, encoded_dim)
            )

            # output_prior: tensor with shape (batch_size * n * k, encoded_dim)
            # output_decoder['prediction_3dmm']: tensor with shape (batch_size, k, (n*window_size)==750, 3dmm_dim==58)
            # Finally obtain k appropriate personalized facial reactions (3dmm)

        return output_prior, output_decoder

    def obtain_shapes(self, modified_layers):
        shape_dict = {}
        for name, module in self.named_modules():
            if name in modified_layers:
                if 'linear' in name or 'to_3dmm' in name:
                    shape_dict[name] = torch.tensor(module.weight.size())
                elif 'multihead_attn' in name or 'self_attn' in name:
                    shape_dict[name] = torch.tensor(module.in_proj_weight.size())
            else:
                continue
        return shape_dict
