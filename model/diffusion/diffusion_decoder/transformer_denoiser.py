import os
from typing import Dict, List
import math
import torch
import torch.nn as nn
from torch import Tensor
from model.diffusion.utils.util import tgt_biased_mask, memory_biased_mask
from model.diffusion.operator.embeddings import (TimestepEmbedding, Timesteps)
from model.diffusion.operator.position_encoding import build_position_encoding
from model.diffusion.operator.cross_attention import (SkipTransformerEncoder,
                                                      TransformerDecoder,
                                                      TransformerDecoderLayer,
                                                      TransformerEncoder,
                                                      TransformerEncoderLayer)


def lengths_to_mask(lengths: List[int],
                    device: torch.device,
                    max_len: int = None) -> Tensor:
    lengths = torch.tensor(lengths, device=device)
    max_len = max_len if max_len else max(lengths)
    mask = torch.arange(max_len, device=device).expand(
        len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


def timestep_embedding(timesteps, dim, max_period=10000, dtype=torch.float32):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=dtype) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].type(dtype) * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TransformerDenoiser(nn.Module):
    def __init__(self,
                 reaction_delay: int = 8,
                 token_len: int = 750,
                 encode_emotion: bool = False,
                 encode_3dmm: bool = False,
                 ablation_skip_connection: bool = True,
                 nfeats: int = 58,
                 latent_dim: int = 512,
                 ff_size: int = 1024,
                 num_layers: int = 7,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 normalize_before: bool = False,
                 activation: str = "gelu",
                 flip_sin_to_cos: bool = True,
                 return_intermediate_dec: bool = False,
                 position_embedding: str = "sine",
                 arch: str = "trans_enc",  # or trans_dec
                 use_attn_mask: bool = True,
                 period: int = 50,
                 freq_shift: int = 0,
                 time_encoded_dim: int = 64,
                 s_audio_dim: int = 78,  # encoded dim of speaker's audio feature
                 s_audio_scale: float = 1.0,  # scale of speaker's audio feature
                 s_emotion_dim: int = 25,  # encoded dim of speaker's emotion encodings
                 l_embed_dim: int = 512,  # encoded dim of listener's embedding
                 s_embed_dim: int = 512,  # encoded dim of speaker's embedding
                 personal_emb_dim: int = 512,
                 s_3dmm_dim: int = 58,  # encoded dim of speaker 3dmm feature
                 concat: str = "concat_first",  # concat_first or concat_last
                 # we use three strategies to do the interactions between input and conditions.
                 condition_concat: str = "token_concat",  # feat_concat | token_concat | cascade
                 guidance_scale: float = 7.5,
                 # condition drop probability
                 l_latent_embed_drop_prob: float = 0.2,  # listener_latent_embed
                 l_personal_embed_drop_prob: float = 0.2,  # listener_personal_embed
                 s_audio_enc_drop_prob: float = 0.2,  # speaker_audio_encodings
                 s_latent_embed_drop_prob: float = 0.2,  # speaker_latent_embed
                 s_3dmm_enc_drop_prob: float = 0.2,  # speaker_3dmm_encodings
                 s_emotion_enc_drop_prob: float = 0.2,  # speaker_emotion_encodings
                 past_l_3dmm_drop_prob: float = 0.2,  # past_listener_emotion
                 use_past_frames: float = 0.2,
                 **kwargs) -> None:

        super().__init__()

        self.reaction_delay = reaction_delay
        self.encode_emotion = encode_emotion
        self.encode_3dmm = encode_3dmm
        self.s_audio_scale = s_audio_scale
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.ablation_skip_connection = ablation_skip_connection
        self.arch = arch
        self.concat = concat
        self.condition_concat = condition_concat
        # for classifier-free guidance
        self.guidance_scale = guidance_scale
        # condition drop probability
        self.l_latent_embed_drop_prob = l_latent_embed_drop_prob
        self.l_personal_embed_drop_prob = l_personal_embed_drop_prob
        self.s_audio_enc_drop_prob = s_audio_enc_drop_prob
        self.s_latent_embed_drop_prob = s_latent_embed_drop_prob
        self.s_3dmm_enc_drop_prob = s_3dmm_enc_drop_prob
        self.s_emotion_enc_drop_prob = s_emotion_enc_drop_prob
        self.past_l_3dmm_drop_prob = past_l_3dmm_drop_prob
        self.use_past_frames = use_past_frames

        # project between 3dmm output feat and 3dmm latent embedding
        self.to_3dmm_embed = nn.Linear(nfeats, self.latent_dim) if nfeats != self.latent_dim else nn.Identity()
        self.to_3dmm_feat = nn.Linear(self.latent_dim, nfeats) if self.latent_dim != nfeats else nn.Identity()

        # project time to latent_dim
        self.time_proj = Timesteps(time_encoded_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(time_encoded_dim, self.latent_dim)

        self.speaker_latent_proj = nn.Sequential(nn.ReLU(), nn.Linear(s_embed_dim, self.latent_dim)) \
            if s_embed_dim != self.latent_dim else nn.Identity()  # TODO: why relu

        self.listener_latent_proj = nn.Sequential(nn.ReLU(), nn.Linear(l_embed_dim, self.latent_dim)) \
            if l_embed_dim != self.latent_dim else nn.Identity()  # TODO: why relu

        self.speaker_audio_proj = nn.Linear(s_audio_dim, self.latent_dim) \
            if s_audio_dim != self.latent_dim else nn.Identity()

        if self.encode_3dmm:  # assume dimension of encoded 3dmm equals latent_dim
            self.speaker_3dmm_proj = nn.Identity()
        else:
            assert s_3dmm_dim != self.latent_dim, "wrong dimension of raw 3dmm features."
            self.speaker_3dmm_proj = nn.Linear(s_3dmm_dim, self.latent_dim)

        if self.encode_emotion:  # assume dimension of encoded emotion equals latent_dim
            self.speaker_emotion_proj = nn.Identity()
        else:
            assert s_emotion_dim != self.latent_dim, "wrong dimension of raw emotion features."
            self.speaker_emotion_proj = nn.Linear(s_emotion_dim, self.latent_dim)

        self.listener_personal_proj = nn.Linear(personal_emb_dim, self.latent_dim) \
            if personal_emb_dim != self.latent_dim else nn.Identity()

        self.listener_3dmm_proj = nn.Linear(nfeats, self.latent_dim) \
            if nfeats != self.latent_dim else nn.Identity()

        # TODO: for specific conditions.
        self.position_embedding = position_embedding
        self.query_pos = build_position_encoding(
            self.latent_dim, batch_first=True, position_embedding=position_embedding)
        self.mem_pos = build_position_encoding(
            self.latent_dim, batch_first=True, position_embedding=position_embedding)

        self.use_attn_mask = use_attn_mask

        self.period = period
        if use_attn_mask:
            self.tgt_biased_mask = tgt_biased_mask(
                n_head=num_heads,
                max_seq_len=token_len,
                period=period,
            )

        # TODO: we use three strategies to do the interactions between input and conditions.
        # we concat conditions (including: speaker 3dmm, speaker audio, past_listener_motion) along last dimension.
        self.condition_proj = nn.Linear(self.latent_dim * 3, self.latent_dim) \
            if self.condition_concat == 'feat_concat' else nn.Identity()

        # define our transformer decoder layer
        decoder_layer = TransformerDecoderLayer(
            d_model=self.latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            batch_first=True,
            activation=activation,
            normalize_before=normalize_before,
        )

        if self.arch == "trans_enc":  # Transformer Encoder
            if self.ablation_skip_connection:
                # use DETR transformer
                encoder_layer = TransformerEncoderLayer(
                    self.latent_dim,
                    num_heads,
                    ff_size,
                    dropout,
                    activation,
                    normalize_before,
                )
                encoder_norm = nn.LayerNorm(self.latent_dim)
                self.encoder = SkipTransformerEncoder(encoder_layer,
                                                      num_layers, encoder_norm)
            else:
                # use torch transformer
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=self.latent_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation)
                self.encoder = nn.TransformerEncoder(encoder_layer,
                                                     num_layers=num_layers)

        elif self.arch == "trans_dec":  # Transformer Decoder
            self.decoder = TransformerDecoder(
                decoder_layer,
                num_layers,
                nn.LayerNorm(self.latent_dim),
                return_intermediate=return_intermediate_dec,
            )

        else:
            raise ValueError(f"Not supported architechure{self.arch}!")

        if self.condition_concat == 'cascade':
            if self.use_past_frames:
                num_layer_ns2pl = 2
                num_layer_ns2sm = 2
                num_layer_ns2sa = 2

                # noised sample <--> past listener 3dmm
                self.transformer_fusion_ns2pl = TransformerDecoder(
                    decoder_layer, num_layer_ns2pl, nn.LayerNorm(self.latent_dim),
                    return_intermediate=return_intermediate_dec,
                )

            else:
                num_layer_ns2sa = 3
                num_layer_ns2sm = 3

            # noised sample <--> speaker audio
            self.transformer_fusion_ns2sa = TransformerDecoder(
                decoder_layer, num_layer_ns2sa, nn.LayerNorm(self.latent_dim),
                return_intermediate=return_intermediate_dec,
            )

            # noised sample <--> speaker 3dmm
            self.transformer_fusion_ns2sm = TransformerDecoder(
                decoder_layer, num_layer_ns2sm, nn.LayerNorm(self.latent_dim),
                return_intermediate=return_intermediate_dec,
            )

            # final interaction
            self.transformer_fusion_final = TransformerDecoder(
                decoder_layer, 1, nn.LayerNorm(self.latent_dim),
                return_intermediate=return_intermediate_dec,
            )

    def mask_cond(self, feature, mode='test', drop_prob=0.0):  # train or test
        bs, _, _ = feature.shape

        # classifier-free guidance
        if mode == 'test':  # inference
            uncond_feat, con_feat = feature.chunk(2)
            # con_feat = con_feat
            uncond_feat = torch.zeros_like(uncond_feat)
            feature = torch.cat((uncond_feat, con_feat), dim=0)

        else:  # train or val mode
            if drop_prob > 0.0:
                mask = torch.bernoulli(
                    torch.ones(bs, device=feature.device) *
                    drop_prob).view(
                    bs, 1, 1)  # 1-> use null_cond, 0-> use real cond
                feature = feature * (1.0 - mask)

        return feature

    def get_model_kwargs(
            self,
            bs,
            mode,
            sample,
            model_kwargs,
    ):  # ALL CONDITIONS:

        listener_latent_embed = model_kwargs.get('listener_latent_embed')
        if listener_latent_embed is None or self.l_latent_embed_drop_prob >= 1.0:
            listener_latent_embed = torch.zeros(size=(bs, 0, self.latent_dim)).to(sample.device)
        else:
            # [1, bs, encoded_dim] => [1, bs, latent_dim]
            listener_latent_embed = self.listener_latent_proj(listener_latent_embed)
            listener_latent_embed = self.mask_cond(listener_latent_embed, mode, self.l_latent_embed_drop_prob)
        # listener_latent_embed = listener_latent_embed.permute(1, 0, 2).contiguous()

        listener_personal_embed = model_kwargs.get('listener_personal_embed')
        if listener_personal_embed is None or self.l_personal_embed_drop_prob >= 1.0:
            listener_personal_embed = torch.zeros(size=(bs, 0, self.latent_dim)).to(sample.device)
        # TODO: we use listener_personal_embed to rewrite the weight.
        else:
            listener_personal_embed = self.listener_personal_proj(listener_personal_embed)
            listener_personal_embed = self.mask_cond(listener_personal_embed, mode, self.l_personal_embed_drop_prob)
        # listener_personal_embed = listener_personal_embed.permute(1, 0, 2).contiguous()

        speaker_audio_encodings = model_kwargs.get('speaker_audio_encodings')
        if speaker_audio_encodings is None or self.s_audio_enc_drop_prob >= 1.0:
            speaker_audio_encodings = torch.zeros(size=(bs, 0, self.latent_dim)).to(sample.device)
        else:
            speaker_audio_encodings = self.speaker_audio_proj(speaker_audio_encodings)
            speaker_audio_encodings = self.mask_cond(speaker_audio_encodings, mode, self.s_audio_enc_drop_prob)
            # TODO: we scale (adjust the magnitude of) speaker_audio_encodings using a 'self.scale'.
            speaker_audio_encodings = self.s_audio_scale * speaker_audio_encodings
        # speaker_audio_encodings = speaker_audio_encodings.permute(1, 0, 2).contiguous()

        speaker_latent_embed = model_kwargs.get('speaker_latent_embed')
        if speaker_latent_embed is None or self.s_latent_embed_drop_prob >= 1.0:
            speaker_latent_embed = torch.zeros(size=(bs, 0, self.latent_dim)).to(sample.device)
        else:
            speaker_latent_embed = self.speaker_latent_proj(speaker_latent_embed)
            speaker_latent_embed = self.mask_cond(speaker_latent_embed, mode, self.s_latent_embed_drop_prob)
        # speaker_latent_embed = speaker_latent_embed.permute(1, 0, 2).contiguous()

        speaker_3dmm_encodings = model_kwargs.get("speaker_3dmm_encodings")
        if speaker_3dmm_encodings is None or self.s_3dmm_enc_drop_prob >= 1.0:
            speaker_3dmm_encodings = torch.zeros(size=(bs, 0, self.latent_dim)).to(sample.device)
        else:
            speaker_3dmm_encodings = self.speaker_3dmm_proj(speaker_3dmm_encodings)
            speaker_3dmm_encodings = self.mask_cond(speaker_3dmm_encodings, mode, self.s_3dmm_enc_drop_prob)
        # speaker_3dmm_encodings = speaker_3dmm_encodings.permute(1, 0, 2).contiguous()

        speaker_emotion_encodings = model_kwargs.get("speaker_emotion_encodings")
        if speaker_emotion_encodings is None or self.s_emotion_enc_drop_prob >= 1.0:
            speaker_emotion_encodings = torch.zeros(size=(bs, 0, self.latent_dim)).to(sample.device)
        else:
            speaker_emotion_encodings = self.speaker_emotion_proj(speaker_emotion_encodings)
            speaker_emotion_encodings = self.mask_cond(speaker_emotion_encodings, mode, self.s_emotion_enc_drop_prob)
        # speaker_emotion_encodings = speaker_emotion_encodings.permute(1, 0, 2).contiguous()

        past_listener_3dmmenc = model_kwargs.get('past_listener_3dmmenc')
        if past_listener_3dmmenc is None or self.past_l_3dmm_drop_prob >= 1.0:
            past_listener_3dmmenc = torch.zeros(size=(bs, 0, self.latent_dim)).to(sample.device)
        else:
            # TODO: use the shared MLP with speaker 3dmm or use its own unique MLP.
            past_listener_3dmmenc = self.listener_3dmm_proj(past_listener_3dmmenc)  # we use its unique MLP.
            # past_listener_3dmmenc = self.to_3dmm_embed(past_listener_3dmmenc)  # we use a shared MLP.
            # TODO: or get its embedding output from auto-encoder.
            past_listener_3dmmenc = self.mask_cond(past_listener_3dmmenc, mode, self.past_l_3dmm_drop_prob)
        # past_listener_3dmmenc = past_listener_3dmmenc.permute(1, 0, 2).contiguous()

        return (listener_latent_embed,
                listener_personal_embed,
                speaker_audio_encodings,
                speaker_latent_embed,
                speaker_3dmm_encodings,
                speaker_emotion_encodings,
                past_listener_3dmmenc)

    def _forward(
            self,
            sample,
            time_embed,
            listener_latent_embed,
            listener_personal_embed,
            speaker_audio_encodings,
            speaker_latent_embed,
            speaker_3dmm_encodings,
            speaker_emotion_encodings,
            past_listener_3dmmenc,
    ):

        # We simply concat all optional conditions.
        # emb_latent = torch.cat((
        #     time_embed,
        #     speaker_audio_encodings,  # optional condition,
        #     speaker_3dmm_encodings,  # optional condition,
        #     speaker_emotion_encodings,  # optional condition,
        #     speaker_latent_embed,  # optional condition,
        #     listener_latent_embed,
        #     past_listener_3dmmenc,
        #     listener_personal_embed,
        # ), dim=1)

        # map noised input sample to latent dim
        sample = self.to_3dmm_embed(sample)

        # attention masks in cross-attention
        if self.use_attn_mask:
            B, seq_len, _ = sample.shape

            # tgt_mask for self-attn
            tgt_mask = self.tgt_biased_mask[:, :seq_len, :seq_len].clone().detach().to(
                device=sample.device).repeat(B, 1, 1)

            # only for timestep embedding
            time_embed_mask = torch.zeros_like(tgt_mask[:, :, :1]).to(sample.device)
            # shape: (B, seq_len, 1)

            # memory_mask for cross-attn
            memory_mask = memory_biased_mask(
                n_head=self.num_heads,
                window_size=seq_len,
                max_seq_len=seq_len,
                period=self.period).clone().detach().to(
                device=sample.device).repeat(B, 1, 1)

            # consider the reaction delay we set
            delay_mask = torch.zeros_like(tgt_mask[:, :, :self.reaction_delay]).to(sample.device)
            memory_mask = torch.cat((
                delay_mask, memory_mask[:, :, :-self.reaction_delay]
            ), dim=-1)  # [B, seq_len, 1+seq_len]

        else:
            tgt_mask = None
            memory_mask = None

        # TODO: we use three strategies to do the interactions between input and conditions.
        #  (1) feat_concat; (2) token_concat; (3) cascade;

        # 1. we concat conditions (including: speaker 3dmm, speaker audio, past_listener_motion) along feature dim.
        if self.condition_concat == 'feat_concat':
            if past_listener_3dmmenc.shape[1] == 0:
                past_listener_3dmmenc = torch.zeros_like(speaker_3dmm_encodings).to(
                    speaker_3dmm_encodings.device)

            emb_latent = torch.cat((
                speaker_audio_encodings,
                speaker_3dmm_encodings,
                past_listener_3dmmenc,
            ), dim=-1)
            emb_latent = self.condition_proj(emb_latent)  # (seq_len, bs, latent_dim)
            # append time embedding
            emb_latent = torch.cat((time_embed, emb_latent), dim=1)

            embed_seq_len = emb_latent.shape[0]
            if self.arch == "trans_enc":
                if self.concat == "concat_first":
                    xseq = torch.cat((emb_latent, sample), dim=1)
                    xseq = self.query_pos(xseq)
                    tokens = self.encoder(xseq)
                    sample = tokens[embed_seq_len:]
                elif self.concat == "concat_last":
                    xseq = torch.cat((sample, emb_latent), dim=1)
                    xseq = self.query_pos(xseq)
                    tokens = self.encoder(xseq)
                    sample = tokens[:embed_seq_len]
                else:
                    raise NotImplementedError("{self.concat} is not supported.")
            elif self.arch == "trans_dec":
                # tgt    - [L~, bs, latent_dim]
                # memory - [token_num, bs, latent_dim]
                sample = self.query_pos(sample)
                if self.position_embedding != 'none':
                    emb_latent = self.mem_pos(emb_latent)

                # if self.use_attn_mask:
                #     memory_mask = torch.cat((
                #         time_embed_mask, memory_mask
                #     ), dim=-1)  # [B, seq_len, 1+seq_len]
                memory_mask = None

                sample = self.decoder(tgt=sample, memory=emb_latent,
                                      tgt_mask=tgt_mask, memory_mask=memory_mask).squeeze(0)

            else:
                raise NotImplementedError("{self.arch} is not supported.")

        # 2. we concat conditions along token dim.
        elif self.condition_concat == 'token_concat':
            if self.arch == "trans_enc":
                emb_latent = torch.cat((
                    time_embed,
                    speaker_audio_encodings,
                    speaker_3dmm_encodings,
                    past_listener_3dmmenc,
                ), dim=1)
                embed_seq_len = emb_latent.shape[0]

                if self.concat == "concat_first":
                    xseq = torch.cat((emb_latent, sample), dim=1)
                    xseq = self.query_pos(xseq)
                    tokens = self.encoder(xseq)
                    sample = tokens[embed_seq_len:]
                elif self.concat == "concat_last":
                    xseq = torch.cat((sample, emb_latent), dim=1)
                    xseq = self.query_pos(xseq)
                    tokens = self.encoder(xseq)
                    sample = tokens[:embed_seq_len]
                else:
                    raise NotImplementedError("{self.concat} is not supported.")
            elif self.arch == "trans_dec":

                # TODO: here we use same PEs for all conditions (
                #  speaker audio; speaker 3dmm; past listener frames).
                if self.position_embedding != 'none':
                    speaker_audio_encodings = self.mem_pos(speaker_audio_encodings)
                    speaker_3dmm_encodings = self.mem_pos(speaker_3dmm_encodings)
                    if past_listener_3dmmenc.shape[1] > 0:
                        past_listener_3dmmenc = self.mem_pos(past_listener_3dmmenc)

                emb_latent = torch.cat((
                    time_embed,
                    speaker_audio_encodings,
                    speaker_3dmm_encodings,
                    past_listener_3dmmenc,
                ), dim=1)

                sample = self.query_pos(sample)

                if self.use_attn_mask:
                    # do not mask past listener 3dmm
                    if past_listener_3dmmenc.shape[1] > 0:
                        past_l_mask = torch.zeros_like(memory_mask).to(memory_mask.device)
                        memory_mask = torch.cat(
                            [time_embed_mask] + [memory_mask] * 2 + [past_l_mask], dim=-1)
                    else:
                        memory_mask = torch.cat(
                            [time_embed_mask] + [memory_mask] * 2, dim=-1)

                sample = self.decoder(tgt=sample, memory=emb_latent,
                                      tgt_mask=tgt_mask, memory_mask=memory_mask).squeeze(0)

            else:
                raise NotImplementedError("{self.arch} is not supported.")

        # 3. we interact the noised input sample with conditions in a cascade manner.
        elif self.condition_concat == 'cascade':
            # add PE
            if self.position_embedding != 'none':
                speaker_audio_encodings = self.mem_pos(speaker_audio_encodings)
                speaker_3dmm_encodings = self.mem_pos(speaker_3dmm_encodings)
                if past_listener_3dmmenc.shape[1] > 0:
                    past_listener_3dmmenc = self.mem_pos(past_listener_3dmmenc)

            sample = self.query_pos(sample)

            if self.use_attn_mask:
                past_l_mask = torch.cat((
                    time_embed_mask, torch.zeros_like(memory_mask).to(memory_mask.device)
                ), dim=-1)  # [B, seq_len, 1+seq_len]

                memory_mask = torch.cat((
                    time_embed_mask, memory_mask
                ), dim=-1)  # [B, seq_len, 1+seq_len]
            else:
                past_l_mask = None

            # sample <--> speaker 3dmm
            memory = torch.cat((time_embed, speaker_3dmm_encodings), dim=1)
            sample = self.transformer_fusion_ns2sm(
                tgt=sample, memory=memory,
                tgt_mask=tgt_mask, memory_mask=memory_mask,
            ).squeeze(0)

            # noised sample <--> speaker audio
            memory = torch.cat((time_embed, speaker_audio_encodings), dim=1)
            sample = self.transformer_fusion_ns2sa(
                tgt=sample, memory=memory,
                tgt_mask=tgt_mask, memory_mask=memory_mask,
            ).squeeze(0)

            # sample <--> past listener 3dmm
            if self.use_past_frames and past_listener_3dmmenc.shape[1] > 0:
                memory = torch.cat((time_embed, past_listener_3dmmenc), dim=1)
                sample = self.transformer_fusion_ns2pl(
                    tgt=sample, memory=memory,
                    tgt_mask=tgt_mask, memory_mask=past_l_mask,
                ).squeeze(0)
                # else:
                #     memory = torch.zeros_like(time_embed).to(time_embed.device)
                #     sample = self.transformer_fusion_ns2pl(
                #         tgt=sample, memory=memory,
                #         tgt_mask=tgt_mask, memory_mask=None,
                #     ).squeeze(0)

            # final interaction
            memory = torch.cat((time_embed, sample), dim=1)
            sample = self.transformer_fusion_final(
                tgt=sample, memory=memory,
                tgt_mask=tgt_mask, memory_mask=memory_mask,
            ).squeeze(0)

        # map donoised sample back to original 3dmm_dim
        sample = self.to_3dmm_feat(sample)
        # [batch_size, seq_len, dim==58]
        # sample = sample.permute(1, 0, 2)

        return sample

    def forward_with_cond_scale(
            self,
            sample,  # noised x_t
            timesteps,
            model_kwargs,
    ):
        # expand the latents if we are doing classifier free guidance
        sample = torch.cat([sample] * 2, dim=0)
        bs, _, _ = sample.shape
        # sample = sample.permute(1, 0, 2).contiguous()

        timesteps = torch.cat([timesteps] * 2, dim=0)
        # with embedding permutation: [batch_size, l, encoded_dim]
        time_emb = self.time_proj(timesteps)  # time_embedding
        time_emb = time_emb.to(dtype=sample.dtype)
        time_embed = self.time_embedding(time_emb).unsqueeze(0)
        time_embed = time_embed.permute(1, 0, 2)

        if model_kwargs is None:
            model_kwargs = {}
        else:
            model_kwargs = model_kwargs.copy()
        # add null embeddings ...
        for k, v in model_kwargs.items():
            model_kwargs[k] = torch.cat(
                (torch.zeros_like(model_kwargs[k], dtype=model_kwargs[k].dtype), model_kwargs[k]),
                dim=0)

        (listener_latent_embed,
         listener_personal_embed,
         speaker_audio_encodings,
         speaker_latent_embed,
         speaker_3dmm_encodings,
         speaker_emotion_encodings,
         past_listener_3dmmenc) = (
            self.get_model_kwargs(
                bs,
                'test',
                sample,
                model_kwargs,
            )
        )

        prediction = self._forward(
            sample,
            time_embed,
            listener_latent_embed,
            listener_personal_embed,
            speaker_audio_encodings,
            speaker_latent_embed,
            speaker_3dmm_encodings,
            speaker_emotion_encodings,
            past_listener_3dmmenc,
        )

        pred_uncond, pred_cond = prediction.chunk(2)
        # classifier-free guidance
        prediction = pred_uncond + self.guidance_scale * (pred_cond - pred_uncond)
        return prediction

    def forward(
            self,
            sample,  # noised x_t
            timesteps,
            model_kwargs,
            **kwargs
    ):
        # sample [batch_size, latent_im[0], latent_dim[1]] => [latent_dim[0], batch_size, latent_dim[1]]
        bs, _, _ = sample.shape
        # sample = sample.permute(1, 0, 2).contiguous()

        # with embedding permutation: [batch_size, l, encoded_dim]
        time_emb = self.time_proj(timesteps)  # time_embedding
        time_emb = time_emb.to(dtype=sample.dtype)
        time_embed = self.time_embedding(time_emb).unsqueeze(0)
        time_embed = time_embed.permute(1, 0, 2)

        (listener_latent_embed,
         listener_personal_embed,
         speaker_audio_encodings,
         speaker_latent_embed,
         speaker_3dmm_encodings,
         speaker_emotion_encodings,
         past_listener_3dmmenc) = (
            self.get_model_kwargs(
                bs,
                'train',
                sample,
                model_kwargs,
            )
        )

        output = self._forward(
            sample,
            time_embed,
            listener_latent_embed,
            listener_personal_embed,
            speaker_audio_encodings,
            speaker_latent_embed,
            speaker_3dmm_encodings,
            speaker_emotion_encodings,
            past_listener_3dmmenc,
        )

        return output

    def get_model_name(self):
        return self.__class__.__name__
