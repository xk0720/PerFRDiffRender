import numpy as np
import torch
import torch.nn as nn
from project_react.model.TransVAE.BasicBlock import ConvBlock, PositionalEncoding, lengths_to_mask, init_biased_mask


class PositionEmbeddingSine(nn.Module):

    def __init__(self, d_model, max_len=5000, batch_first=True):
        super(PositionEmbeddingSine, self).__init__()
        self.batch_first = batch_first

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        if self.batch_first:
            pos = self.pe.permute(1, 0, 2)[:, :x.shape[1], :]
        else:
            pos = self.pe[:x.shape[0], :]
        return pos


class VideoEncoder(nn.Module):
    def __init__(self, img_size=224, feature_dim=128, device='cuda:0'):
        super(VideoEncoder, self).__init__()

        self.img_size = img_size
        self.feature_dim = feature_dim
        self.Conv3D = ConvBlock(3, feature_dim)
        self.fc = nn.Linear(feature_dim, feature_dim)
        self.device = device

    def forward(self, video):
        """
        input:
        speaker_video_frames x: (batch_size, seq_len, 3, img_size, img_size)

        output:
        speaker_temporal_tokens y: (batch_size, seq_len, token_dim)
        """

        video_input = video.transpose(1, 2)  # B C T H W
        token_output = self.Conv3D(video_input).transpose(1, 2)
        token_output = self.fc(token_output)  # B T C
        return token_output


class SpeakerBehaviourEncoder(nn.Module):
    def __init__(self, img_size=224, audio_dim=78, feature_dim=128, device='cuda:0'):
        super(SpeakerBehaviourEncoder, self).__init__()

        self.img_size = img_size
        self.audio_dim = audio_dim
        self.feature_dim = feature_dim
        self.device = device

        self.video_encoder = VideoEncoder(img_size=img_size, feature_dim=feature_dim, device=device)
        self.audio_feature_map = nn.Linear(self.audio_dim, self.feature_dim)
        self.fusion_layer = nn.Linear(self.feature_dim * 2, self.feature_dim)

    def forward(self, video, audio):
        video_feature = self.video_encoder(video)  # shape: (bs, window_size, feature_dim==128)
        audio_feature = self.audio_feature_map(audio)  # shape: (bs, window_size, feature_dim==128)

        speaker_behaviour_feature = self.fusion_layer(torch.cat((video_feature, audio_feature), dim=-1))

        return speaker_behaviour_feature


class VAEModel(nn.Module):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int = 256,
                 position_embedding: str = "sine",
                 **kwargs) -> None:
        super(VAEModel, self).__init__()

        self.latent_dim = latent_dim
        self.in_channels = in_channels
        if position_embedding == "sine":
            self.pos_embedding = PositionEmbeddingSine(d_model=latent_dim)
        elif position_embedding == "none":
            self.pos_embedding = nn.Identity()
        else:
            raise NotImplementedError("Position embedding method {} not implemented.".format(self.pos_embedding))

        self.linear = nn.Linear(in_channels, latent_dim)

        seq_trans_encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim,
                                                             nhead=4,
                                                             dim_feedforward=latent_dim * 2,
                                                             dropout=0.1)

        self.seqTransEncoder = nn.TransformerEncoder(seq_trans_encoder_layer, num_layers=1)
        self.mu_token = nn.Parameter(torch.randn(latent_dim))
        self.logvar_token = nn.Parameter(torch.randn(latent_dim))

    def forward(self, input):
        x = self.linear(input)  # B T D
        B, T, D = input.shape
        lengths = [len(item) for item in input]

        mu_token = torch.tile(self.mu_token, (B,)).reshape(B, 1, -1)
        logvar_token = torch.tile(self.logvar_token, (B,)).reshape(B, 1, -1)

        x = torch.cat([mu_token, logvar_token, x], dim=1)
        x = x + self.pos_embedding(x)
        x = x.permute(1, 0, 2)

        token_mask = torch.ones((B, 2), dtype=bool, device=input.get_device())
        mask = lengths_to_mask(lengths, input.get_device())
        aug_mask = torch.cat((token_mask, mask), 1)

        x = self.seqTransEncoder(x, src_key_padding_mask=~aug_mask)

        mu = x[0]
        logvar = x[1]
        std = logvar.exp().pow(0.5)
        dist = torch.distributions.Normal(mu, std)
        motion_sample = self.sample_from_distribution(dist).to(input.get_device())

        return motion_sample, dist

    def sample_from_distribution(self, distribution):
        return distribution.rsample()


class Decoder(nn.Module):
    def __init__(self, output_3dmm_dim=58, output_emotion_dim=25, feature_dim=128, device='cpu', max_seq_len=751,
                 n_head=4, window_size=8):
        super(Decoder, self).__init__()

        self.feature_dim = feature_dim
        self.window_size = window_size
        self.device = device

        self.vae_model = VAEModel(feature_dim, feature_dim)

        decoder_layer = nn.TransformerDecoderLayer(d_model=feature_dim, nhead=n_head, dim_feedforward=2 * feature_dim,
                                                   batch_first=True)
        self.reaction_decoder_1 = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.reaction_decoder_2 = nn.TransformerDecoder(decoder_layer, num_layers=1)

        self.biased_mask = init_biased_mask(n_head=n_head, max_seq_len=max_seq_len, period=max_seq_len)

        self.reaction_3dmm_map_layer = nn.Linear(feature_dim, output_3dmm_dim)
        self.reaction_emotion_map_layer = nn.Sequential(
            nn.Linear(feature_dim + output_3dmm_dim, feature_dim),
            nn.Linear(feature_dim, output_emotion_dim)
        )
        self.PE = PositionalEncoding(feature_dim)

    def forward(self, encoded_feature):
        B, TL = encoded_feature.shape[0], encoded_feature.shape[1]

        motion_sample, dist = self.vae_model(encoded_feature)
        time_queries = torch.zeros(B, TL, self.feature_dim, device=encoded_feature.get_device())
        time_queries = self.PE(time_queries)
        tgt_mask = self.biased_mask[:, :TL, :TL].clone().detach().to(device=self.device).repeat(B, 1, 1)

        reaction = self.reaction_decoder_1(tgt=time_queries, memory=motion_sample.unsqueeze(1), tgt_mask=tgt_mask)
        reaction = self.reaction_decoder_2(reaction, reaction, tgt_mask=tgt_mask)

        out_3dmm = self.reaction_3dmm_map_layer(reaction)
        out_emotion = self.reaction_emotion_map_layer(torch.cat((out_3dmm, reaction), dim=-1))

        return out_3dmm, out_emotion, motion_sample, dist

    def reset_window_size(self, window_size):
        self.window_size = window_size


class TransformerVAE(nn.Module):
    def __init__(self, img_size=224, audio_dim=78, output_3dmm_dim=58, output_emotion_dim=25, feature_dim=128,
                 seq_len=751, window_size=50, device='cpu'):
        super(TransformerVAE, self).__init__()

        self.img_size = img_size
        self.feature_dim = feature_dim
        self.output_3dmm_dim = output_3dmm_dim
        self.output_emotion_dim = output_emotion_dim
        self.seq_len = seq_len
        self.window_size = window_size

        self.speaker_behaviour_encoder = SpeakerBehaviourEncoder(img_size, audio_dim, feature_dim, device)
        self.reaction_decoder = Decoder(output_3dmm_dim=output_3dmm_dim, output_emotion_dim=output_emotion_dim,
                                        feature_dim=feature_dim, device=device, window_size=self.window_size)
        self.fusion = nn.Linear(feature_dim + self.output_3dmm_dim + self.output_emotion_dim, feature_dim)

    def _encode(self, video, audio):
        """
        input:
        video: (batch_size * n, windom_size, 3, img_size, img_size)
        audio: (batch_size * n, windom_size, raw_wav)
        """
        encoded_feature = self.speaker_behaviour_encoder(video, audio)
        out_3dmm, out_emotion, motion_sample, dist = self.reaction_decoder(encoded_feature)
        return out_3dmm, out_emotion, motion_sample, dist

    def encode(self, video, audio):
        """
        input:
        video: (batch_size * n, windom_size, 3, img_size, img_size)
        audio: (batch_size * n, windom_size, raw_wav)
        """
        batch_size, seq_len = video.shape[:2]
        if seq_len > self.window_size:
            selected_window = np.random.randint(0, seq_len // self.window_size)
            selected_video_seq = video[:, selected_window * self.window_size: (selected_window + 1) * self.window_size, :]
            selected_audio_seq = audio[:, selected_window * self.window_size: (selected_window + 1) * self.window_size, :]
        elif seq_len == self.window_size:
            selected_video_seq = video
            selected_audio_seq = audio
        else:
            raise ValueError("seq_len must be at least window_size")

        return self._encode(selected_video_seq, selected_audio_seq)

    def encode_all(self, video, audio):
        batch_size, frame_num = video.shape[:2]
        video = video.reshape(batch_size * (frame_num // self.window_size), self.window_size, -1)
        audio = audio.reshape(batch_size * (frame_num // self.window_size), self.window_size, -1)
        _, _, motion_sample, _ = self._encode(video, audio)

        return motion_sample # shape: (batch_size * n, dim)

    def forward(self, video, audio, reaction_3dmm, reaction_emotion):
        """
        input:
        video: (batch_size, seq_len, 3, img_size, img_size)
        audio: (batch_size, seq_len, raw_wav)

        output:
        3dmm_vector: (batch_size, seq_len, output_3dmm_dim)
        emotion_vector: (batch_size, seq_len, output_emotion_dim)
        distribution: [dist_1,...,dist_n]
        """
        batch_size, frame_num, c, h, w = video.shape

        distribution = []
        # stack window_size frames together
        reaction_3dmm = reaction_3dmm.reshape(batch_size * (frame_num // self.window_size), self.window_size, -1)
        reaction_emotion = reaction_emotion.reshape(batch_size * (frame_num // self.window_size), self.window_size, -1)
        # shape (batch_size * n, window_size, ...)

        video = video.reshape(batch_size * (frame_num // self.window_size), self.window_size, c, h, w)
        audio = audio.reshape(batch_size * (frame_num // self.window_size), self.window_size, -1)

        # feed to encode
        out_3dmm, out_emotion, _, dist = self._encode(video, audio) # or self.encode(video, audio)
        distribution.append(dist)

        return {
            "gt_emotion": reaction_emotion,
            "gt_3dmm": reaction_3dmm,
            "pred_emotion": out_emotion,
            "pred_3dmm": out_3dmm,
            "distribution": distribution,
        }

    def reset_window_size(self, window_size):
        self.window_size = window_size
        self.reaction_decoder.reset_window_size(window_size)


if __name__ == "__main__":
    pass
