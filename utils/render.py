import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision import transforms
import cv2
from utils.util import torch_img_to_np, _fix_image, torch_img_to_np2
from external.FaceVerse import get_faceverse
from external.PIRender import FaceGenerator
from skimage.io import imsave
import skvideo.io
from pathlib import Path
from tqdm import auto
import argparse


def obtain_seq_index(index, num_frames, semantic_radius=13):
    seq = list(range(index - semantic_radius, index + semantic_radius + 1))
    seq = [min(max(item, 0), num_frames - 1) for item in seq]
    return seq


def transform_semantic(semantic):
    semantic_list = []
    for i in range(semantic.shape[0]):
        index = obtain_seq_index(i, semantic.shape[0])
        semantic_item = semantic[index, :].unsqueeze(0)
        semantic_list.append(semantic_item)
    semantic = torch.cat(semantic_list, dim=0)
    return semantic.transpose(1, 2)


class Render(object):
    """Computes and stores the average and current value"""

    def __init__(self, device='cpu'):
        self.faceverse, _ = get_faceverse(device=device, img_size=224)
        self.faceverse.init_coeff_tensors()
        self.id_tensor = torch.from_numpy(np.load('external/FaceVerse/reference_full.npy')).float().view(1, -1)[:, :150]
        self.pi_render = FaceGenerator().to(device)
        self.pi_render.eval()
        checkpoint = torch.load('./external/PIRender/cur_model_fold.pth')
        self.pi_render.load_state_dict(checkpoint['state_dict'])

        self.mean_face = torch.FloatTensor(
            np.load('external/FaceVerse/mean_face.npy').astype(np.float32)).view(1, 1, -1).to(device)
        # mean_face.shape: [1, 1, 58]

        self.std_face = torch.FloatTensor(
            np.load('external/FaceVerse/std_face.npy').astype(np.float32)).view(1, 1, -1).to(device)

        self._reverse_transform_3dmm = transforms.Lambda(lambda e: e + self.mean_face)

    def rendering(self, path, ind, listener_vectors, speaker_video_clip, listener_reference):

        # 3D video
        T = listener_vectors.shape[0]
        listener_vectors = self._reverse_transform_3dmm(listener_vectors)[0]

        self.faceverse.batch_size = T
        self.faceverse.init_coeff_tensors()

        self.faceverse.exp_tensor = listener_vectors[:, :52].view(T, -1).to(listener_vectors.get_device())
        self.faceverse.rot_tensor = listener_vectors[:, 52:55].view(T, -1).to(listener_vectors.get_device())
        self.faceverse.trans_tensor = listener_vectors[:, 55:].view(T, -1).to(listener_vectors.get_device())
        self.faceverse.id_tensor = self.id_tensor.view(1, 150).repeat(T, 1).view(T, 150).to(
            listener_vectors.get_device())

        pred_dict = self.faceverse(self.faceverse.get_packed_tensors(), render=True, texture=False)
        rendered_img_r = pred_dict['rendered_img']
        rendered_img_r = np.clip(rendered_img_r.cpu().numpy(), 0, 255)
        rendered_img_r = rendered_img_r[:, :, :, :3].astype(np.uint8)

        # 2D video
        # listener_vectors = torch.cat((listener_exp.view(T,-1), listener_trans.view(T, -1), listener_rot.view(T, -1)))
        semantics = transform_semantic(listener_vectors.detach()).to(listener_vectors.get_device())
        C, H, W = listener_reference.shape
        output_dict_list = []
        duration = listener_vectors.shape[0] // 20
        listener_reference_frames = listener_reference.repeat(listener_vectors.shape[0], 1, 1).view(
            listener_vectors.shape[0], C, H, W)

        for i in range(20):
            if i != 19:
                listener_reference_copy = listener_reference_frames[i * duration:(i + 1) * duration]
                semantics_copy = semantics[i * duration:(i + 1) * duration]
            else:
                listener_reference_copy = listener_reference_frames[i * duration:]
                semantics_copy = semantics[i * duration:]
            output_dict = self.pi_render(listener_reference_copy, semantics_copy)
            fake_videos = output_dict['fake_image']
            fake_videos = torch_img_to_np2(fake_videos)
            output_dict_list.append(fake_videos) # is this differentiable rendering?

        listener_videos = np.concatenate(output_dict_list, axis=0)
        speaker_video_clip = torch_img_to_np2(speaker_video_clip)

        out = cv2.VideoWriter(os.path.join(path, ind + "_val.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 25, (672, 224))
        for i in range(rendered_img_r.shape[0]):
            combined_img = np.zeros((224, 672, 3), dtype=np.uint8)
            combined_img[0:224, 0:224] = speaker_video_clip[i]
            combined_img[0:224, 224:448] = rendered_img_r[i]
            combined_img[0:224, 448:] = listener_videos[i]
            out.write(combined_img)
        out.release()

    def rendering_for_fid(self, path, ind, listener_vectors, speaker_video_clip, listener_reference,
                          listener_video_clip, step=30):
        # 3D video
        T = listener_vectors.shape[0]
        listener_vectors = self._reverse_transform_3dmm(listener_vectors)[0]

        self.faceverse.batch_size = T
        self.faceverse.init_coeff_tensors()

        # self.faceverse.exp_tensor = listener_vectors[:, :52].view(T, -1).to(listener_vectors.get_device())
        # self.faceverse.rot_tensor = listener_vectors[:, 52:55].view(T, -1).to(listener_vectors.get_device())
        # self.faceverse.trans_tensor = listener_vectors[:, 55:].view(T, -1).to(listener_vectors.get_device())
        self.faceverse.exp_tensor = listener_vectors[:, :52].reshape(T, -1).to(listener_vectors.get_device())
        self.faceverse.rot_tensor = listener_vectors[:, 52:55].reshape(T, -1).to(listener_vectors.get_device())
        self.faceverse.trans_tensor = listener_vectors[:, 55:].reshape(T, -1).to(listener_vectors.get_device())
        self.faceverse.id_tensor = self.id_tensor.view(1, 150).repeat(T, 1).view(T, 150).to(
            listener_vectors.get_device())

        pred_dict = self.faceverse(self.faceverse.get_packed_tensors(), render=True, texture=False)
        rendered_img_r = pred_dict['rendered_img'] # differentiable rendering used in FaceVerse
        rendered_img_r = np.clip(rendered_img_r.detach().cpu().numpy(), 0, 255)
        # rendered_img_r = np.clip(rendered_img_r.cpu().numpy(), 0, 255)
        rendered_img_r = rendered_img_r[:, :, :, :3].astype(np.uint8)

        # TODO: debug: output the rendered image.
        # save_dir = "./results/train_main/temp_save/exp_test/iter_{}".format(str(ind))
        # os.makedirs(save_dir, exist_ok=True)
        # for i in range(rendered_img_r.shape[0]):
        #     img = rendered_img_r[i]
        #     cv2.imwrite(os.path.join(save_dir, 'img_{}.png'.format(str(i))), img)
        #     5/0

        # 2D video
        # listener_vectors = torch.cat((listener_exp.view(T,-1), listener_trans.view(T, -1), listener_rot.view(T, -1)))
        semantics = transform_semantic(listener_vectors.detach()).to(listener_vectors.get_device())
        C, H, W = listener_reference.shape
        output_dict_list = []
        duration = listener_vectors.shape[0] // 20
        listener_reference_frames = listener_reference.repeat(listener_vectors.shape[0], 1, 1).view(
            listener_vectors.shape[0], C, H, W)

        for i in range(20):
            if i != 19:
                listener_reference_copy = listener_reference_frames[i * duration:(i + 1) * duration]
                semantics_copy = semantics[i * duration:(i + 1) * duration]
            else:
                listener_reference_copy = listener_reference_frames[i * duration:]
                semantics_copy = semantics[i * duration:]
            output_dict = self.pi_render(listener_reference_copy, semantics_copy)
            fake_videos = output_dict['fake_image']
            fake_videos = torch_img_to_np2(fake_videos)
            output_dict_list.append(fake_videos)

        listener_videos = np.concatenate(output_dict_list, axis=0)
        speaker_video_clip = torch_img_to_np2(speaker_video_clip)

        listener_video_clip = torch_img_to_np2(listener_video_clip)

        if not os.path.exists(os.path.join(path, 'results_videos')):
            os.makedirs(os.path.join(path, 'results_videos'))

        out = cv2.VideoWriter(os.path.join(path, 'results_videos', ind + "_val.avi"), cv2.VideoWriter_fourcc(*"MJPG"),
                              25, (672, 224))
        # out = cv2.VideoWriter(os.path.join(path, 'results_videos', ind + "_val.avi"), cv2.VideoWriter_fourcc(*"MJPG"),
        #                       25, (448, 224))

        start_frame = 475
        end_frame = rendered_img_r.shape[0]
        for i in range(start_frame, end_frame):
            # combined_img = np.zeros((224, 448, 3), dtype=np.uint8)
            # combined_img[0:224, 0:224] = speaker_video_clip[i]
            # combined_img[0:224, 224:] = listener_videos[i]

            combined_img = np.zeros((224, 672, 3), dtype=np.uint8)
            combined_img[0:224, 0:224] = speaker_video_clip[i]
            combined_img[0:224, 224:448] = listener_video_clip[i]
            combined_img[0:224, 448:] = listener_videos[i]
            out.write(combined_img)
        out.release()

        # listener_video_clip = torch_img_to_np2(listener_video_clip)
        path_real = os.path.join(path, 'fid', 'real')
        if not os.path.exists(path_real):
            os.makedirs(path_real)
        path_fake = os.path.join(path, 'fid', 'fake')
        if not os.path.exists(path_fake):
            os.makedirs(path_fake)

        for i in range(0, rendered_img_r.shape[0], step):
            cv2.imwrite(os.path.join(path_fake, ind + '_' + str(i + 1) + '.png'), listener_videos[i])
            cv2.imwrite(os.path.join(path_real, ind + '_' + str(i + 1) + '.png'), listener_video_clip[i])
