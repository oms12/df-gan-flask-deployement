import os, sys
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.utils as vutils
import argparse
import torch
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from models.DAMSM import RNN_ENCODER
from models.GAN import NetG
from text_processing import tokenize,  prepare_sample_data


# def parse_args():
#     # Training settings
#     parser = argparse.ArgumentParser(description='DF-GAN')
#     parser.add_argument('--cfg', dest='cfg_file', type=str, default='app/cfg/bird.yml',
#                         help='optional config file')
#     parser.add_argument('--imgs_per_sent', type=int, default=16,
#                         help='the number of images per sentence')
#     parser.add_argument('--imsize', type=int, default=256,
#                         help='image szie')
#     parser.add_argument('--cuda', type=bool, default=False,
#                         help='if use GPU')
#     parser.add_argument('--train', type=bool, default=False,
#                         help='if training')
#     parser.add_argument('--multi_gpus', type=bool, default=False,
#                         help='if use multi-gpu')
#     parser.add_argument('--gpu_id', type=int, default=2,
#                         help='gpu id')
#     parser.add_argument('--local_rank', default=-1, type=int,
#         help='node rank for distributed training')
#     parser.add_argument('--random_sample', action='store_true',default=True, 
#         help='whether to sample the dataset with random sampler')
#     args = parser.parse_args()
#     return args

# args = parse_args()


netgpath = "/home/pria/Desktop/jugaad/DF-GAN Deploy/app/models/netG.pth"
textencoderpath = "/home/pria/Desktop/jugaad/DF-GAN Deploy/app/models/text_encoder200.pth"

# now we have to preprocess my input caption as per model requirements
pickle_path = "/home/pria/Desktop/jugaad/DF-GAN Deploy/app/models/captions_DAMSM.pickle"

def build_word_dict(pickle_path):
    with open(pickle_path, 'rb') as f:
        x = pickle.load(f)
        wordtoix = x[3]
        del x
        n_words = len(wordtoix)
        print('Load from: ', pickle_path)
    return n_words, wordtoix

vocab_size, wordtoix = build_word_dict(pickle_path)


# # generator is done
nf = 32
z_dim = 100
cond_dim = 256
imsize = 256
ch_size = 3
netG = NetG(nf, z_dim, cond_dim, imsize, ch_size)

def load_model_weights(model, weights, multi_gpus, train=True):
    if list(weights.keys())[0].find('module')==-1:
        pretrained_with_multi_gpu = False
    else:
        pretrained_with_multi_gpu = True
    if (multi_gpus==False) or (train==False):
        if pretrained_with_multi_gpu:
            state_dict = {
                key[7:]: value
                for key, value in weights.items()
            }
        else:
            state_dict = weights
    else:
        state_dict = weights
    model.load_state_dict(state_dict)
    return model
checkpoint = torch.load(netgpath)
netG = load_model_weights(netG, checkpoint, False, False)
netG.eval()


# text encoder is done
text_encoder = RNN_ENCODER(vocab_size, nhidden=256)
state_dict = torch.load(textencoderpath)
text_encoder.load_state_dict(state_dict)
for p in text_encoder.parameters():
    p.requires_grad = False
text_encoder.eval()




# # truncuation of the noise

def truncated_noise(batch_size=1, dim_z=100, truncation=1., seed=None):
    from scipy.stats import truncnorm
    state = None if seed is None else np.random.RandomState(seed)
    values = truncnorm.rvs(-2, 2, size=(batch_size, dim_z), random_state=state).astype(np.float32)
    return truncation * values

# main sample function

def sample(sentences):
    batch_size, device = 1, 'cpu'
    truncation, trunc_rate = True, 0.88
    z_dim = 100
    vocab_size, wordtoix = build_word_dict(pickle_path)
    captions, cap_lens, _ = tokenize(wordtoix, sentences)
    sent_embs, _  = prepare_sample_data(captions, cap_lens, text_encoder, device)
    caption_num = sent_embs.size(0)
    # get noise
    if truncation==True:
        noise = truncated_noise(batch_size, z_dim, trunc_rate)
        noise = torch.tensor(noise, dtype=torch.float).to(device)
    else:
        noise = torch.randn(batch_size, z_dim).to(device)
    # sampling
    with torch.no_grad():
        fakes = []
        for i in range(caption_num):
            sent_emb = sent_embs[i].unsqueeze(0).repeat(batch_size, 1)
            fakes = netG(noise, sent_emb)
        img = fakes[0]
        im = img.data.cpu().numpy()
        # [-1, 1] --> [0, 255]
        im = (im + 1.0) * 127.5
        im = im.astype(np.uint8)
        im = np.transpose(im, (1, 2, 0))
        im = Image.fromarray(im)
        im.save("static/newimage.jpg")
        # now it has generated the image and hence we have to conver the tensor back into the images
    # return "gautam"





