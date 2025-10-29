# EDIT OF FILE NVIDIA/BigVGAN/inference_e2e.py FOR DIRECT INFERENCE BASED ON GIVEN MEL-SPEC IN .NPY-FORMAT

from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import numpy as np
import argparse
import json
import torch
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import MAX_WAV_VALUE
from bigvgan import BigVGAN as Generator

h = None
device = None
torch.backends.cudnn.benchmark = False


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print(f"Loading '{filepath}'")
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + "*")
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ""
    return sorted(cp_list)[-1]


def inference(a, h):
    generator = Generator(h, use_cuda_kernel=a.use_cuda_kernel).to(device)

    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g["generator"])

    # Create output dir
    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()
    with torch.no_grad():
        
        # Load the mel spectrogram in .npy format
        x = np.load(a.mel_spec)
        x = torch.FloatTensor(x).to(device)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)

        y_g_hat = generator(x)

        audio = y_g_hat.squeeze()
        audio = audio * MAX_WAV_VALUE
        audio = audio.cpu().numpy().astype("int16")

        output_file = os.path.join(
            a.output_dir, a.output_filename
        )
        write(output_file, h.sampling_rate, audio)
        print(output_file)


def main():
    print("Initializing Inference Process..")

    parser = argparse.ArgumentParser()
    parser.add_argument("--mel_spec", default=None, required=True)
    parser.add_argument("--output_dir", default=os.path.join(
                                                            os.path.dirname(__file__),
                                                            "..",
                                                            "..", 
                                                            "data",
                                                            "generated"))
    parser.add_argument("--output_filename", required=True)
    parser.add_argument("--checkpoint_file", required=True)
    parser.add_argument("--use_cuda_kernel", action="store_true", default=False)

    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], "config.json")
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    inference(a, h)


if __name__ == "__main__":
    main()
