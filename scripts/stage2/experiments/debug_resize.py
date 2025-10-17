


import cv2
import numpy as np
import os
from PIL import Image


def upsample_gm_redsdr_size(gm_image, original_sdr_image):
    return cv2.resize(gm_image, (original_sdr_image.shape[1], original_sdr_image.shape[0]), interpolation=cv2.INTER_CUBIC)

def save_hdr_image(apply_HDR, save_dir, filename, qmax):
    apply_HDR = apply_HDR / (qmax+1)
    apply_HDR_image = apply_HDR.astype(np.float32)
    cv2.imwrite(os.path.join(save_dir, f"{filename}"), apply_HDR_image)

def apply_gm_to_sdr(gm, sdr, qmax=9, eps=1/64):
    """
    Applies the given GM (Guidance Map) to SDR (Standard Dynamic Range) image to compute HDR (High Dynamic Range) output.
    Note: SDR and GM should be NumPy arrays with values in [0, 1] for SDR.
    """
    # Ensure SDR is within [0, 1] and linearize using sRGB gamma
    sdr_clamped = np.clip(sdr, 0, 1)
    sdr_linear = sdr_clamped ** 2.2

    # Compute HDR output
    output_hdr = (sdr_linear + eps) * (1 + gm * qmax) - eps
    return output_hdr


if __name__ == "__main__":
    

    for i in range(1, 120):
        fname = f"{i:03d}.png"
        input_dir = "/path/to/stage2/results"  
        sdr_path = rf"/path/to/hdrtv1k/test_sdr/{fname}"
        gm_path = rf"{input_dir}/gm_{fname}"
        print(gm_path)
        sdr_vae_path = rf"{input_dir}/sdr_{fname}"
        output_dir = rf"{input_dir}/second_apply"
        os.makedirs(output_dir, exist_ok=True)

        sdr = cv2.imread(sdr_path, cv2.IMREAD_UNCHANGED) / 255.0
        gm = cv2.imread(gm_path, cv2.IMREAD_UNCHANGED) / 255.0
        sdr_vae = cv2.imread(sdr_vae_path, cv2.IMREAD_UNCHANGED) / 255.0

        
        apply_HDR_vae = apply_gm_to_sdr(gm, sdr_vae, qmax=99)
        save_hdr_image(apply_HDR_vae, output_dir, f"vae_hdr_{i:03d}.hdr", 99)

        upsample_gm = upsample_gm_redsdr_size(gm_image=gm, original_sdr_image=sdr)
        apply_HDR = apply_gm_to_sdr(upsample_gm, sdr, qmax=99)
        save_hdr_image(apply_HDR, output_dir, f"hdr_{i:03d}.hdr", 99)


    print("Done")



