# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import cv2

import numpy as np
from skimage.metrics import structural_similarity


def psnr_metric(img_pred, img_gt):
    mse = np.mean((img_pred - img_gt) ** 2)
    psnr = -10 * np.log(mse) / np.log(10)
    return psnr

def eval_score(gt_img_path, pred_img_path):
    img_gt = cv2.imread(gt_img_path).astype(np.float32)/255.
    img_pred = cv2.imread(pred_img_path).astype(np.float32)/255.

    psnr = psnr_metric(img_pred, img_gt)
    ssim = structural_similarity(img_pred, img_gt, multichannel=True)
    return dict(psnr=psnr, ssim=ssim)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src_img', type=str)
    parser.add_argument('-t', '--trg_img', type=str)
    args = parser.parse_args()

    print(eval_score(args.trg_img, args.src_img))