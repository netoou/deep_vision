# this augment code is from Kakaobrain
# https://github.com/kakaobrain/fast-autoaugment/blob/master/FastAutoAugment/augmentations.py

# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
import random

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
from torchvision.transforms.transforms import Compose, ToTensor, Normalize

random_mirror = True


def ShearX(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random_mirror and random.random() > 0.5:
        v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random_mirror and random.random() > 0.5:
        v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateXAbs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v <= 10
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateYAbs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v <= 10
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Rotate(img, v):  # [-30, 30]
    assert -30 <= v <= 30
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.rotate(v)


def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Flip(img, _):  # not from the paper
    return PIL.ImageOps.mirror(img)


def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)


def Posterize(img, v):  # [4, 8]
    assert 4 <= v <= 8
    v = int(v)
    return PIL.ImageOps.posterize(img, v)


def Posterize2(img, v):  # [0, 4]
    assert 0 <= v <= 4
    v = int(v)
    return PIL.ImageOps.posterize(img, v)


def Contrast(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Color(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Color(img).enhance(v)


def Brightness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Sharpness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2]
    assert 0.0 <= v <= 0.2
    if v <= 0.:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, v)


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def SamplePairing(imgs):  # [0, 0.4]
    def f(img1, v):
        i = np.random.choice(len(imgs))
        img2 = PIL.Image.fromarray(imgs[i])
        return PIL.Image.blend(img1, img2, v)

    return f


def Identity(imgs):
    return imgs


def augment_list(for_autoaug=True):  # 16 oeprations and their ranges
    l = [
        (ShearX, -0.3, 0.3),  # 0
        (ShearY, -0.3, 0.3),  # 1
        (TranslateX, -0.45, 0.45),  # 2
        (TranslateY, -0.45, 0.45),  # 3
        (Rotate, -30, 30),  # 4
        (AutoContrast, 0, 1),  # 5
        (Invert, 0, 1),  # 6
        (Equalize, 0, 1),  # 7
        (Solarize, 0, 256),  # 8
        (Posterize, 4, 8),  # 9
        (Contrast, 0.1, 1.9),  # 10
        (Color, 0.1, 1.9),  # 11
        (Brightness, 0.1, 1.9),  # 12
        (Sharpness, 0.1, 1.9),  # 13
        (Cutout, 0, 0.2),  # 14
        # (SamplePairing(imgs), 0, 0.4),  # 15
    ]
    if for_autoaug:
        l += [
            (CutoutAbs, 0, 20),  # compatible with auto-augment
            (Posterize2, 0, 4),  # 9
            (TranslateXAbs, 0, 10),  # 9
            (TranslateYAbs, 0, 10),  # 9
        ]
    return l


augment_dict = {fn.__name__: (fn, v1, v2) for fn, v1, v2 in augment_list()}
AUGMENT_NAMES = list(augment_dict.keys())
RAND_AUGMENT_NAMES = list(augment_dict.keys()) + ['Identity']


def get_augment(name):
    return augment_dict[name]


def apply_augment(img, name, level):
    augment_fn, low, high = get_augment(name)
    return augment_fn(img.copy(), level * (high - low) + low)


class AugmentPolicyTransform:
    # set this class to transform position at dataset object
    def __init__(self, policy, totensor=True):
        self.totensor = ToTensor() if totensor else None
        self.sub_policies = [self._set_transform_fn(sub_policy) for sub_policy in policy]

    def __call__(self, img):
        if self.totensor:
            return self.totensor(np.random.choice(self.sub_policies)(img))
        else:
            return np.random.choice(self.sub_policies)(img)

    def _set_transform_fn(self, sub_policy):
        sub_policy = sub_policy

        def apply_transform(img):
            if not type(img) == PIL.Image.Image:
                img = PIL.Image.fromarray(img)

            for name, prob, mag in sub_policy:
                if np.random.rand() < prob:
                    img = apply_augment(img, name, mag)
            return img

        return apply_transform


class RandAugmentPolicyTransform:
    def __init__(self, N: int, M: int, totensor=True):
        self.totensor = ToTensor() if totensor else None
        self.N = N
        self.M = M
        self.magnitude_controller = 1

    def __call__(self, img):
        policy = self.randaugment(self.N, self.M * self.magnitude_controller)

        if self.totensor:
            return self.totensor(self._apply_transform(img, policy))
        else:
            return self._apply_transform(img, policy)

    def _apply_transform(self, img, policy):
        if not type(img) == PIL.Image.Image:
            img = PIL.Image.fromarray(img)

        for name, mag in policy:
            img = apply_augment(img, name, mag)

        return img

    def randaugment(self, n, m):
        sampled_ops = np.random.choice(RAND_AUGMENT_NAMES, n)
        return [(op, m) for op in sampled_ops]

    def step_magnitude_controller(self, value):
        self.magnitude_controller = value


if __name__ == '__main__':
    pc = AugmentPolicyTransform([
        [('Solarize', 0.66, 0.34), ('Equalize', 0.56, 0.61)],
        [('Equalize', 0.43, 0.06), ('AutoContrast', 0.66, 0.08)],
        [('Color', 0.72, 0.47), ('Contrast', 0.88, 0.86)],
        [('Brightness', 0.84, 0.71), ('Color', 0.31, 0.74)],
        [('Rotate', 0.68, 0.26), ('TranslateX', 0.38, 0.88)]])

    cp = Compose([
        pc,
        Normalize((0,0,0), (0,0,0)),
    ])

    inp = (np.random.rand(4,4,3)*255).astype(np.uint8)
    for i in range(10):
        print(np.asarray(pc(inp)))
        print(cp(inp))

