import face_recognition
from PIL import Image
from skimage.transform import resize
import os

import torch
from torchvision import transforms


def add_noise(noise_profile, meta, save=False):
    images = []
    for noise, m in zip(noise_profile, meta):
        top, right, bottom, left = m['face_location']
        image = face_recognition.load_image_file(m['filename'])
        noise_resize = 255 * resize(
            noise[0].numpy(),
            (bottom-top, right-left),
        )
        image = image.astype('float')
        for i in range(image.shape[-1]):
            image[top:bottom, left:right, i] -= noise_resize
        image = image.astype('uint8')
        if save:
            Image.fromarray(image).save(m['filename'] + '.adv.png')
        images.append(image.astype('uint8'))
    return images


def align_image(image_path):
    try:
        image = face_recognition.load_image_file(image_path)
    except:
        return
    face_locations = face_recognition.face_locations(image)
    for face_location in face_locations:
        top, right, bottom, left = face_location
        face_image = image[top:bottom, left:right]
        meta = {
            "filename": image_path,
            "face_location": face_location,
        }
        yield Image.fromarray(face_image), meta


def align_image_dir(images_path):
    for filename in os.listdir(images_path):
        absfilename = os.path.join(images_path, filename)
        yield from align_image(absfilename)


def images_to_tensor(images):
    t = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    images, meta = zip(*images)
    data = list(map(t, images))
    return torch.stack(data), meta
