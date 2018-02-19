import torch
from torch import nn
from torch import autograd

import align
from targets import LightCNN


def find_target_vector(model, target_data):
    output = model(target_data)
    return output.mean(0)


def mse_loss(input, target):
    """
    TODO: change loss fxn
    """
    return torch.sum((input - target)**2) / input.data.nelement()


def attack(image, model, target_vector, eps=0.001, n_epoch=25):
    X = autograd.Variable(image, requires_grad=True)
    X_orig = X.clone()
    output_orig = None
    for i in range(n_epoch):
        print("Iteration:", i)
        output = model(X)
        if output_orig is None:
            output_orig = output.clone()
        loss = mse_loss(output, target_vector)
        loss.backward(retain_graph=True)

        X_grad = torch.sign(X.grad.data)
        X_adv = X.data - eps * X_grad

        output_adv = model(autograd.Variable(X_adv))
        cossim = nn.CosineSimilarity()
        print("Orig -> Adv:", float(cossim(output_orig, output_adv)))
        print("Orig -> Target:",
              float(cossim(output_orig,
                           torch.unsqueeze(target_vector, 0))))
        print("Adv -> Target:",
              float(cossim(output_adv,
                           torch.unsqueeze(target_vector, 0))))
        X = autograd.Variable(X_adv, requires_grad=True)

    return X_adv - X_orig.data


if __name__ == "__main__":
    print("loading model")
    model = LightCNN()

    print("Loading target")
    target_images, _ = align.images_to_tensor(align.align_image_dir("./images/zuck/original"))
    target_data = autograd.Variable(target_images)
    target_vector = find_target_vector(model, target_data)

    print("Loading adv image")
    data, meta = align.images_to_tensor(align.align_image_dir("./images/micha/original"))

    noise_profile = attack(data, model, target_vector)
    align.add_noise(noise_profile, meta, save=True)
