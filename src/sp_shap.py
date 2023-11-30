from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from captum.attr import GradientShap
from captum.attr import visualization as viz
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

from sp_model import SpModel
from sp_transform import SpDataTransforms

DEVICE = "cpu"

PREDICTION_LABELS = {
    0: "Agl lag",
    1: "Alp gra",
    2: "Alp ven",
    3: "Arg arg",
    4: "Chi spi",
    5: "Cor con",
    6: "Cte orn",
    7: "Eri eda",
    8: "Gas can",
    9: "Has ada",
    10: "Het ven",
    11: "Lat geo",
    12: "Lyc ery",
    13: "Mar nig",
    14: "Meg sut",
    15: "Men biv",
    16: "Mic fis",
    17: "Nep cru",
    18: "Nes ruf",
    19: "Oxy sal",
    20: "Peu rub",
    21: "Ple pay",
    22: "Scy fus",
    23: "Tri cla",
    24: "Zos gen",
}


def get_shap(ckpt: str, img_path: str, n_samples: int):
    model = get_model(ckpt)
    transformed_img, input_tensor = get_transformed_imgs(img_path)
    pred_label_idx = get_pred(model, input_tensor)

    gradient_shap = GradientShap(model)
    rand_img_dist = torch.cat([input_tensor * 0, input_tensor * 1])

    attributions_gs = gradient_shap.attribute(
        input_tensor,
        n_samples=n_samples,
        stdevs=0.0001,
        baselines=rand_img_dist,
        target=pred_label_idx,
    )
    vis = viz.visualize_image_attr_multiple(
        np.transpose(attributions_gs.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        ["original_image", "heat_map"],
        ["all", "absolute_value"],
        cmap=get_cmap(),
        show_colorbar=True,
    )
    return vis[0]


def save_shap_fig(save_path, fig):
    fig.savefig(save_path, bbox_inches="tight", dpi=300)


def get_cmap():
    cmap = LinearSegmentedColormap.from_list(
        "custom blue", [(0, "#ffffff"), (0.25, "#000000"), (1, "#000000")], N=256
    )
    return cmap


def get_pred(model: SpModel, input_tensor: Any):
    with torch.no_grad():
        output = model(input_tensor)

    output = F.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1)

    pred_label_idx.squeeze_()
    predicted_label = PREDICTION_LABELS[pred_label_idx.item()]
    print("Predicted:", predicted_label, "(", prediction_score.squeeze().item(), ")")
    return pred_label_idx


def get_model(ckpt: str) -> SpModel:
    model = SpModel.load_from_checkpoint(ckpt).to(DEVICE)
    model = model.eval()
    return model


def get_transformed_imgs(img_path: str):
    transforms = SpDataTransforms()
    img = Image.open(img_path)

    transformed_img = transforms.tensor_transform(img)
    normalized_transformed_img = transforms.normalize_transform(transformed_img)
    input_tensor = normalized_transformed_img.unsqueeze(0).to(DEVICE)
    return transformed_img, input_tensor


def main():
    ckpt = {
        "resnet": "ckpts/resnet.ckpt",
        "resnext": "ckpts/resnext.ckpt",
        "convnext": "ckpts/convnext.ckpt",
        "vit": "ckpts/vit.ckpt",
        "swin": "ckpts/swin.ckpt",
        "maxvit": "ckpts/maxvit.ckpt",
    }
    n_samples = 200
    model_key = "convnext"
    img_path = "Lycosa erythrognatha/Lycosa_erythrognatha_300.jpg"
    save_img_path = f"Imgs/SHAP/{model_key}"

    shap = get_shap(ckpt["convnext"], img_path, n_samples)
    save_shap_fig(save_img_path, shap)


if __name__ == "__main__":
    main()
