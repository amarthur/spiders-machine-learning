from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from captum.attr import GradientShap, IntegratedGradients
from captum.attr import visualization as viz
from matplotlib.figure import Figure
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


def shap_grads(ckpt: Path, img_path: Path, n_samples: int) -> Figure:
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
    return get_vis(attributions_gs, transformed_img)


def integrated_grads(ckpt: Path, img_path: Path, n_steps: int) -> Figure:
    model = get_model(ckpt)
    transformed_img, input_tensor = get_transformed_imgs(img_path)
    pred_label_idx = get_pred(model, input_tensor)

    integrated_gradients = IntegratedGradients(model)
    attributions_ig = integrated_gradients.attribute(
        input_tensor,
        target=pred_label_idx,
        n_steps=n_steps,
    )

    return get_vis(attributions_ig, transformed_img)


def get_vis(attributions: torch.Tensor, transformed_img: torch.Tensor) -> Figure:
    vis = viz.visualize_image_attr_multiple(
        to_ndarray(attributions),
        to_ndarray(transformed_img),
        ["heat_map"],
        ["positive"],
        show_colorbar=True,
    )
    return vis[0]


def to_ndarray(tensor: torch.Tensor) -> np.ndarray:
    return np.transpose(tensor.squeeze().cpu().detach().numpy(), (1, 2, 0))


def save_vis_fig(save_path: Path, fig: Figure, transparent: bool = False):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=300, transparent=transparent)


def get_pred(model: SpModel, input_tensor: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        output = model(input_tensor)

    output = F.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1)

    pred_label_idx.squeeze_()
    predicted_label = PREDICTION_LABELS[pred_label_idx.item()]
    print("Predicted:", predicted_label, "(", prediction_score.squeeze().item(), ")")
    return pred_label_idx


def get_model(ckpt: Path) -> SpModel:
    model = SpModel.load_from_checkpoint(ckpt).to(DEVICE)
    model = model.eval()
    return model


def get_transformed_imgs(img_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    transforms = SpDataTransforms()
    img = Image.open(img_path)

    transformed_img = transforms.tensor_transform(img)
    normalized_transformed_img = transforms.normalize_transform(transformed_img)
    input_tensor = normalized_transformed_img.unsqueeze(0).to(DEVICE)
    return transformed_img, input_tensor


def save_base_img(img_path: Path, grad_type: str, transparent: bool = True) -> None:
    img, _ = get_transformed_imgs(img_path)
    transformed_image = to_ndarray(img)
    save_path = Path(f"Imgs/{grad_type}/{img_path.stem}.png")

    fig, ax = plt.subplots()
    ax.imshow(transformed_image)
    ax.axis("off")

    save_vis_fig(save_path, fig, transparent)


def main():
    ckpt = {
        "Swin": Path("Swin.ckpt"),
        "ResNet": Path("Resnet.ckpt"),
        "ConvNeXt": Path("Convnext.ckpt"),
        "ResNeXt": Path("ResNeXt.ckpt"),
        "ViT": Path("ViT.ckpt"),
        "MaxViT": Path("MaxViT.ckpt"),
    }
    imgs = [
        "Lycosa erythrognatha/Lycosa_erythrognatha_300.jpg",
        "Micrathena fissispina/Micrathena_fissispina_8.jpg",
        "Argiope argentata/Argiope_argentata_8.JPG",
        "Trichonephila clavipes/Trichonephila_clavipes_109.jpg",
        "Argiope argentata/Argiope_argentata_4066.jpeg",
        "Gasteracantha cancriformis/Gasteracantha_cancriformis_10.jpeg",
    ]

    n = 0
    n_samples = 300
    grad_type = "IG"

    img_path = Path(imgs[n])
    grad_dict = {"IG": integrated_grads, "SHAP": shap_grads}

    save_base_img(img_path, grad_type)

    for model_name, ckpt in ckpt.items():
        save_img_path = Path(f"Imgs/{grad_type}/{img_path.stem}_{model_name.lower()}")
        vis = grad_dict[grad_type](ckpt, img_path, n_samples)
        save_vis_fig(save_img_path, vis, True)


if __name__ == "__main__":
    main()
