from PIL.Image import Image
import torch
import matplotlib.pyplot as plt
from math import ceil
import numpy as np
from PIL import Image as PILImage
import matplotlib.cm as cm
from matplotlib import colors


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _render_overlay(
    image: PILImage.Image,
    activation: np.ndarray,
    cmap="viridis",
    alpha=0.5,
    vmin=0.0,
    vmax=1.0,
):

    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap_fn = cm.get_cmap(cmap)

    act_norm = norm(activation)
    act_rgba = cmap_fn(act_norm)  # (H, W, 4)
    act_img = (act_rgba * 255).astype(np.uint8)

    act_pil = PILImage.fromarray(act_img)
    act_pil = act_pil.resize(image.size, resample=PILImage.BILINEAR)

    base = image.convert("RGBA")
    overlay = PILImage.new("RGBA", image.size)
    overlay.paste(act_pil, (0, 0), act_pil)

    blended = PILImage.alpha_composite(base, PILImage.blend(base, overlay, alpha))
    return blended


def overlay_heatmap(
    image: Image,
    feature_activations: torch.Tensor,
    timesteps: list[int] | None = None,
    cols: int = 4,
    cmap: str = "viridis",
    alpha: float = 0.5,
    figsize: tuple[int, int] | None = None,
):
    """
    Create a multi-subplot figure overlaying per-timestep activations on `image`.

    Args:
        image: PIL Image
        feature_activations: torch.Tensor with shape (T, H, W)
        timesteps: list of timestep indices to visualize
        cols: number of columns in the subplot grid
        cmap, alpha: visualization params

    Returns:
        matplotlib.Figure
    """

    arr = _to_numpy(feature_activations)
    if arr.ndim != 3:
        raise ValueError("feature_activations must have shape (T, H, W)")

    T = arr.shape[0]
    timesteps = [t for t in timesteps if 0 <= t < T]

    n = len(timesteps)
    rows = ceil(n / cols)
    if figsize is None:
        figsize = (cols * 3, rows * 3)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if n > 1 else [axes]

    for ax_idx, t in enumerate(timesteps):
        activation = arr[t]
        # normalize per-timestep to [0,1] to avoid needing a shared scale
        a_min = float(np.nanmin(activation))
        a_max = float(np.nanmax(activation))
        if a_max - a_min < 1e-8:
            print(
                f"Warning: timestep {t} has near-constant activation (min={a_min:.4f}, max={a_max:.4f})"
            )

        norm_act = (activation - a_min) / (a_max - a_min)

        overlay_img = _render_overlay(
            image, norm_act, cmap=cmap, alpha=alpha, vmin=0.0, vmax=1.0
        )
        ax = axes[ax_idx]
        ax.imshow(overlay_img)
        ax.set_title(f"t={t}")
        ax.axis("off")

    plt.tight_layout()
    return fig
