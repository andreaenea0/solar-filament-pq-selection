"""Illustrate the competing tree configurations on one held-out training observation."""

from tools.figures.palette import AMBER, BLUE, COMPARISON_GREEN
from report.observed_pipeline import PENALTY, STYLE, arguments, contour, observed, plt, save


def draw(data, directory):
    """Place the score arithmetic below the relevant masks, outside the solar image."""
    image, _, masks, gains, levels, drawings = data
    fig, axes = plt.subplots(2, 2, figsize=(3.33, 3.30))
    fig.subplots_adjust(left=.02, right=.98, top=.91, bottom=.05, wspace=.12, hspace=.50)
    for ax in axes.flat:
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_axis_off()
    top, bottom = axes
    for ax, title in zip(axes.flat, ("Parent alternative", "Retained pair", "First drawing", "Second drawing")):
        ax.set_title(title, fontsize=8.7, pad=3)
    contour(top[0], masks[0], AMBER)
    for i, mask in enumerate(masks[1:], 1):
        contour(top[1], mask, BLUE)
        yy, xx = mask.nonzero()
        top[1].text(xx.mean(), yy.mean(), str(i), color="white", size=8, ha="center", va="center",
                    bbox={"boxstyle":"circle,pad=.10", "fc":BLUE, "ec":"white", "lw":.45})
    weights = gains - PENALTY
    for ax, line in zip(top, (f"Weight  {weights[0]:.3f}",
                               f"{weights[1]:.3f} + {weights[2]:.3f} = {sum(weights[1:]):.3f}")):
        ax.text(.5, -.075, line, transform=ax.transAxes, ha="center", va="top", fontsize=7.5)
    for ax, record in zip(bottom, drawings, strict=True):
        for drawn in record:
            contour(ax, drawn, COMPARISON_GREEN)
    print("Probability thresholds for parent and retained descendants:", levels)
    save(fig, directory, "observed_selection")


if __name__ == "__main__":
    args = arguments()
    with plt.rc_context(STYLE):
        draw(observed(args), args.out)
