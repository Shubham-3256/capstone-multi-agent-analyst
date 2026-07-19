"""Style theme mapping providing color palettes and formatting grids."""

from typing import Any

import matplotlib as mpl

from src.core.logger import get_logger

logger = get_logger(__name__)


class Theme:
    """Design configurations model exposing background, font and categorical colors."""

    def __init__(
        self,
        name: str,
        background_color: str,
        text_color: str,
        grid_color: str,
        color_palette: list[str],
    ) -> None:
        """Initialize Theme structure.

        Args:
            name: Theme key.
            background_color: Hex color string.
            text_color: Hex color string.
            grid_color: Hex color string.
            color_palette: Palette array.
        """
        self.name = name
        self.background_color = background_color
        self.text_color = text_color
        self.grid_color = grid_color
        self.color_palette = color_palette


class ThemeManager:
    """Resolves professional light, dark, and corporate themes for matplotlib and plotly engines."""

    # Curated color configurations
    THEMES: dict[str, Theme] = {
        "light": Theme(
            name="light",
            background_color="#FFFFFF",
            text_color="#2D3748",
            grid_color="#E2E8F0",
            color_palette=[
                "#3182CE",
                "#E53E3E",
                "#38A169",
                "#D69E2E",
                "#805AD5",
                "#319795",
            ],
        ),
        "dark": Theme(
            name="dark",
            background_color="#1A202C",
            text_color="#EDF2F7",
            grid_color="#4A5568",
            color_palette=[
                "#63B3ED",
                "#FC8181",
                "#68D391",
                "#F6AD55",
                "#B794F4",
                "#4FD1C5",
            ],
        ),
        "corporate": Theme(
            name="corporate",
            background_color="#F7FAFC",
            text_color="#1A202C",
            grid_color="#E2E8F0",
            color_palette=[
                "#1A365D",
                "#742A2A",
                "#22543D",
                "#744210",
                "#44337A",
                "#234E52",
            ],
        ),
    }

    @classmethod
    def apply_matplotlib_theme(cls, theme_name: str = "corporate") -> Theme:
        """Configure matplotlib rcParams to use styling settings.

        Args:
            theme_name: Key of selected theme.

        Returns:
            Theme: Theme object.
        """
        theme = cls.THEMES.get(theme_name.lower(), cls.THEMES["corporate"])
        logger.info(f"ThemeManager: Applying Matplotlib theme: {theme.name.upper()}")

        # Update rcParams
        mpl.rcParams["figure.facecolor"] = theme.background_color
        mpl.rcParams["axes.facecolor"] = theme.background_color
        mpl.rcParams["axes.edgecolor"] = theme.grid_color
        mpl.rcParams["axes.labelcolor"] = theme.text_color
        mpl.rcParams["xtick.color"] = theme.text_color
        mpl.rcParams["ytick.color"] = theme.text_color
        mpl.rcParams["text.color"] = theme.text_color
        mpl.rcParams["grid.color"] = theme.grid_color
        mpl.rcParams["grid.alpha"] = 0.5
        mpl.rcParams["axes.grid"] = True

        # Apply color cycle
        mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=theme.color_palette)

        return theme

    @classmethod
    def get_plotly_layout(cls, theme_name: str = "corporate") -> dict[str, Any]:
        """Generate dict templates of theme parameters for Plotly configurations.

        Args:
            theme_name: Key of selected theme.

        Returns:
            Dict[str, Any]: Layout parameters.
        """
        theme = cls.THEMES.get(theme_name.lower(), cls.THEMES["corporate"])

        plotly_template = "plotly_white" if theme.name != "dark" else "plotly_dark"

        return {
            "template": plotly_template,
            "paper_bgcolor": theme.background_color,
            "plot_bgcolor": theme.background_color,
            "font": {"color": theme.text_color},
            "colorway": theme.color_palette,
            "xaxis": {"gridcolor": theme.grid_color, "linecolor": theme.grid_color},
            "yaxis": {"gridcolor": theme.grid_color, "linecolor": theme.grid_color},
        }
