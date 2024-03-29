"""Convenient plot functions for reservoir model."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from mpl_tools import place, place_ax
from mpl_tools.misc import axprops

coord_type = "absolute"
"""Define scaling of `Plot2D.plt_field` axes.
- "relative": `(0, 1)  x (0, 1)`
- "absolute": `(0, Lx) x (0, Ly)`
- "index"   : `(0, Ny) x (0, Ny)`
"""

# Colormap for saturation
lin_cm = mpl.colors.LinearSegmentedColormap.from_list
cm_ow = lin_cm("", [(0, "#1d9e97"), (.3, "#b2e0dc"), (1, "#f48974")])
# cOil, cWater = "red", "blue"        # Plain
# cOil, cWater = "#d8345f", "#01a9b4" # Pastel/neon
# cOil, cWater = "#e58a8a", "#086972" # Pastel
# ccnvrt = lambda c: np.array(mpl.colors.colorConverter.to_rgb(c))
# cMiddle = .3*ccnvrt(cWater) + .7*ccnvrt(cOil)
# cm_ow = lin_cm("", [cWater, cMiddle, cOil])


styles = dict(
    default = dict(
        title  = "",
        transf = lambda x: x,
        cmap   = "viridis",
        levels = 10,
        cticks = None,
        # Note that providing vmin/vmax (and not a levels list) to mpl
        # yields prettier colobar ticks, but destorys the consistency
        # of the colorbars from one figure to another.
        locator = None,
    ),
    oil = dict(
        title  = "Oil saturation",
        transf = lambda x: 1 - x,
        cmap   = cm_ow,
        levels = np.linspace(0 - 1e-7, 1 + 1e-7, 20),
        cticks = np.linspace(0, 1, 6),
    ),
)
"""Default `Plot2D.plt_field` plot styling values."""


class Plot2D:
    """Plots specialized for 2D fields."""

    def plt_field(self, ax, Z, style="default", wells=True,
                  argmax=False, colorbar=True, labels=True, grid=False,
                  finalize=True, **kwargs):
        """Contour-plot of the (flat) unravelled field `Z`.

        `kwargs` falls back to `styles[style]`, which falls back to `styles['defaults']`.
        """
        # Populate kwargs with fallback style
        kwargs = {**styles["default"], **styles[style], **kwargs}
        # Pop from kwargs. Remainder goes to countourf
        ax.set(**axprops(kwargs))
        cticks = kwargs.pop("cticks")

        # Why extent=(0, Lx, 0, Ly), rather than merely changing ticks?
        # set_aspect("equal") and mouse hovering (reporting x,y).
        if "rel" in coord_type:
            Lx, Ly = 1, 1
        elif "abs" in coord_type:
            Lx, Ly = self.Lx, self.Ly
        elif "ind" in coord_type:
            Lx, Ly = self.Nx, self.Ny
        else:
            raise ValueError(f"Unsupported coord_type: {coord_type}")

        # Apply transform
        Z = np.asarray(Z)
        Z = kwargs.pop("transf")(Z)

        # Need to transpose coz orientation is model.shape==(Nx, Ny),
        # while contour() displays the same orientation as array printing.
        Z = Z.reshape(self.shape).T

        # Did we bother to specify set_over/set_under/set_bad ?
        has_out_of_range = getattr(kwargs["cmap"], "_rgba_over", None) is not None

        # Unlike `ax.imshow(Z[::-1])`, `contourf` does not simply fill pixels/cells (but
        # it does provide nice interpolation!) so there will be whitespace on the margins.
        # No fix is needed, and anyway it would not be trivial/fast,
        # ref https://github.com/matplotlib/basemap/issues/406 .
        collections = ax.contourf(
            Z, **kwargs,
            # origin=None,  # ⇒ NB: falsely stretches the field!!!
            origin="lower",
            extent=(0, Lx, 0, Ly),
            extend="both" if has_out_of_range else "neither",
        )

        # Contourf does not plot (at all) the bad regions. "Fake it" by facecolor
        if has_out_of_range:
            ax.set_facecolor(getattr(kwargs["cmap"], "_rgba_bad", "w"))

        # Grid (reflecting the model grid)
        # NB: If not showing grid, then don't locate ticks on grid, because they're
        #     generally uglier that mpl's default/automatic tick location. But, it
        #     should be safe to go with 'g' format instead of 'f'.
        ax.xaxis.set_major_formatter('{x:.3g}')
        ax.yaxis.set_major_formatter('{x:.3g}')
        ax.tick_params(which='minor', length=0, color='r')
        ax.tick_params(which='major', width=1.5, direction="in")
        if grid:
            n1 = 10
            xStep = 1 + self.Nx//n1
            yStep = 1 + self.Ny//n1
            ax.xaxis.set_major_locator(MultipleLocator(self.hx*xStep))
            ax.yaxis.set_major_locator(MultipleLocator(self.hy*yStep))
            ax.xaxis.set_minor_locator(MultipleLocator(self.hx))
            ax.yaxis.set_minor_locator(MultipleLocator(self.hy))
            ax.grid(True, which="both")

        # Axis lims
        ax.set_xlim((0, Lx))
        ax.set_ylim((0, Ly))
        ax.set_aspect("equal")
        # Axis labels
        if labels:
            if "abs" in coord_type:
                ax.set_xlabel("x")
                ax.set_ylabel("y")
            else:
                ax.set_xlabel(f"x ({coord_type})")
                ax.set_ylabel(f"y ({coord_type})")

        # Add well markers
        if wells:
            if wells == "color":
                wells = {"color": [f"C{i}" for i in range(len(self.prd_xy))]}
            elif wells in [True, 1]:
                wells = {}
            self.well_scatter(ax, self.prd_xy, False, **wells)
            wells.pop("color", None)
            self.well_scatter(ax, self.inj_xy, True, **wells)

        # Add argmax marker
        if argmax:
            idx = Z.T.argmax()  # reverse above transpose
            xy = self.ind2xy(idx)
            for c, ms in zip(['b', 'r', 'y'],
                             [10, 6, 3]):
                ax.plot(*xy, "o", c=c, ms=ms, label="max", zorder=98)

        # Add colorbar
        if colorbar:
            if isinstance(colorbar, type(ax)):
                cax = dict(cax=colorbar)
            else:
                cax = dict(ax=ax, shrink=.8)
            ax.figure.colorbar(collections, **cax, ticks=cticks)

        tight_show(ax.figure, finalize)
        return collections


    def well_scatter(self, ax, ww, inj=True, text=None, color=None, size=1):
        """Scatter-plot the wells of `ww` onto a `Plot2D.plt_field`."""
        # Well coordinates
        ww = self.sub2xy(*self.xy2sub(*ww.T)).T
        # NB: make sure ww array data is not overwritten (avoid in-place)
        if   "rel" in coord_type: s = 1/self.Lx, 1/self.Ly                     # noqa
        elif "abs" in coord_type: s = 1, 1                                     # noqa
        elif "ind" in coord_type: s = self.Nx/self.Lx, self.Ny/self.Ly         # noqa
        else: raise ValueError("Unsupported coordinate type: %s" % coord_type) # noqa
        ww = ww * s

        # Style
        if inj:
            c  = "w"
            ec = "gray"
            d  = "k"
            m  = "v"
        else:
            c  = "k"
            ec = "gray"
            d  = "w"
            m  = "^"

        if color:
            c = color

        # Markers
        sh = ax.plot(*ww.T, 'r.', ms=3, clip_on=False)
        sh = ax.scatter(*ww.T, s=(size * 26)**2, c=c, marker=m, ec=ec,
                        clip_on=False,
                        zorder=1.5,  # required on Jupypter
                        )

        # Text labels
        if text is not False:
            for i, w in enumerate(ww):
                if not inj:
                    w[1] -= 0.01
                ax.text(*w[:2], i if text is None else text,
                        color=d, fontsize=size*12, ha="center", va="center")

        return sh

    def plt_production(self, ax, production, obs=None,
                       legend_outside=True, finalize=True):
        """Production time series. Multiple wells in 1 axes => not ensemble compat."""
        hh = []
        tt = 1+np.arange(len(production))
        for i, p in enumerate(1-production.T):
            hh += ax.plot(tt, p, "-", label=i)

        if obs is not None:
            for i, y in enumerate(1-obs.T):
                ax.plot(tt, y, "*", c=hh[i].get_color())

        # Add legend
        if legend_outside:
            kws = dict(
                  bbox_to_anchor=(1, 1),
                  loc="upper left",
                  ncol=1+len(production.T)//10,
            )
        else:
            kws = dict(loc="lower left")
        ax.legend(title="Well #.", **kws)

        ax.set_title("Oil saturation in producers")
        ax.set_xlabel("Time index")
        # ax.set_ylim(-0.01, 1.01)
        ax.axhline(0, c="xkcd:light grey", ls="--", zorder=1.8)
        ax.axhline(1, c="xkcd:light grey", ls="--", zorder=1.8)

        tight_show(ax.figure, finalize)
        return hh

    # Note: See note in mpl_setup.py about properly displaying the animation.
    def anim(self, wsats, prod, title="", figsize=(10, 3.5), pause=200, animate=True,
             **kwargs):
        """Animate the saturation and production time series."""

        # Create figure and axes
        title = "Animation" + ("-- " + title if title else "")
        fig, (ax1, ax2) = place.freshfig(title, ncols=2, figsize=figsize,
                                         gridspec_kw=dict(width_ratios=(2, 3)))
        fig.suptitle(title)  # coz animation never (any backend) displays title
        # Saturations
        kwargs.update(wells="color", colorbar=True, finalize=False)
        ax2.cc = self.plt_field(ax2, wsats[-1], "oil", **kwargs)
        # Production
        hh = self.plt_production(ax1, prod, legend_outside=False, finalize=False)
        fig.tight_layout()

        if animate:
            from matplotlib import animation
            tt = np.arange(len(wsats))

            def update_fig(iT):
                # Update field
                for c in ax2.cc.collections:
                    try:
                        ax2.collections.remove(c)
                    except (AttributeError, ValueError):
                        pass  # occurs when re-running script
                kwargs.update(wells=False, colorbar=False)
                ax2.cc = self.plt_field(ax2, wsats[iT], "oil", **kwargs)

                # Update production lines
                if iT >= 1:
                    for h, p in zip(hh, prod.T):
                        h.set_data(tt[1:1+iT], 1 - p[:iT])

            ani = animation.FuncAnimation(
                fig, update_fig, len(tt), blit=False, interval=pause,
                # Prevent busy/idle indicator constantly flashing, despite %%capture
                # and even manually clearing the output of the calling cell.
                repeat=False,  # flashing stops once the (unshown) animation finishes.
                # An alternative solution is to do this in the next cell:
                # animation.event_source.stop()
                # but it does not work if using "run all", even with time.sleep(1).
            )

            return ani

def tight_show(figure, enabled):
    if enabled:
        figure.tight_layout()
        plt.show()
