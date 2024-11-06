import numpy
from matplotlib import animation, patches, pyplot

if __name__ == '__main__':
    fig, (axl, axr) = pyplot.subplots(
        ncols=2,
        sharey=True,
        figsize=(6, 2),
        gridspec_kw=dict(width_ratios=[1, 3], wspace=0),)
    axl.set_aspect(1)
    axr.set_box_aspect(1/3)
    axr.yaxis.set_visible(False)
    axr.xaxis.set_ticks([0, numpy.pi, 2 * numpy.pi], ["0", r"$\pi$", r"$2\pi$"])
    
    # draw circle with initial point in left Axes
    x = numpy.linspace(0, 2 * numpy.pi, 50)
    axl.plot(numpy.cos(x), numpy.sin(x), "k", lw=0.3)
    point, = axl.plot(0, 0, "o")
    
    # draw full curve to set view limits in right Axes
    sine, = axr.plot(x, numpy.sin(x))
    
    # draw connecting line between both graphs
    con = patches.ConnectionPatch(
        (1, 0),
        (0, 0),
        "data",
        "data",
        axesA=axl,
        axesB=axr,
        color="C0",
        ls="dotted",
    )
    fig.add_artist(con)
    
    def animate(i):
        x = numpy.linspace(0, i, int(i * 25 / numpy.pi))
        sine.set_data(x, numpy.sin(x))
        x, y = numpy.cos(i), numpy.sin(i)
        point.set_data([x], [y])
        con.xy1 = x, y
        con.xy2 = i, y
        return point, sine, con

    animation = animation.FuncAnimation(
        fig,
        animate,
        interval=40,
        blit=False,  # blitting can't be used with Figure artists
        frames=x,
        repeat=False,)
    animation.save("temp/sine_animation.gif", writer="pillow")
    pyplot.show()
