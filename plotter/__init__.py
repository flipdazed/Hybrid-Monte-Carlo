from pretty_plotting import Pretty_Plotter
from colours import *
from matplotlib.lines import Line2D
import random
from matplotlib import colors
# path is relative to the main directory at ../
PLOT_LOC = 'results/figures/'
ANIM_LOC = 'results/animations/'


# Fix colours/markers: A bug sometimes forces all colours the same
mlist = []
for m in Line2D.filled_markers: # generate a list of markers
    try:
        if len(m) == 1 and m != ' ':
            mlist.append(m)
    except TypeError:
        pass
# randomly order and create an iterable of markers to select from
marker = [i for i in random.sample(mlist, len(mlist))]

# generatte basic colours list
clist = [i for i in colors.ColorConverter.colors if i != 'w']
colour = [i for i in random.sample(clist, len(clist))]

# generate only dark colours
darkclist = [i for i in colors.cnames if 'dark' in i]
darkcolour = [i for i in random.sample(darkclist, len(darkclist))]
lightcolour = map(lambda strng: strng.replace('dark',''), darkcolour)

theory_colours = iter(darkcolour)
measured_colours = iter(lightcolour)
markers = iter(marker)