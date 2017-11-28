import pickle

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

video_data = load_obj("video_pred_data")

# print len(video_data)
import numpy as np
from sklearn.manifold import TSNE
l = []
names = []
for item in video_data:
	vec = video_data[item][1]
	l.append(vec)
	names.append(video_data[item][0])
l = np.array(l)
l = TSNE(n_components=2).fit_transform(l)
x, y = l.T
from matplotlib import pyplot as plt
import matplotlib as m
# cm = m.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)
norm = m.colors.Normalize(vmin=1.5, vmax=4.5)
fig,ax = plt.subplots()
c = norm(y)
cmap = plt.cm.rainbow
sc = plt.scatter(x,y, c=c, cmap=cmap)
names = np.array(names)
annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

def update_annot(ind):
	annot.set_visible(False)
	pos = sc.get_offsets()[ind["ind"][0]]
	annot.xy = pos
	text = "{}, {}".format(" ".join(list(map(str,ind["ind"]))), 
							" ".join([names[n] for n in ind["ind"]]))
	annot.set_text(text)
	annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
	annot.get_bbox_patch().set_alpha(0.4)


def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", hover)
plt.colorbar()
plt.show()