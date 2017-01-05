import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save_fig(path, G):
    fig = plt.gcf()
    fig.subplots_adjust(left=0,bottom=0,right=1,top=1)
    rows = 10
    cols = G.shape[0]/rows
    nums = rows*cols
    for i in range(nums):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.axis("off")
        ax.imshow(G[i,:,:,:])
        plt.savefig(path)
        plt.pause(0.01)
