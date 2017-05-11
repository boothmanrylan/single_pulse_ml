import numpy as np
import matplotlib 
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib import gridspec

def plot_gallery(data_arr, titles, h, w, n_row=3, n_col=4, 
                    figname=None, cmap='RdBu', suptitle=''):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.suptitle(suptitle, fontsize=35, color='blue', alpha=0.5)
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(min(n_row * n_col, len(data_arr))):
        d_arr = data_arr[i].reshape((h, w))
        d_arr -= np.median(d_arr)
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(d_arr, cmap=cmap, aspect='auto')        
        plt.title(titles[i], size=12, color='red')
        plt.xticks(())
        plt.yticks(())
    if figname:
    	plt.savefig(figname)


def get_title(y, target_names):
    prediction_titles = y.astype(str)
    prediction_titles[prediction_titles=='0'] = target_names[0]
    prediction_titles[prediction_titles=='1'] = target_names[1]

    return prediction_titles

def get_title2(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]]
    true_name = target_names[y_test[i]]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)


def plot_image_probabilities(FT_arr, DT_arr, FT_prob_spec, DT_prob_spec):

    assert (len(FT_arr.shape)==2) and (len(DT_arr.shape)==2), "Input data should be (nfreq, ntimes)"

    gs2 = gridspec.GridSpec(4, 3)
    ax1 = plt.subplot(gs2[:2, :2])
    ax1.xaxis.set_ticklabels('')
    ax1.yaxis.set_ticklabels('')
    plt.ylabel('Freq', fontsize=18)
    plt.xlabel('Time', fontsize=18)
    ax1.imshow(FT_arr, cmap='RdBu', interpolation='nearest', aspect='auto')

    ax2 = plt.subplot(gs2[:2, 2:])
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position('right')
    plt.ylabel('probability', fontsize=18)
    ax2.plot(FT_prob_spec)
    ax2.semilogy()

    ax3 = plt.subplot(gs2[2:, :2])
    ax3.xaxis.set_ticklabels('')
    ax3.yaxis.set_ticklabels('')
    plt.ylabel('Freq', fontsize=18)
    plt.xlabel('Time', fontsize=18)
    ax3.imshow(DT_arr, cmap='RdBu', interpolation='nearest', aspect='auto')

    ax4 = plt.subplot(gs2[2:, 2:])
    ax4.yaxis.set_label_position('right')
    ax4.yaxis.tick_right()
    plt.ylabel('probability', fontsize=18)
    ax4.bar([0, 1], DT_prob_spec)
    plt.xticks([0.5, 1.5], ['RFI', 'Pulse'])
    #ax4.semilogy()
