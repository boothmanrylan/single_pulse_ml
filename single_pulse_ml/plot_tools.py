import matplotlib 
matplotlib.use('Agg')

import matplotlib.pyplot as plt

def plot_gallery(data_arr, titles, h, w, n_row=3, n_col=4, figname=None):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
#        plt.imshow(data_arr[i].reshape((h, w)), cmap=plt.cm.gray, aspect='auto')
        plt.imshow(data_arr[i].reshape((h, w)), cmap='Greys', aspect='auto')        
        plt.title(titles[i], size=14)
        plt.xticks(())
        plt.yticks(())
    if figname:
    	plt.savefig(figname)


def get_title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)