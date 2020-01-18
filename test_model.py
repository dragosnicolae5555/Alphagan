import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix



def test_model(model,epoch,test_X,test_y,experiment):
    inliers = []
    outliers = []
    for x,label in zip(test_X,test_y):
        sum_of_losses = np.abs(model.get_losses(x.reshape((1,-1))))
        if label == 0:
            inliers.append(sum_of_losses)
        else:
            outliers.append(sum_of_losses)

    inliers = np.array(inliers)
    outliers = np.array(outliers)
    maximum = max(inliers.max(),outliers.max())

    # Converting the losses into probabilities.
    adjusted_outliers = outliers/maximum
    adjusted_inliers = inliers/maximum

    thresh_hold = (np.mean(adjusted_outliers)+np.mean(adjusted_inliers))/2

    predictions_in = adjusted_inliers<thresh_hold
    predicted_out = adjusted_outliers> thresh_hold

    inl = np.zeros_like(predictions_in)
    outl = np.ones_like(predicted_out)

    y_true = np.concatenate((inl,outl))
    y_pred = np.concatenate((predictions_in,predicted_out))
    y_pred_prob = np.concatenate((adjusted_inliers,adjusted_inliers))


    experiment.log_metric("F1-score", f1_score(y_true, y_pred), step=epoch)
    experiment.log_metric("Accuracy", accuracy_score(y_true, y_pred), step=epoch)
    experiment.log_metric("AUC", auc(y_true, y_pred_prob), step=epoch)
    plot_density(inliers, outliers, experiment,train=False)


def plot_density(inl, outl, experiment,train=False,show_chart=False):
    sns.distplot(inl, hist=True, kde=True,
                hist_kws={'edgecolor': 'black'},
                kde_kws={'linewidth': 4}, label='inliers')

    sns.distplot(outl, hist=True, kde=True,
                hist_kws={'edgecolor': 'black'},
                kde_kws={'linewidth': 4}, label='outliers')
    if train:
        plt.title('Train dist of L2 norms of inliers and outliers')
    else:
        plt.title('Test dist of L2 norms of inliers and outliers')
    plt.xlabel('Sum of Loss')
    plt.ylabel('Sample Count')
    plt.legend()
    experiment.log_figure(figure=plt)
    if show_chart:
        plt.show()