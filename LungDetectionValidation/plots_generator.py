from sklearn.metrics import (
    auc, precision_recall_curve
)
import matplotlib.pyplot as plt


def precision_recall(y_true, scores):
    
    """
    source:
    https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/#:~:text=Precision%2DRecall%20Curves%20in%20Python&text=The%20precision%20and%20recall%20can,precision%2C%20recall%20and%20threshold%20values.
    Precision-Recall curves should be used when there is a moderate to large class imbalance.
    -articles in the source confirm phrase above
    """
    
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    lr_recall = auc(recall, precision)
    
    fig = plt.figure(figsize=(8,8))
    plt.plot(recall, precision, linewidth=2,
             color='darkorange',
             label="Precision-Recall curve (area = %0.4f)" % lr_recall)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0,1.0])
    plt.ylim([0,1.0])
    plt.grid(True)
    plt.title("Precision-Recall curve")
    plt.legend(loc="lower right")
    ax = plt.gca()
    ax.set_aspect('equal')

    """
    source:
    https://medium.com/data-hackers/como-avaliar-seu-modelo-de-classifica%C3%A7%C3%A3o-34e6f6011108
    Precision-Recall-Threshold makes it easy to see the relationship between precision and recall for any given threshold. The ideal threshold setting is the highest possible recall and precision rate.
    """
    fig1, ax1 = plt.subplots(figsize = (12,3))
    plt.plot(thresholds, precision[:-1], 'b--', label = 'Precisão')
    plt.plot(thresholds, recall[:-1], 'g-', label = 'Recall')
    plt.xlabel('Threshold')
    plt.legend(loc = 'center right')
    plt.title('Precisão x Recall', fontsize = 14)
    
    return fig, fig1
