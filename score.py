from sklearn.metrics import confusion_matrix 

# Helper functions 
# Defining the metrics we want to and join them in a single function
def score(model, X, y, verbose=False):
    """
    Accepts a model (implementing a .predict method) and some data, X, with labels y, 
    and returns the accuracy, precision, recall and F1 score.

    Warning: the lenght of the return value is assumed to be equal to the lenght of the socre names

    """
    y_hat = model.predict(X)
    if verbose:
        print(confusion_matrix(y, y_hat))
        
    tn, fp, fn, tp = confusion_matrix(y, y_hat).ravel()
    acc = (tp + tn)/(tp + fp + fn + tn)
    prec = (tp)/(tp + fp)
    rec = (tp)/(tp + fn)
    f1 = 2*prec*rec/(prec+rec)

    return acc, prec, rec, f1



# Note: the shape of this list must match the shape of the return value of score().
score_names = ['Accuracy', 'Precision', 'Recall', 'F1']
