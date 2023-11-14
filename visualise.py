import matplotlib.pyplot as plt

def visualise_feature_importance(models, figfold, config_str=""): 
    """
    Visualising feature importance using the DecisionTree()'s build in feature importance measure
    and plots a histogram of the feature importance of each feature.

    Parameters:
        models (list): list of models to be analyzed   
    
    """

    feature_importance = {} # Save features as a dict.
    for model in models: # Loop through all models
        fnames = model.feature_names_in_    # get feature names
        fimp = model.feature_importances_   # and corresponding importances

        for i in list(zip(fnames,fimp)):
            if i[0] in feature_importance.keys():
                feature_importance[i[0]] = feature_importance[i[0]] + [i[1]] # add importance to list if exists
            else:
                feature_importance[i[0]] =[i[1]] # create new list if first encounter with feature

    # Finally, loop through dict and plot histograms
    for feature in feature_importance.keys():
        plt.hist(feature_importance[feature])
        plt.title(f"{feature} {config_str}")
        plt.xlim([-0.01,0.6]) # add common x-axis for ease of comparison.
        plt.savefig(figfold + feature + '_dist.png')
        plt.clf()

def visualise_results(train_scores, test_scores, score_names, figfold, config_str=""):
    """
    Plots histograms of each metric on both train and test set in a single figure.

    Parameters:
        train_scores (pd.DataFrame): DataFrame of train scores
        test_scores (pd.DataFrame): DataFrame of test scores
        score_names (list): list of score names
    """
    fig, axs = plt.subplots(len(score_names)//2 + len(score_names)%2, 2, figsize=(8, 6)) 
    
    for i, score in enumerate(score_names):
        j = i%2 
        k = i//2
        axs[j,k].hist(train_scores[score], label='train')
        axs[j,k].hist(test_scores[score], label='test')
        axs[j,k].set_title(f'{score} {config_str}')
        axs[j,k].legend()
        axs[k,j].set_xlim([0.65 ,1.01])

    plt.savefig(figfold + 'result_hist.png')
    plt.clf()
