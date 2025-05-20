import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, roc_curve,roc_auc_score,brier_score_loss
import statsmodels.api as sm
import numpy as np
from math import sqrt

def model_performance_plots(df):

    # Plotting encounter statuses frequency distribution
    status_counts = df['status_name'].value_counts().sort_values(ascending=True)
    print(status_counts)

    # Plotting calibration plot
    
    # Calculate the percentage of event occurances for each bin
    percentages = df.groupby('probability_bin', observed=False)['event_binary'].apply(lambda x: (x == 1).mean() * 100)
    proportions = percentages.values / 100
    
    # Calculating confidence intervals
    encounters_per_bin = df['probability_bin'].value_counts()
    cilb, ciub = sm.stats.proportion_confint(
        count=proportions * encounters_per_bin,
        nobs=encounters_per_bin.values,
        alpha=0.05)
    cilb_prc = cilb * 100
    ciub_prc = ciub * 100
    lower_error = cilb_prc
    upper_error = ciub_prc

    # Find indices of bins with values
    valid_bin_indices = np.where(percentages.notnull())[0]
    #valid_bin_indices=valid_bin_indices[0:-1]
    predicted_rates = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]

    # Finding R2 value
    actual_rates = percentages.values[valid_bin_indices]
    actual_rates = [0 if pd.isna(value) else value for value in actual_rates]
    print(actual_rates)
    print(predicted_rates[:len(valid_bin_indices)])
    # Predicted values (regression line)
    # Calculate the mean of the actual values (observations)
    mean_observations = np.mean(actual_rates)
    # Calculate the total sum of squares (SST)
    sst = np.sum((actual_rates - mean_observations) ** 2)
    # Calculate the sum of squares of residuals (SSR)
    ssr = np.sum((np.array(actual_rates) - np.array(predicted_rates[:len(valid_bin_indices)])) ** 2)
    # Calculate R-squared
    r2_val = 1 - (ssr / sst)
    #r2_val = r2_score(actual_no_show_rates, predicted_no_show_rates[:len(valid_bin_indices)])
    print(f'R^2 Value: {r2_val:.2f}')
    # Finding confidence interval for R2 value
    # 67%CI = R2 ± SER2
    # 95%CI = R2 ± 1.96SER2
    n=df.shape[0]
    k=1
    r2=r2_val
    if r2 < 0: r2=0
    SER2 = sqrt((4*r2*((1-r2)**2)*((n-k-1)**2))/((n**2-1)*(n + 3)))

    # Calculating Brier score
    brier_score=brier_score_loss(df['event_binary'], df['prob_1'])
    print(f'Brier Score: {brier_score:.2f}')

    # Calculating ECE
    # Modified from guide: https://towardsdatascience.com/expected-calibration-error-ece-a-step-by-step-visual-explanation-with-python-code-c3e9aa12937d
    M=5
    bin_boundaries = np.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    confidences = df['prob_max'].to_numpy()
    predicted_label = df['event_binary_predicted'].to_numpy()
    true_labels=df['event_binary'].to_numpy()
    accuracies = predicted_label==true_labels
    ece = np.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Determines if sample is in bin m (between bin lower & upper)
        in_bin = np.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
        # Calculates the empirical probability of a sample falling into bin m: (|Bm|/n)
        prob_in_bin = in_bin.mean()
        if prob_in_bin.item() > 0:
            # Gets the accuracy of bin m: acc(Bm)
            accuracy_in_bin = accuracies[in_bin].mean()
            # Gets the average confidence of bin m: conf(Bm)
            avg_confidence_in_bin = confidences[in_bin].mean()
            # Calculates |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
    print(f'ECE: {ece[0]:.2f}')

    fig, ax = plt.subplots()

    # Plotting confidence intervals
    plt.errorbar(
        percentages.index[valid_bin_indices],
        percentages.values[valid_bin_indices],
        yerr=[lower_error[valid_bin_indices], upper_error[valid_bin_indices]],
        fmt='o',
        color='grey',
        label='95% Confidence Intervals'
    )
    # Line chart to visualize the observed rate the event occured
    plt.plot(
        percentages.index[valid_bin_indices],
        percentages.values[valid_bin_indices],
        marker='o',
        label= 'Observed Rate'
        #label= 'Observed Rate (R² = {:.2f})'.format(r2_val)
    )
    # Line chart to visualize the rate the event was predicted to occur
    plt.plot(
        percentages.index[valid_bin_indices][:len(predicted_rates)],
        predicted_rates[:len(valid_bin_indices)],
        linestyle='--',
        color='red',
        label='Predicted Rate'
    )
    
    plt.xlabel('Predicted Probability (%)')
    plt.ylabel('Rate (%)')
    plt.title('Calibration Plot for Prediction Model')
    plt.ylim([-5, 105])
    plt.legend(loc='upper left')
    plt.savefig('calibrationplot'+'.svg')
    plt.savefig('calibrationplot'+'.png')
    plt.show()
    
    plt.clf()
    
    print(encounters_per_bin)
    
    # Plotting ROC curve
    
    fpr, tpr, threshold = roc_curve(df['event_binary'], df['probability_percent'])
    roc_auc = roc_auc_score(df['event_binary'], df['probability_percent'])
    threshold= threshold / 100
    # Finding the maximum Youden's index, which is often used as a criterion for selecting the optimal threshold.
    # Represented graphically as the height above the random classifier line.
    j_scores= tpr - fpr
    best_threshold_index= j_scores.argmax()
    best_threshold= threshold[best_threshold_index]
    # Finding confidence interval for AUC
    # 67%CI = AUC ± SE_AUC
    # 95%CI = AUC ± 1.96SE_AUC
    N1 = sum(df['event_binary'] == 1)
    N2 = sum(df['event_binary'] != 1)
    Q1 = roc_auc / (2 - roc_auc)
    Q2 = 2*roc_auc**2 / (1 + roc_auc)
    SE_AUC = sqrt((roc_auc*(1 - roc_auc) + (N1 - 1)*(Q1 - roc_auc**2) + (N2 - 1)*(Q2 - roc_auc**2)) / (N1*N2))

    # print(f"The best threshold Based on Youden's Index is: {best_threshold}")
    # Plotting ROC curve, random classifier line, threshold, and best threshold
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0,1], [0,1], 'k--', label='Random Classifier')
    #plt.plot(fpr, threshold, label='Threshold')
    best_fpr= fpr[best_threshold_index]
    best_tpr= tpr[best_threshold_index]
    #plt.scatter(best_fpr, best_tpr, color='darkorange', label=f'Best Threshold = {best_threshold*100:0.2f}%', zorder=5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Prediction Model')
    plt.legend(loc='upper left')
    plt.savefig('roccurve'+'.svg')
    plt.savefig('roccurve'+'.png')

    plt.show()

    event_count=(df['event_binary'] == 1).sum()
    event_prop=event_count/n
    data = [[n, event_count, event_prop, r2_val, SER2, roc_auc, SE_AUC]]
    series = pd.DataFrame(data, columns=['N', 'Event Count', 'Event Prop', 'R2', 'SE R2', 'AUC', 'SE AUC'])
    
    return series

if __name__ == "__main__":

    # Creating an example dataframe to use an input in the function
    # Status A is the event we are trying to predict in this scenario
    dataset = {'status_name': ['A', 'B', 'C', 'A', 'A', 'B', 'B', 'A', 'C', 'D', 'A', 'B', 'A', 'A', 'C', 'A', 'D', 'D', 'B'], 
            'probability_percent': [30.1, 39.2, 75.8, 70.2, 45.6, 0.1, 12.1, 90.1, 0.5, 67.3, 89.7, 21.3, 56.7, 91.5, 42.8, 66.4, 5.4, 15.1, 51.1]}
    df = pd.DataFrame(dataset)
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']
    df['probability_bin'] = pd.cut(df['probability_percent'], bins=bins, labels=labels)
    df['event_binary'] = df['status_name'].apply(lambda x: 1 if x=='A' else 0)
    df['event_binary_predicted'] = np.where(df['probability_percent'] <50, 0, np.where(df['probability_percent'] >= 50, 1, np.nan)).astype(int)
    df['prob_1'] = df['probability_percent']/100
    df['prob_0'] = 1-df['prob_1']
    df['prob_max'] = df[['prob_1', 'prob_0']].max(axis=1)
    
    res=model_performance_plots(df)
