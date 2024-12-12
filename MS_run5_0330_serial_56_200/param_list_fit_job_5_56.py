
from pickle import TRUE
import pandas as pd
from boruta import BorutaPy
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import sklearn.metrics
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from hyperopt import hp
from hyperopt import STATUS_OK, Trials, fmin, tpe, space_eval, SparkTrials, hp
import optuna 
from optuna import Trial, visualization
from optuna.samplers import TPESampler
import joblib
import mlflow
import warnings
warnings.filterwarnings("ignore")
import pyspark
import datetime
import plotly.express as px
import os
import sys
from sklearn.metrics import precision_recall_curve, auc

# =============================================================================
# Set time now
# =============================================================================
start_time = datetime.datetime.now()
current_date = start_time.date()
current_time = start_time.time()

# =============================================================================
# Set file paths
# =============================================================================
output_dir = "tune_output_" + str(current_time)
os.makedirs(output_dir, exist_ok=True)
input_file = "/projects/siuyinlee@xsede.org/scripts/Data/compiled_all_tiles_clean_v2.csv"
my_data = pd.read_csv(input_file)
diagnostics_file = os.path.join(output_dir, 'Tune_Log' + str(current_date) + '.txt')
history_file = os.path.join(output_dir, 'Tune_History_' + str(current_date) + '.csv')
param_imp_plot = os.path.join(output_dir, 'Plot_Param_Imp_' + str(current_date) + '.html')
hist_plot = os.path.join(output_dir, 'Plot_Opt_History_' + str(current_date) + '.html')
slice_plot = os.path.join(output_dir, 'Plot_Slice_' + str(current_date) + '.html')
para_coor_plot = os.path.join(output_dir, 'Plot_Para_Coor' + str(current_date) + '.html')
lc_b = os.path.join(output_dir, 'lc_b' + str(current_date) + '.png')
prauc_plot = os.path.join(output_dir, 'Plot_PRAUC' + str(current_date) + '.html')
pr_thre_plot = os.path.join(output_dir, 'Plot_THRES_PR' + str(current_date) + '.html')

# =============================================================================
# Direct output to the diagnostic log file
# =============================================================================
original_stdout = sys.stdout
log_file = open(diagnostics_file, "w")
print(log_file)
sys.stdout = log_file

data_train_frac = 0.6
random_state = 25
data_subsample_frac = 1.0
data_shuffle = True
response_name = "fire"

predictor_choice = ["nbrt1_diff_3yr_over_mean",
                            "nbr2_difference_3yr",
                            "mirbi_difference_3yr",
                            "band6",
                            "nbrt1_z_score",
                            "ndmi",
                            "gemi_mean_3yr",
                            "nbr2_z_score",
                            "nbrt1_sd_3yr",
                            "savi_mean_3yr",
                            "TCbrightness_mean_3yr",
                            "mirbi",
                            "TCgreenness_diff_3yr_over_mean",
                            "nbr2",
                            "nbrt1_difference_3yr",
                            "TCgreenness_mean_3yr",
                            "nbr_difference_3yr",
                            "mirbi_z_score",
                            "vi6t_difference_3yr",
                            "TCbrightness_z_score",
                            "nbr2_diff_3yr_over_mean",
                            "TCgreenness_z_score",
                            "vi46_diff_3yr_over_mean",
                            "vi47_z_score",
                            "vi47_diff_3yr_over_mean",
                            "vi57_z_score",
                            "ndwi_z_score",
                            "nbr2_mean_3yr",
                            "ndvi_mean_3yr",
                            "evi_z_score",
                            "evi_sd_3yr",
                            "vi43_diff_3yr_over_mean",
                            "TCwetness_sd_3yr",
                            "band1",
                            "ndvi",
                            "ndvi_difference_3yr",
                            "TCwetness_diff_3yr_over_mean",
                            "vi43_sd_3yr",
                            "ndwi_diff_3yr_over_mean",
                            "vi43_z_score",
                            "TCgreenness_difference_3yr",
                            "bai_diff_3yr_over_mean",
                            "evi_difference_3yr",
                            "vi6t_z_score",
                            "nbrt1",
                            "TCgreenness",
                            "vi45_mean_3yr",
                            "bai_mean_3yr",
                            "vi43_mean_3yr",
                            "TCwetness_mean_3yr",
                            "bai_z_score",
                            "savi_difference_3yr",
                            "vi57_diff_3yr_over_mean",
                            "band7",
                            "evi",
                            "nbr_mean_3yr"]


if data_subsample_frac < 1.0:
    my_data = my_data.sample(frac=data_subsample_frac, random_state=random_state)
    my_data.reset_index(inplace=True)

    # shuffled to make sure data is not ordered by year
if data_shuffle:
    my_data = my_data.sample(frac=1.0, random_state=random_state)
    my_data.reset_index(inplace=True)

X_all = my_data[predictor_choice]
y_all = my_data[response_name]


X_all, X_valid, y_all, y_valid = train_test_split(
    X_all, y_all, test_size=0.2, random_state=random_state, stratify=y_all)

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=data_train_frac, random_state=random_state, stratify=y_all)

#the test set up to this point has 0.8*0.4 = 0.32 data
#So further split 50% will get us to 16% for test set and 16% for cv set
X_test, X_cv, y_test, y_cv = train_test_split(
    X_test, y_test, test_size=0.5, random_state=random_state, stratify=y_test)

print("Time now is ", str(start_time))
print("The current executed script is ", __file__)
print("Log file output directory is ", diagnostics_file)
print("Data input directory is ", input_file)

print('#' * 80)

print("data proportion used is ", data_subsample_frac)
print("predictor names are ", predictor_choice)
print("response name is ", response_name)
print("random state is set at ", random_state)

print('#' * 80)

#Create Optuna study
study = optuna.create_study(direction='maximize',sampler=TPESampler(multivariate=True))

#Create Objective function
def objective(trial: Trial,X,y) -> float:
#def objective(trial):
    joblib.dump(study, 'study.pkl')

    #X_train, X_test, y_train, y_test = train_test_split(
    #    X_all, y_all, test_size=data_train_frac, random_state=random_state, stratify=y_all)

    #the test set up to this point has 0.8*0.4 = 0.32 data
    #So further split 50% will get us to 16% for test set and 16% for cv set
    #X_test, X_cv, y_test, y_cv = train_test_split(
    #    X_test, y_test, test_size=0.5, random_state=random_state, stratify=y_test)

    #evaluation = [( X_train, y_train), ( X_test, y_test)]

    param = {
                "n_jobs": -1,
                "verbosity": 0,
                "n_estimators" : 1000,
                "objective" : "binary:logistic",
                "booster": "gbtree",
                'reg_alpha':trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda':trial.suggest_float('reg_lambda', 0, 10),
                'subsample': trial.suggest_float("subsample", 0.1, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
                "max_depth": trial.suggest_int("max_depth", 2, 15),
                "min_child_weight" : trial.suggest_int("min_child_weight", 2, 100),
                "learning_rate" : trial.suggest_float("learning_rate", 0, 1),
                "gamma" : trial.suggest_float("gamma", 0, 10),
                "grow_policy" : trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
                "max_delta_step" : trial.suggest_float("max_delta_step", 0, 10),
                "scale_pos_weight" : trial.suggest_float("scale_pos_weight", 0, 20),
                "use_label_encoder" : False
            }                  
    
    clf = XGBClassifier(**param)
    #clf.fit(X_train, y_train,
    #        eval_set=evaluation, eval_metric="aucpr",
    #        early_stopping_rounds=10,verbose=False)
    
    return cross_val_score(clf, X, y, cv=3, scoring="average_precision").mean()

#Run the optimizer
print('#' * 80)
start_opt = datetime.datetime.now()
print('Start time for tuning proceess is ', start_opt)

study.optimize(lambda trial : objective(trial,X_cv,y_cv), n_trials= 200)

end_opt = datetime.datetime.now()
print('End time for the full model fit is ', end_opt)
print('Total tuning time is ', str(end_opt - start_opt))


#Return best score and best parameter set
print('Best trial: score {},\nparams {}'.format(study.best_trial.value,study.best_trial.params))
print('#' * 80)

print("Outputting History CSV", str(datetime.datetime.now()))
hist = study.trials_dataframe()
hist.to_csv(history_file)

print('#' * 80)
print("Param Importance:")
print(optuna.importance.get_param_importances(study))

print('#' * 80)
print("Outputting Param Importance Plot", str(datetime.datetime.now()))
param_imp = visualization.plot_param_importances(study)
param_imp.write_html(param_imp_plot)

print('#' * 80)
print("Outputting History Plot", str(datetime.datetime.now()))
opt_hist_plot = visualization.plot_optimization_history(study)
opt_hist_plot.write_html(hist_plot)

print('#' * 80)
print("Outputting Slice Plot", str(datetime.datetime.now()))
opt_slice_plot = visualization.plot_slice(study)
opt_slice_plot.write_html(slice_plot)

print('#' * 80)
print("Outputting Parallel-Coor Plot", str(datetime.datetime.now()))
opt_para_coor = visualization.plot_parallel_coordinate(study)
opt_para_coor.write_html(para_coor_plot)

end_time = datetime.datetime.now()

print("Tuning Processing finished. Total Tuning time is ", str(end_time - start_time))

print('#' * 80)
print('#' * 80)
print("Now we will use the chosen parameters to examine the learning curves")
print("Final Param Set is:")
final_param = study.best_trial.params
final_param.update({"n_jobs": -1, "verbosity": 0, "n_estimators" : 1000, "objective" : "binary:logistic", "booster": "gbtree", "use_label_encoder": False})
print(final_param)


start_fit = datetime.datetime.now()
print("Start fit", start_fit)
clf = XGBClassifier(use_label_encoder=False)
clf.set_params(**final_param)
clf.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], eval_metric = 'aucpr', verbose = False)
print(clf.get_params())
end_fit = datetime.datetime.now()
print("End fit", end_fit)
print("Fitting time: ", str(end_fit - start_fit))

LC_results_b = clf.evals_result()
LC_train_aucpr_b = LC_results_b['validation_0']['aucpr']
LC_test_aucpr_b = LC_results_b['validation_1']['aucpr']

learning_curve_b = plt.figure(figsize = (6.5 ,4))
plt.style.use('seaborn')
plt.ylim(0,1)
plt.plot(LC_train_aucpr_b, label = 'Train')
plt.plot(LC_test_aucpr_b, label = "Test")
#plt.axvline(x=Final_ES_n_est, ymin=0, ymax=1, color = 'red', linestyle = 'dotted', label = 'ES = ' + str(Final_ES_n_est))
plt.legend()
plt.xlabel('Estimators')
plt.ylabel('AUCPR Score')
plt.title('Learning Curve after feature selection')
learning_curve_b.savefig(lc_b, dpi=300)

print("Done plotting learning curve")

print('#' * 80)
start_val_score = datetime.datetime.now()
print('Start time for val is ', start_val_score)

clf = XGBClassifier(use_label_encoder=False)
clf.set_params(**final_param)
print(clf.get_params())
clf.fit(X_train, y_train)

y_pred_proba_valid = clf.predict_proba(X_valid)[:,1]
y_pred_valid = clf.predict(X_valid)
final_auc = round(sklearn.metrics.roc_auc_score(y_valid, y_pred_proba_valid), 3)
final_ap = round(sklearn.metrics.average_precision_score(y_valid, y_pred_proba_valid), 3)
final_logloss = round(sklearn.metrics.log_loss(y_valid, y_pred_proba_valid), 3)

print('Final validation scores (auc, ap, logloss):')
print(final_auc, final_ap, final_logloss)
end_val_score = datetime.datetime.now()
process_val_score = str(end_val_score - start_val_score)
print('Process time for the val_score is ', process_val_score)
print('#' * 80)

print('#' * 80)
print('Classification report for validation data:')
print(sklearn.metrics.classification_report(y_valid, y_pred_valid))
print('#' * 80)

print('#' * 80)
print('Confusion matrix for validation data:')
print(sklearn.metrics.confusion_matrix(y_valid, y_pred_valid))
print('#' * 80)

print('#' * 80)
print('Confusion matrix for validation data (Normalized by true label):')
print(sklearn.metrics.confusion_matrix(y_valid, y_pred_valid, normalize = 'true'))
print('#' * 80)


print('#' * 80)
print("Outputting PRAUC Plot", str(datetime.datetime.now()))
precision, recall, thresholds = precision_recall_curve(y_valid, y_pred_proba_valid)
opt_prauc_plot = px.area(
    x=recall, y=precision,
    title=f'Precision-Recall Curve (AUC={auc(recall, precision):.3f})',
    labels=dict(x='Recall', y='Precision'),
    width=700, height=500
)
opt_prauc_plot.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=1, y1=0
)
opt_prauc_plot.update_yaxes(scaleanchor="x", scaleratio=1)
opt_prauc_plot.update_xaxes(constrain='domain')
opt_prauc_plot.write_html(prauc_plot)

df = pd.DataFrame({
    'Precision': precision[:len(thresholds)],
    'Recall': recall[:len(thresholds)]
}, index=thresholds)
df.index.name = "Thresholds"
df.columns.name = "Rate"

opt_pr_thres = px.line(
    df, title='Precision and Recall at every threshold',
    width=700, height=500
)

opt_pr_thres.update_yaxes(scaleanchor="x", scaleratio=1)
opt_pr_thres.update_xaxes(range=[0, 1], constrain='domain')
opt_pr_thres.write_html(pr_thre_plot)

# =============================================================================
# Close logfile
# =============================================================================
sys.stdout = original_stdout
log_file.close()
