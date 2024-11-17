import matplotlib.pyplot as plt
import joblib

train_X_column_name = ['blh', 'r', 'sp', 't2m', 'wd', 'ws', 'band1', 'band2', 'dem_no', 'pop', 'ndvi']
machine_path = r'./去除高值加月平均/第一层模型/firest_ET_month_mean_droplow.m'
random_forest_model = joblib.load(machine_path)
random_forest_importance=list(random_forest_model.feature_importances_)
random_forest_feature_importance=[(feature,round(importance,8))
                                  for feature, importance in zip(train_X_column_name,random_forest_importance)]
random_forest_feature_importance=sorted(random_forest_feature_importance,key=lambda x:x[1],reverse=True)
plt.figure(3)
plt.clf()
importance_plot_x_values=list(range(len(random_forest_importance)))
plt.bar(importance_plot_x_values,random_forest_importance,orientation='vertical')
plt.xticks(importance_plot_x_values, train_X_column_name, rotation='vertical')
plt.xlabel('Variable')
plt.ylabel('Importance')
plt.title('Variable Importances')
plt.show()
