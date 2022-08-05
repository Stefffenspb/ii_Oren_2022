#Библиотеки
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import sklearn.metrics


#Подготовим данные


def data_prepare(df):
    #df = df.drop(columns='id')
    df = df.drop(columns='spent_time_to_complete_hw')

    #df['month_id'] =df['month_id'].astype(str).str[-4:]
    #df['month_id'].astype(str).str[0:2]

    ds = df['month_id'].astype(str).str[-4:]
    dm = df['month_id'].astype(str).str[0:2]
    dm = dm.str.extract('(\w+)', expand=False)
    if len(dm) < 2:
        dm = '0' + dm
    df['month_id'] = ds + dm
    dsc = df['carts_created_at'].astype(str).str[-4:]
    dmc = df['carts_created_at'].astype(str).str[0:2]
    dmc = dmc.str.extract('(\w+)', expand=False)
    if len(dmc) < 2:
        dmc = '0' + dmc
    df['carts_created_at'] = dsc + dmc
    df['time'] = df['month_id'].astype(int) - df['carts_created_at'].astype(int)
    df['completed_hw']=df['completed_hw'] - df['reworked_hw']
    df['failed_hw'] = df['failed_hw'] - df['reworked_hw']
    df = df.drop(columns='reworked_hw')
    df = df.drop(columns='hw_leader')
    df = df.drop(columns='month_id')
    df = df.drop(columns='carts_created_at')
    #df['carts_created_at'] = df['carts_created_at'].astype(str).str[-4:]
    df = df.replace({'promo': {'-': '0', '+': '1'}})
    df = df.replace({'communication_type': {'phone': '1', 'web': '2', 'order': '3'}})
    df = df.replace({'ABC': {'A': '1', 'B': '2', 'C': '3', 'D': '4'}})
    df = df.drop(columns='failed_hw')
    df['city'] = df['city'].fillna('0')
    df['country'] = df['country'].fillna('0')
    df['city'].where(~(df.city != '0'), other=1, inplace=True)
    df['country'].where(~(df.country != '0'), other=1, inplace=True)
    #df = df.drop(columns='city')
    #df = df.drop(columns='speed_recall')
    df = df.drop(columns='p_missed_calls')
    #df = df.drop(columns='p_total_calls')

    df = df.drop(columns='bought_d1')
    df = df.drop(columns='bought_d2')
    df = df.drop(columns='bought_d3')
    df = df.drop(columns='bought_d4')
    df = df.drop(columns='bought_d5')
    #df = df.drop(columns='student_id')
    df = df.drop(columns='p_was_conversations')
    #df = df.drop(columns='country')
    df = df.replace({'os': {'Android': '1', 'iOS': '2', 'Linux': '3', 'Mac OS X': '4', 'Ubuntu': '5', 'Windows': '6', 'Fedora': '7', 'Chrome OS': '8'}})
    df = df.drop(columns='browser')
    df = df.replace({'platform': {'mobile': '1', 'pc': '2', 'tablet': '3'}})

    df = df.fillna('0')
    df = df.astype('float64')
    print(df.skew())
    return df



#Обучение и Прогнозирование
def science(col, preds, filecsv_from, filecsv_to,colim):

    df_train = df
    for diag in colim:

        df_train.drop(columns=diag)

    X = df_train.drop(col, axis=1)
    y = df_train[col]
    test_size = 0.2
    seed = 42
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=test_size, random_state=seed)
    num_folds = 10
    n_estimators = 100
    scoring = 'precision'
    models = []
    #Список Моделей
 ####################
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    ss = StandardScaler()
    X_train_scaled = ss.fit_transform(X_train)
    X_test_scaled = ss.transform(X_validation)
    y_train = np.array(Y_train)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import recall_score
    rfc = RandomForestClassifier()
    rfc.fit(X_train_scaled, y_train)
    rfc.score(X_train_scaled, y_train)
    feats = {}
    for feature, importance in zip(df.columns, rfc.feature_importances_):
        feats[feature] = importance
    importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-Importance'})
    importances = importances.sort_values(by='Gini-Importance', ascending=False)
    importances = importances.reset_index()
    importances = importances.rename(columns={'index': 'Features'})
    sns.set(font_scale=5)
    sns.set(style="whitegrid", color_codes=True, font_scale=1.7)
    fig, ax = plt.subplots()
    fig.set_size_inches(30, 15)
    sns.barplot(x=importances['Gini-Importance'], y=importances['Features'], data=importances, color='skyblue')
    plt.xlabel('Importance', fontsize=25, weight='bold')
    plt.ylabel('Features', fontsize=25, weight='bold')
    plt.title('Feature Importance', fontsize=25, weight='bold')
    plt.show()
    print(importances)
    ###


    '''
    pca_test = PCA(n_components=52)
    pca_test.fit(X_train_scaled)
    sns.set(style='whitegrid')
    plt.plot(np.cumsum(pca_test.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.axvline(linewidth=4, color='r', linestyle='--', x=52, ymin=0, ymax=1)
    plt.show()
    evr = pca_test.explained_variance_ratio_
    cvr = np.cumsum(pca_test.explained_variance_ratio_)
    pca_df = pd.DataFrame()
    pca_df['Cumulative Variance Ratio'] = cvr
    pca_df['Explained Variance Ratio'] = evr
    pca_df.head(52)
    '''
##########


    #models.append(('LR', LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=300, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1, l1_ratio=None)))
    #models.append(('LDA', LinearDiscriminantAnalysis()))
    #models.append(('KNN', KNeighborsClassifier()))
    models.append(('RF',RandomForestClassifier(random_state=42)))
    #models.append(('CART', DecisionTreeClassifier(random_state=42)))
    #models.append(('NB', GaussianNB()))
    #models.append(('SVM', SVC(gamma='auto')))




    # оцениваем модель на каждой итерации
    scores = []
    names = []
    results = []
    predictions = []
    msg_row = []
    best = []
    for name, model in models:
        modi = model.fit(X_train_scaled, y_train)

        predi = modi.predict(X_test_scaled)
        #predi = pd.DataFrame({col: predi})
        recall = sklearn.metrics.recall_score(Y_validation, predi,average='macro')
        presc = sklearn.metrics.precision_score(Y_validation, predi,average='macro')
        kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold)
        results.append(cv_results)
        names.append(name)
        # Оценка Моделей
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
        print(0.2*recall + 0.8 * presc)
        best.append(0.2*recall + 0.8 * presc)
    index = best.index(max(best))


    #Выберем Лучшую
    best_train = models[index][1]

    df_test = pd.read_csv('data/test_dataset_test.csv')
    df_test.head()
    df_test_2 = pd.read_csv(filecsv_to)
    df_test_2.head()

    df_test_2 = df_test_2.rename(columns={'id': 'id_y'})
    df_merged = pd.merge(df_test, df_test_2, left_on='id', right_on='id_y', how='outer')
    df_merged = data_prepare(df_merged)
    df_merged = df_merged.drop(columns='id_y')
    df_merged_id = pd.read_csv(filecsv_to)
    df_merged_id.head()

    df_merged = df_merged.drop(columns=col)

    #df_merged_d = df_merged.drop(columns='id')
    #for diag in colim:
    #    df_merged_d.drop(columns=diag)
    gnb = best_train

    model = gnb.fit(X,  y)

    preds = model.predict(df_merged)
    preds = pd.DataFrame({col: preds})
    #preds = pd.DataFrame({col: preds.tolist()})
    df_merged_id = df_merged_id.drop(columns=col)
    df_merged_id = pd.concat([df_merged_id, preds], axis=1)

    df_merged_id = df_merged_id[['id', 'target']]
    df_merged_id['target']= df_merged_id['target'].astype('int64')
    df_merged_id.to_csv(filecsv_to, index=False)
    #print(df_merged_id)
    #Значения предсказания

    return preds


df = pd.read_csv('data/train_dataset_train.csv')
df.head()
df = df.dropna(subset=['target'])
df = data_prepare(df)
#df['city'].to_excel('train_dataset_train.xlsx')
#df = df.drop(columns='ID')
#Соотношение Столбцов

#print(df.info())



#Построим Матрицу
# scatter_matrix(df)
# plt.show()
preds = []

filecsv_from = 'data/sample_solution.csv'
df_result = pd.read_csv(filecsv_from)
df_result.head()
df_merged_id = df_result['id']
filecsv_to = 'result/sample_solution.csv'
df_result.to_csv(filecsv_to, index=False)

column_list = pd.Series(['target'])
colim = column_list
colim.index=column_list
for col in column_list:
    colim = colim.drop(col)

    science(col, preds, filecsv_from, filecsv_to,colim)
#for col in column_list:
#    science(col, preds, filecsv_from, filecsv_to)
