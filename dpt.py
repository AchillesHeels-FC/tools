# coding=utf-8
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from collections import defaultdict
from configparser import ConfigParser

warnings.filterwarnings("ignore")


class DataProcess(object):
    def __init__(self, conf_file):
        self.conf_file = conf_file
        self.cfg = ConfigParser()
        self.cfg.read(self.conf_file)
        self.black_path = self.cfg.get('sample_path', 'black_file_path')
        self.white_path = self.cfg.get('sample_path', 'white_file_path')
        self.black_data = pd.read_csv(self.black_path, sep=self.cfg.get('sep_string', 'sep'))
        self.white_data = pd.read_csv(self.white_path, sep=self.cfg.get('sep_string', 'sep'))

    def distinct(self, dataframe):
        """
        去除重复值
        """
        df = dataframe.copy()
        df.drop_duplicates(inplace=True)  # 去除重复值
        df.index = range(df.shape[0])  # 去重后恢复索引
        return df

    def fill_nan(self, dataframe):
        """
        填充缺失值
        """
        df = dataframe.copy()
        feature_count = df.shape[1]
        thresh = int(float(feature_count) / 3)
        df.dropna(axis=0, thresh=thresh, inplace=True)
        scatter_features = self.cfg.get('features', 'scatter_features')
        continuous_features = self.cfg.get('features', 'continuous_features')
        fill_dict = defaultdict(float)
        if scatter_features:
            scatter_features = scatter_features.strip().split(',')
            for name in scatter_features:
                mode = df[name].mode()[0]
                fill_dict[name] = mode
        if continuous_features:
            continuous_features = continuous_features.strip().split(',')
            for name in continuous_features:
                mean = df[name].mean()
                fill_dict[name] = mean
        df.fillna(value=fill_dict, inplace=True)
        df.index = range(df.shape[0])
        return df

    def feature_encoder(self, dataframe):
        """
        离散特征编码.
        """
        df = dataframe.copy()
        onehot_features = self.cfg.get('features', 'onehot_features')
        label_encoding_feature = self.cfg.get('features', 'label_encoding_feature')
        if onehot_features:
            onehot_features = onehot_features.strip().split(',')
            for name in onehot_features:
                df = df.join(pd.get_dummies(df[name]))
                df.drop([name], axis=1, inplace=True)
        if label_encoding_feature:
            label_encoding_feature = label_encoding_feature.strip().split(',')
            for name in label_encoding_feature:
                feaMap = {elem: index + 1 for index, elem in enumerate(set(df[name]))}
                df[name] = df[name].map(feaMap)
        df.index = range(df.shape[0])
        return df

    def outlier_process(self, dataframe, classes):
        """
        异常值处理函数
        """
        df = dataframe.copy()
        outlier_func = self.cfg.get('outlier', 'function')
        continuous_features = self.cfg.get('features', 'continuous_features').strip().split(',')
        outlier_list = []
        if outlier_func not in ('quantile', '3sigma'):
            print("Please select function and save to config: quantile or 3sigma")
            return df
        else:
            if outlier_func == 'quantile':
                for col in continuous_features:
                    iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
                    u_th = df[col].quantile(0.75) + 1.5 * iqr  # 上界
                    l_th = df[col].quantile(0.25) - 1.5 * iqr  # 下界
                    outlier_list.append([u_th, l_th])
            elif outlier_func == '3sigma':
                for col in continuous_features:
                    mean = df[col].mean()
                    std = df[col].std()  # 无偏样本方差
                    u_th = mean + 3 * std
                    l_th = mean - 3 * std
                    outlier_list.append([u_th, l_th])
            else:
                pass
            outlier_df = pd.DataFrame(outlier_list, columns=['up_th', 'low_th'], index=continuous_features)
            print('%s features have abnormal values and up && low limit table follow:' % (classes))
            print(outlier_df)
            deal_bool = input("是否处理异常值(y/n): ")
            print('\n')
            if deal_bool in ('y', 'yes'):
                df_ = df.copy()
                for row_name, row in df_.iterrows():
                    for fea_name, fea_info in outlier_df.iterrows():
                        if df_.loc[row_name, fea_name] < outlier_df.loc[fea_name, 'low_th'] \
                                or df_.loc[row_name, fea_name] > outlier_df.loc[fea_name, 'up_th']:
                            df_ = df_.drop(row_name)
                            break
                        else:
                            continue
                df_.index = range(df_.shape[0])
                return df_
            else:
                df.index = range(df.shape[0])
                return df

    def no_dimension_process(self, dataframe):
        """
        无纲量化函数
        """
        df = dataframe.copy()
        nor_func = self.cfg.get('no_dimension', 'normalization')
        continuous_features = self.cfg.get('features', 'continuous_features').strip().split(',')
        for fea in continuous_features:
            if nor_func == 'm':
                df[fea] = (df[fea] - df[fea].min()) / (df[fea].max() - df[fea].min())
            elif nor_func == 'z':
                df[fea] = (df[fea] - df[fea].mean()) / (df[fea].std())
        return df

    def distribute(self, black_data, white_data):
        """
        分布占比及可视化
        """

        def dist_ratio(dataframe, feature, bins, classes):
            df = dataframe.copy()
            q = None
            if isinstance(bins, int):
                q = pd.qcut(df[feature], q=bins)
            elif isinstance(bins, list):
                feature_arr = np.array(df[feature], dtype=float)
                feature_arr = np.squeeze(feature_arr)
                q = pd.cut(feature_arr, bins)
            q_count = q.value_counts()
            q_df = q_count.to_frame(name='频数')
            q_df['频率'] = q_df / q_df['频数'].sum()
            q_df['频率%'] = q_df['频率'].map(lambda x: '%.2f%%' % (x * 100))
            q_df = q_df.drop(['频率'], axis=1)
            print('%s %s dist ratio table following: ' % (feature, classes))
            print(q_df)

        black_df = black_data.copy()
        white_df = white_data.copy()
        target_features = self.cfg.get('distribute', 'target_features').strip().split(',')
        features_count = len(target_features)
        plt.figure('Feature distribute', figsize=(30, 15))
        for index, fea in enumerate(target_features):
            try:
                bin_type = self.cfg.get('distribute', fea + '_bins')
            except Exception as e:
                bin_type = ''
            min_w = white_df[fea].min()
            max_w = white_df[fea].max()
            min_b = black_df[fea].min()
            max_b = black_df[fea].max()
            min_val = min(min_w, min_b)
            max_val = max(max_w, max_b)
            if bin_type.isdigit():
                q_bins = int(self.cfg.get('distribute', fea + '_bins'))
                diff = int((max_val - min_val) / float(q_bins))
                bins = 0 if max_val == min_val or diff == 0 \
                    else [i for i in range(int(min_val), int(max_val + 1), diff)]
            else:
                if bin_type:
                    bins = self.cfg.get('distribute', fea + '_bins').strip().split(',')
                    bins.insert(0, min_val)
                    bins.append(max_val)
                    bins = list(map(lambda x: float(x), bins))
                    bins.sort()
                else:
                    q_bins = 5
                    diff = int((max_val - min_val) / float(q_bins))
                    bins = 0 if max_val == min_val or diff == 0 \
                        else [i for i in range(int(min_val), int(max_val + 1), diff)]
            print('=========================================')
            if bins:
                dist_ratio(white_df, fea, bins, 'white')
                print('\n')
                dist_ratio(black_df, fea, bins, 'black')
                plt.subplot(4, features_count / 4 + 1, index + 1)
                plt.subplots_adjust(wspace=0.2, hspace=0.25)
                sns.distplot(white_data[fea], bins, kde=True, color='b')
                sns.distplot(black_data[fea], bins, kde=True, color='r')
            else:
                print("Feature %s not have valid bins" % (fea))
            print('=========================================')
        plt.show()

    def statis_information(self, dataframe, classes):
        df = dataframe.copy()
        continuous_features = self.cfg.get('features', 'continuous_features').strip().split(',')
        statis_info = []
        for index, fea in enumerate(continuous_features):
            min_val = df[fea].min()
            max_val = df[fea].max()
            mean = df[fea].mean()
            var = df[fea].var()
            std = df[fea].std()
            median = df[fea].median()
            statis_info.append([fea, min_val, max_val, mean, var, std, median])
        columns = ['feature', 'min', 'max', 'mean', 'var', 'std', 'median']
        statis_df = pd.DataFrame(statis_info, columns=columns)
        statis_df.set_index(['feature'], inplace=True)
        print('============================================================================\n')
        print('%s data statis information table follow:' % classes)
        print(statis_df)
        print('\n============================================================================\n')

    def compare_feature(self, white_data, black_data):
        white_df = white_data.copy()
        black_df = black_data.copy()
        white_df.loc[:, 'label'] = 0
        black_df.loc[:, 'label'] = 1
        df = pd.concat([white_df, black_df])
        continuous_features = self.cfg.get('features', 'continuous_features').strip().split(',')
        features_count = len(continuous_features)
        plt.figure('Feature distribute', figsize=(30, 15))
        for index, fea in enumerate(continuous_features):
            plt.subplot(4, features_count / 4 + 1, index + 1)
            plt.subplots_adjust(wspace=0.2, hspace=0.25)
            sns.barplot(x='label', y=fea, data=df, hue='label')
        plt.show()

    def feature_importance(self, white_data, black_data):
        white_df = white_data.copy()
        black_df = black_data.copy()
        dataframe = pd.concat([white_df, black_df])
        columns = dataframe.columns.tolist()
        x = dataframe.values
        y = [[0]] * white_df.shape[0] + [[1]] * black_df.shape[0]
        model = XGBClassifier(n_estimators=100, learning_rate=0.2)
        model.fit(x, y)
        feature_import_df = pd.DataFrame([model.feature_importances_], columns=columns, index=None)
        print('Feature importance table follow:')
        print(feature_import_df)
        print('\n')


def pre_process(dataset, process, classes):
    df = dataset.copy()
    # df = process.distinct(df)
    df = process.fill_nan(df)
    df = process.feature_encoder(df)
    df = process.outlier_process(df, classes)
    df_n = process.no_dimension_process(df)
    return df, df_n


def main():
    dp = DataProcess(r'./dpt.conf')
    black_data = dp.black_data
    white_data = dp.white_data
    black_df, black_df_n = pre_process(black_data, dp, 'Black')
    white_df, white_df_n = pre_process(white_data, dp, 'White')
    dp.distribute(black_df, white_df)
    dp.statis_information(white_data, 'White')
    dp.statis_information(black_data, 'Black')
    dp.compare_feature(white_df, black_df)
    dp.feature_importance(white_df, black_df)


if __name__ == '__main__':
    Begin_title = """
 ______   ________   _________  ________       ______   ______    ______   ______   ______   ______   ______      
/_____/\ /_______/\ /________/\/_______/\     /_____/\ /_____/\  /_____/\ /_____/\ /_____/\ /_____/\ /_____/\     
\:::_ \ \\::: _  \ \\__.::.__\/\::: _  \ \    \:::_ \ \\:::_ \ \ \:::_ \ \\:::__\/ \::::_\/_\::::_\/_\::::_\/_    
 \:\ \ \ \\::(_)  \ \  \::\ \   \::(_)  \ \    \:(_) \ \\:(_) ) )_\:\ \ \ \\:\ \  __\:\/___/\\:\/___/\\:\/___/\   
  \:\ \ \ \\:: __  \ \  \::\ \   \:: __  \ \    \: ___\/ \: __ `\ \\:\ \ \ \\:\ \/_/\\::___\/_\_::._\:\\_::._\:\  
   \:\/.:| |\:.\ \  \ \  \::\ \   \:.\ \  \ \    \ \ \    \ \ `\ \ \\:\_\ \ \\:\_\ \ \\:\____/\ /____\:\ /____\:\ 
    \____/_/ \__\/\__\/   \__\/    \__\/\__\/     \_\/     \_\/ \_\/ \_____\/ \_____\/ \_____\/ \_____\/ \_____\/ 
"""
    Begin_pic = """
                           /| 
                         //    /  ||  || 
                 /|  // // .·´   /|\/ /  //    /| 
            |\  / ||/ / /|    /|/  |/  |/ /¸,./  / 
            |  /  /   |    /   | /   /   /    / 
         \`·.\ \  |  | \  | | |||´   /  /| /  /.·´/ 
           `·¸,.-·´¯¯¯\´_/_\\__/  /   /,./, 
            / .-.-==-./     ¯¯  / /  //.´   / 
           |/      \=,_     _.=- \./¯\_.·´ 
                     \-´`, ´`--´   ,'µ / 
                       `·.`--.  ,.·    /  \ 
           ,   _¸,.-.__ ,¯¯  /    /-../~__,.-, 
           |`·´  /    /¯¯`·.´¯¯¯¯ //// /       /'`·. 
            \,  \    \,  /      .·´ //     |    \´       \ 
            /\ -._     `·.¸.-´¨¯      /    \  (   --.   ./ 
            `/._  \   `· \-·´    //            /.    |-   | 
             `,  '   \      '     /          |   \     |    | 
             `/   ._  \     _.·         .·´    |     /   ´. 
              ,\___`·._ ·´____/.____.·´¨ \.-·     `, 
            /    {==============|   /___    | 
       ,-,/=\   \¯==¯¯=¯¯==¯==¯=/  /===-- / 
      /`/`/`/`/=||¯| ¯¯¯¯| ¯¯¯¯¯¯¯¯ \|¨|¨|¯|¯| /=| 
"""
    print(Begin_title)
    print(Begin_pic)
    main()
