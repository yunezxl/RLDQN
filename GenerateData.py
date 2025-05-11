import numpy as np
import os
import pickle
import scipy.stats as stats
from scipy.stats import norm, expon
import pandas as pd
import random

def GenerationData(DisType_Demand, DisParams_Demand, TrainSampleSize):
    warmup = 0
    TestSampleSize = 60 + warmup
    rho = 0.5

    if DisType_Demand == "RealData":
        TrainSamples = []
        TestSamples = []

        # df = pd.read_excel('JD.xlsx')
        # for i in range(2):
        #     if i == 1:
        #         column_name = '2ddb64e05a'
        #         selected_columns = df[column_name]
        #         split_index = int(len(selected_columns) * 20 / 31)
        #         item_train_data = df.iloc[:split_index]
        #         item_test_data = df.iloc[split_index:]
        #         item_train_data = item_train_data['2ddb64e05a'].tolist()
        #         item_test_data = item_test_data['2ddb64e05a'].tolist()
        #     if i == 0:
        #         column_name = 'cdee05b50c'
        #         selected_columns = df[column_name]
        #         split_index = int(len(selected_columns) * 20 / 31)
        #         item_train_data = df.iloc[:split_index]
        #         item_test_data = df.iloc[split_index:]
        #         item_train_data = item_train_data['cdee05b50c'].tolist()
        #         item_test_data = item_test_data['cdee05b50c'].tolist()
        #     TrainSamples.append(item_train_data)
        #     TestSamples.append(item_test_data)

        df = pd.read_csv(open('Kaggle 2018.csv'))
        for i in range(2):
            if i == 0:
                item_data = df[df['item'] == 26]
                train_data = []
                test_data = []
                for index, row in item_data.iterrows():
                    if len(train_data) < int(len(item_data) * 0.75):
                        train_data.append(row)
                    else:
                        test_data.append(row)
                train_data = pd.DataFrame(train_data)
                test_data = pd.DataFrame(test_data)
                train_data = train_data['sales'].tolist()
                test_data = test_data['sales'].tolist()
            if i == 1:
                item_data = df[df['item'] == 40]
                train_data = []
                test_data = []
                for index, row in item_data.iterrows():
                    if len(train_data) < int(len(item_data) * 0.75):
                        train_data.append(row)
                    else:
                        test_data.append(row)
                train_data = pd.DataFrame(train_data)
                test_data = pd.DataFrame(test_data)
                train_data = train_data['sales'].tolist()
                test_data = test_data['sales'].tolist()
            TrainSamples.append(train_data)
            TestSamples.append(test_data)


    elif DisType_Demand == "SyntData":
        TrainSamples = [[None for _ in range(TrainSampleSize)] for _ in range(2)]
        for i in range(TrainSampleSize):
            distribution = np.random.choice([0,1,2])
            if distribution == 0:
                temp_sample = np.random.multivariate_normal(DisParams_Demand[0][0], DisParams_Demand[0][1], 1)
                temp_sample = np.clip(temp_sample, 0, 300)
                TrainSamples[0][i] = int(temp_sample[0][0])
                TrainSamples[1][i] = int(temp_sample[0][1])
            elif distribution == 1:
                correlation_matrix = np.array([[1, rho], [rho, 1]])
                L = np.linalg.cholesky(correlation_matrix)
                standard_uniform_samples = np.random.uniform(size=(1, 2))
                standard_normal_samples = norm.ppf(standard_uniform_samples)
                correlated_normal_samples = np.dot(standard_normal_samples, L)
                exponential_samples_1 = expon.ppf(norm.cdf(correlated_normal_samples[:, 0]), scale=DisParams_Demand[1][0])
                exponential_samples_2 = expon.ppf(norm.cdf(correlated_normal_samples[:, 1]), scale=DisParams_Demand[1][1])
                TrainSamples[0][i] = int(np.clip(exponential_samples_1, 0, 300))
                TrainSamples[1][i] = int(np.clip(exponential_samples_2, 0, 300))
            else:
                correlation_matrix = np.array([[1, rho], [rho, 1]])
                L = np.linalg.cholesky(correlation_matrix)
                standard_uniform_samples = np.random.uniform(size=(1, 2))
                correlated_samples = np.dot(standard_uniform_samples, L)
                scaled_samples = np.array([
                    correlated_samples[:, 0] * (200 / np.sqrt(3)) + 100,
                    correlated_samples[:, 1] * (280 / np.sqrt(3)) + 140
                ])
                TrainSamples[0][i] = int(scaled_samples[0])
                TrainSamples[1][i] = int(scaled_samples[1])

        TestSamples = [[None for _ in range(TestSampleSize)] for _ in range(2)]
        for i in range(TestSampleSize):
            distribution = np.random.choice([0, 1, 2])
            if distribution == 0:
                temp_sample = np.random.multivariate_normal(DisParams_Demand[0][0],
                                                            DisParams_Demand[0][1],
                                                            1)
                temp_sample = np.clip(temp_sample, 0, 300)
                TestSamples[0][i] = int(temp_sample[0][0])
                TestSamples[1][i] = int(temp_sample[0][1])
            elif distribution == 1:
                correlation_matrix = np.array([[1, rho], [rho, 1]])
                L = np.linalg.cholesky(correlation_matrix)
                standard_uniform_samples = np.random.uniform(size=(1, 2))
                standard_normal_samples = norm.ppf(standard_uniform_samples)
                correlated_normal_samples = np.dot(standard_normal_samples, L)
                exponential_samples_1 = expon.ppf(norm.cdf(correlated_normal_samples[:, 0]),
                                                  scale=DisParams_Demand[1][0])
                exponential_samples_2 = expon.ppf(norm.cdf(correlated_normal_samples[:, 1]),
                                                  scale=DisParams_Demand[1][1])
                TestSamples[0][i] = int(np.clip(exponential_samples_1, 0, 300))
                TestSamples[1][i] = int(np.clip(exponential_samples_2, 0, 300))
            else:
                correlation_matrix = np.array([[1, rho], [rho, 1]])
                L = np.linalg.cholesky(correlation_matrix)
                standard_uniform_samples = np.random.uniform(size=(1, 2))
                correlated_samples = np.dot(standard_uniform_samples, L)
                scaled_samples = np.array([
                    correlated_samples[:, 0] * (200 / np.sqrt(3)) + 100,
                    correlated_samples[:, 1] * (280 / np.sqrt(3)) + 140
                ])
                TestSamples[0][i] = int(scaled_samples[0])
                TestSamples[1][i] = int(scaled_samples[1])

    else:
        pass

    return TrainSamples, TestSamples

def SaveData(Data, filename, path):
    file_path = os.path.join(path, filename)
    with open(file_path, 'wb') as file:
        pickle.dump(Data, file)

