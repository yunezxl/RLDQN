from GenerateData import GenerationData
from BootstrapReSample import expand_demand
from SolverTeacher import DRO, DRO_Test
from SolverDQN import DQN
from SolverRLDQN import RLDQN
from TestDQN import load_DQN
from TestRLDQN import load_RLDQN
import numpy as np
import random
import tensorflow as tf

seed_number = 42
np.random.seed(seed_number)
random.seed(seed_number)
tf.set_random_seed(seed_number)

class TrainAndTest():
    def __init__(self, DisType_Demand, DisParams_Demand, TrainSampleSize, CaseParams, Repeat, Round, Time):
        self.DisType = DisType_Demand
        self.DisParams_Demand = DisParams_Demand
        self.TrainSampleSize = TrainSampleSize
        self.CaseParams = CaseParams
        self.Repeat = Repeat
        self.Round = Round
        self.Time = Time
        self.AvgReward_DRO = []
        self.AvgReward_DQN = []
        self.AvgReward_RLDQN = []
        self.avgReward_DRO = []
        self.avgReward_DQN = []
        self.avgReward_RLDQN = []
        self.train_DRO_ord = 0
        self.train_DRO_profit = 0
        self.Order_DRO = []
        self.Order_DQN = []
        self.Order_RLDQN = []
        self.order_DQN = []
        self.order_RLDQN = []
        self.Repeat_DRO = []
        self.Repeat_DQN = []
        self.Repeat_RLDQN = []   #


    def Train(self):
        TrainSet, _ = GenerationData(self.DisType, self.DisParams_Demand, self.TrainSampleSize)
        DRO_profit, Order_DRO = DRO(self.CaseParams, TrainSet)
        self.train_DRO_ord = Order_DRO
        self.Order_DRO.append(Order_DRO)
        self.train_DRO_profit = DRO_profit

        ExpandDataSet = expand_demand(TrainSet, ExpandSize=10000)

        DQN_Agent = DQN(CaseParams=self.CaseParams, Data=TrainSet, Demand_Train=ExpandDataSet,
                        DQNHyparams=[11, 300, 0.90, 0.98, 0.01, 0.002, 64, 128],
                        DisType=self.DisType)
        DQN_Agent.train()

        RLDQN_Agent = RLDQN(CaseParams=self.CaseParams, Data=TrainSet, Demand_Train=ExpandDataSet,
                                    RLDQNHyparams=[13, 300, 0.90, 0.98, 0.01, 0.002, 64, 128, 20, Order_DRO,
                                                   DRO_profit], DisType=self.DisType)
        RLDQN_Agent.train()


    def CleanStore(self, inner = True):
        if inner:
            self.avgReward_DRO = []
            self.avgReward_DQN = []
            self.avgReward_RLDQN = []
            self.order_DQN = []
            self.order_RLDQN = []
        else:
            self.AvgReward_DRO = []
            self.AvgReward_DQN = []
            self.AvgReward_RLDQN = []
            self.Order_DRO = []

    def Test(self):
        _, TestSet = GenerationData(self.DisType, self.DisParams_Demand, self.TrainSampleSize)

        tempReward_DRO = DRO_Test(CaseParams=self.CaseParams, Order=self.train_DRO_ord, TestDemand=TestSet)

        DQN_Agent = load_DQN(CaseParams=self.CaseParams, TestData=TestSet, TrainSampleSize = self.TrainSampleSize,
                             DQNHyparams=[11, 300, 0.90, 0.98, 0.01, 0.002, 64, 128], DisType=self.DisType)
        DQN_AvgReward, DQN_order = DQN_Agent.Test()
        self.order_DQN.append(DQN_order)


        RLDQN_Agent_DRO = load_RLDQN(CaseParams=self.CaseParams, TestData=TestSet, TrainSampleSize = self.TrainSampleSize,
                                         RLDQNHyparams=[13, 300, 0.90, 0.98, 0.01, 0.002, 64, 128, 10, self.train_DRO_ord, tempReward_DRO],
                                         DisType=self.DisType)
        result_RLDQN, RLDQN_Order = RLDQN_Agent_DRO.Test()
        self.order_RLDQN.append(RLDQN_Order)

        return [tempReward_DRO, DQN_AvgReward, result_RLDQN]

    def StoreProfit(self, input, inner = True):
        if inner == True:
            self.avgReward_DRO.append(input[0])
            self.avgReward_DQN.append(input[1])
            self.avgReward_RLDQN.append(input[2])
        else:
            self.AvgReward_DRO.append(input[0])
            self.AvgReward_DQN.append(input[1])
            self.AvgReward_RLDQN.append(input[2])

    def StoreOrder(self):
        Mean_Order_DQN = np.mean(self.order_DQN)
        Mean_Order_RLDQN = np.mean(self.order_RLDQN)
        self.Order_DQN.append(Mean_Order_DQN)
        self.Order_RLDQN.append(Mean_Order_RLDQN)

    def AvgPermance(self):
        m1 = np.mean(self.avgReward_DRO)
        m2 = np.mean(self.avgReward_DQN)
        m3 = np.mean(self.avgReward_RLDQN)
        return [m1, m2, m3]

    def AvgOrder(self):
        o1 = np.mean(self.order_DQN)
        o2 = np.mean(self.order_RLDQN)
        return [o1, o2]

    def Evaluation(self):
        Temp_DQN = (np.array(self.AvgReward_DQN) / np.array(self.AvgReward_DRO) - 1) * 100
        Mean_DQN_Profit = np.round(np.mean(Temp_DQN), 2)
        ci_DQN_Profit = round(1.96 * np.std(Temp_DQN) / np.sqrt(len(Temp_DQN)), 2)

        Temp_RLDQN = (np.array(self.AvgReward_RLDQN) / np.array(self.AvgReward_DRO) - 1) * 100
        Mean_RLDQN_Profit = np.round(np.mean(Temp_RLDQN), 2)
        ci_RLDQN_Profit = round(1.96 * np.std(Temp_RLDQN) / np.sqrt(len(Temp_RLDQN)), 2)

        Mean_Order_DRO = np.round(np.mean(self.Order_DRO), 2)
        ci_order_DRO = round(1.96 * np.std(self.Order_DRO) / np.sqrt(len(self.Order_DRO)), 2)

        Mean_Order_DQN = np.round(np.mean(self.Order_DQN), 2)
        ci_order_DQN = round(1.96 * np.std(self.Order_DQN) / np.sqrt(len(self.Order_DQN)), 2)

        Mean_Order_RLDQN = np.round(np.mean(self.Order_RLDQN), 2)
        ci_order_RLDQN = round(1.96 * np.std(self.Order_RLDQN) / np.sqrt(len(self.Order_RLDQN)), 2)


        self.Repeat_DRO.append([round(Mean_Order_DRO), ci_order_DRO, np.round(np.mean(self.AvgReward_DRO), 2)])
        self.Repeat_DQN.append([round(Mean_Order_DQN), ci_order_DQN, np.round(np.mean(self.AvgReward_DQN), 2), Mean_DQN_Profit, ci_DQN_Profit])
        self.Repeat_RLDQN.append([round(Mean_Order_RLDQN), ci_order_RLDQN, np.round(np.mean(self.AvgReward_RLDQN), 2), Mean_RLDQN_Profit, ci_RLDQN_Profit])


    def Rep_Print(self):
        for i in range(self.Repeat):
            print(' ')
            print('**********************************')
            print('Mean Profit Gap of DQN | ', self.Repeat_DQN[i][3], "±", self.Repeat_DQN[i][4])
            print('Mean Profit Gap of RLDQN | ', self.Repeat_RLDQN[i][3], "±",
                  self.Repeat_RLDQN[i][4])
            print(' ')


    def Todo(self):
        for _ in range(self.Repeat):
            self.CleanStore(inner=False)
            for r in range(self.Round):
                self.Train()
                self.CleanStore(inner=True)
                for t in range(self.Time):
                    TestResult = self.Test()
                    self.StoreProfit(TestResult, inner=True)
                AvgTProfit = self.AvgPermance()
                AvgTOrder = self.AvgOrder()
                self.StoreProfit(AvgTProfit, inner=False)
                self.StoreOrder()
            self.Evaluation()
        self.Rep_Print()



# ——————————————————————————————————————————————————————————————————————————————————————————————— #
A_Case = TrainAndTest("SyntData",
                      [[[100, 140], [[25 ** 2, 0.5*25*35], [0.5*25*35, 35 ** 2]]],
                       [100, 140],
                       [[0, 0], [200, 280]]],
                      40, [40, 60, 4, 4, 6, 6, 20],
                      Repeat = 1, Round = 100, Time = 1)



# A_Case = TrainAndTest("RealData", None, 20, [40, 60, 4, 4, 6, 6, 20],
#                       Repeat = 1, Round = 100, Time = 1)

A_Case.Todo()


