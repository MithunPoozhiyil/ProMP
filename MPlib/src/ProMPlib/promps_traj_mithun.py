import matplotlib.pyplot as plt
import numpy as np
import phase as phase
import basis as basis
import json
import load_data

import scipy.stats as stats

class ProMP:

    def __init__(self, basis, phase, numDoF):
        self.basis = basis
        self.phase = phase
        self.numDoF = numDoF
        self.numWeights = basis.numBasis * self.numDoF
        self.mu = np.zeros(self.numWeights)
        self.covMat = np.eye(self.numWeights)
        self.observationSigma = np.ones(self.numDoF)

    def getTrajectorySamples(self, time, n_samples=1):
        phase = self.phase.phase(time)
        basisMultiDoF = self.basis.basisMultiDoF(phase, self.numDoF)
        weights = np.random.multivariate_normal(self.mu, self.covMat, n_samples)
        weights = weights.transpose()
        trajectoryFlat = basisMultiDoF.dot(weights)
        #a = trajectoryFlat
        trajectoryFlat = trajectoryFlat.reshape((self.numDoF,int(trajectoryFlat.shape[0] / self.numDoF), n_samples))
        trajectoryFlat = np.transpose(trajectoryFlat, (1, 0, 2))
        #trajectoryFlat = trajectoryFlat.reshape((a.shape[0] / self.numDoF, self.numDoF, n_samples))

        return trajectoryFlat

    def getMeanAndCovarianceTrajectory(self, time):
        basisMultiDoF = self.basis.basisMultiDoF(time, self.numDoF)
        trajectoryFlat = basisMultiDoF.dot(self.mu.transpose())
        trajectoryMean = trajectoryFlat.reshape((self.numDoF, int(trajectoryFlat.shape[0] / self.numDoF)))
        trajectoryMean = np.transpose(trajectoryMean, (1, 0))
        covarianceTrajectory = np.zeros((self.numDoF, self.numDoF, len(time)))

        for i in range(len(time)):

            basisSingleT = basisMultiDoF[slice(i, (self.numDoF - 1) * len(time) + i + 1, len(time)), :]
            covarianceTimeStep = basisSingleT.dot(self.covMat).dot(basisSingleT.transpose())
            covarianceTrajectory[:, :, i] = covarianceTimeStep

        return trajectoryMean, covarianceTrajectory

    def getMeanAndStdTrajectory(self, time):
        basisMultiDoF = self.basis.basisMultiDoF(time, self.numDoF)
        trajectoryFlat = basisMultiDoF.dot(self.mu.transpose())
        trajectoryMean = trajectoryFlat.reshape((self.numDoF, trajectoryFlat.shape[0] / self.numDoF))
        trajectoryMean = np.transpose(trajectoryMean, (1, 0))
        stdTrajectory = np.zeros((len(time), self.numDoF))

        for i in range(len(time)):

            basisSingleT = basisMultiDoF[slice(i, (self.numDoF - 1) * len(time) + i + 1, len(time)), :]
            covarianceTimeStep = basisSingleT.dot(self.covMat).dot(basisSingleT.transpose())
            stdTrajectory[i, :] = np.sqrt(np.diag(covarianceTimeStep))

        return trajectoryMean, stdTrajectory

    def getMeanAndCovarianceTrajectoryFull(self, time):
        basisMultiDoF = self.basis.basisMultiDoF(time, self.numDoF)

        meanFlat = basisMultiDoF.dot(self.mu.transpose())
        covarianceTrajectory = basisMultiDoF.dot(self.covMat).dot(basisMultiDoF.transpose())

        return meanFlat, covarianceTrajectory

    def jointSpaceConditioning(self, time, desiredTheta, desiredVar):
        newProMP = ProMP(self.basis, self.phase, self.numDoF)
        basisMatrix = self.basis.basisMultiDoF(time, self.numDoF)
        temp = self.covMat.dot(basisMatrix.transpose())
        L = np.linalg.solve(desiredVar + basisMatrix.dot(temp), temp.transpose())
        L = L.transpose()
        newProMP.mu = self.mu + L.dot(desiredTheta - basisMatrix.dot(self.mu))
        newProMP.covMat = self.covMat - L.dot(basisMatrix).dot(self.covMat)
        return newProMP


    def getTrajectoryLogLikelihood(self, time , trajectory):

        trajectoryFlat =  trajectory.transpose().reshape(trajectory.shape[0] * trajectory.shape[1])
        meanFlat, covarianceTrajectory = self.getMeanAndCovarianceTrajectoryFull(self, time)

        return stats.multivariate_normal.logpdf(trajectoryFlat, mean=meanFlat, cov=covarianceTrajectory)

    def getWeightsLogLikelihood(self, weights):

        return stats.multivariate_normal.logpdf(weights, mean=self.mu, cov=self.covMat)

    def plotProMP(self, time, indices = None):
        import plotter as plotter

        trajectoryMean, stdTrajectory = self.getMeanAndStdTrajectory(time)


        plotter.plotMeanAndStd(time, trajectoryMean, stdTrajectory, indices)



class MAPWeightLearner():

    def __init__(self, proMP, regularizationCoeff=10**-9, priorCovariance=10**-4, priorWeight=1):
        self.proMP = proMP
        self.priorCovariance = priorCovariance
        self.priorWeight = priorWeight
        self.regularizationCoeff = regularizationCoeff

    def learnFromData(self, trajectoryList, timeList):

        numTraj = len(trajectoryList)
        weightMatrix = np.zeros((numTraj, self.proMP.numWeights))
        for i in range(numTraj):

            trajectory = trajectoryList[i]
            time = timeList[i]
            trajectoryFlat = trajectory.transpose().reshape(trajectory.shape[0] * trajectory.shape[1])
            basisMatrix = self.proMP.basis.basisMultiDoF(time, self.proMP.numDoF)
            temp = basisMatrix.transpose().dot(basisMatrix) + np.eye(self.proMP.numWeights) * self.regularizationCoeff
            weightVector = np.linalg.solve(temp, basisMatrix.transpose().dot(trajectoryFlat))
            weightMatrix[i, :] = weightVector

        self.proMP.mu = np.mean(weightMatrix, axis=0)

        sampleCov = np.cov(weightMatrix.transpose())
        self.proMP.covMat = (numTraj * sampleCov + self.priorCovariance * np.eye(self.proMP.numWeights)) / (numTraj + self.priorCovariance)


if __name__ == "__main__":
    hh = load_data.LoadData
    aa = hh.get_no_feedback
    print(aa.shape)
    # with open('/home/mithun/promp/examples/python_promp/data_jayanth/data/sa00_nofeedback0.json') as json_file1:
    #     data1 = json.load(json_file1)
    # with open('/home/mithun/promp/examples/python_promp/data_jayanth/data/sa00_nofeedback1.json') as json_file2:
    #     data2 = json.load(json_file2)
    # with open('/home/mithun/promp/examples/python_promp/data_jayanth/data/sa00_nofeedback2.json') as json_file3:
    #     data3 = json.load(json_file3)
    #
    # with open('/home/mithun/promp/examples/python_promp/data_jayanth/data/br00_nofeedback0.json') as json_file4:
    #     data4 = json.load(json_file4)
    # with open('/home/mithun/promp/examples/python_promp/data_jayanth/data/br00_nofeedback1.json') as json_file5:
    #     data5 = json.load(json_file5)
    # with open('/home/mithun/promp/examples/python_promp/data_jayanth/data/br00_nofeedback2.json') as json_file6:
    #     data6 = json.load(json_file6)
    #
    # with open('/home/mithun/promp/examples/python_promp/data_jayanth/data/la00_nofeedback0.json') as json_file7:
    #     data7 = json.load(json_file7)
    # with open('/home/mithun/promp/examples/python_promp/data_jayanth/data/la00_nofeedback1.json') as json_file8:
    #     data8 = json.load(json_file8)
    # with open('/home/mithun/promp/examples/python_promp/data_jayanth/data/la00_nofeedback2.json') as json_file9:
    #     data9 = json.load(json_file9)
    #
    # with open('/home/mithun/promp/examples/python_promp/data_jayanth/data/va00_nofeedback0.json') as json_file10:
    #     data10 = json.load(json_file10)
    # with open('/home/mithun/promp/examples/python_promp/data_jayanth/data/va00_nofeedback1.json') as json_file11:
    #     data11 = json.load(json_file11)
    # with open('/home/mithun/promp/examples/python_promp/data_jayanth/data/va00_nofeedback2.json') as json_file12:
    #     data12 = json.load(json_file12)
    #
    # with open('/home/mithun/promp/examples/python_promp/data_jayanth/data/di00_nofeedback0.json') as json_file13:
    #     data13 = json.load(json_file13)
    # with open('/home/mithun/promp/examples/python_promp/data_jayanth/data/di00_nofeedback1.json') as json_file14:
    #     data14 = json.load(json_file14)
    # with open('/home/mithun/promp/examples/python_promp/data_jayanth/data/di00_nofeedback2.json') as json_file15:
    #     data15 = json.load(json_file15)

    with open('/home/mithun/promp/examples/python_promp/data_jayanth/data/sa01_torquefeedback0.json') as json_file1:
        data1 = json.load(json_file1)
    with open('/home/mithun/promp/examples/python_promp/data_jayanth/data/sa01_torquefeedback1.json') as json_file2:
        data2 = json.load(json_file2)
    with open('/home/mithun/promp/examples/python_promp/data_jayanth/data/sa01_torquefeedback2.json') as json_file3:
        data3 = json.load(json_file3)

    with open('/home/mithun/promp/examples/python_promp/data_jayanth/data/br01_torquefeedback2.json') as json_file4:
        data4 = json.load(json_file4)
    with open('/home/mithun/promp/examples/python_promp/data_jayanth/data/br01_torquefeedback3.json') as json_file5:
        data5 = json.load(json_file5)
    with open('/home/mithun/promp/examples/python_promp/data_jayanth/data/br01_torquefeedback4.json') as json_file6:
        data6 = json.load(json_file6)

    with open('/home/mithun/promp/examples/python_promp/data_jayanth/data/la01_torquefeedback0.json') as json_file7:
        data7 = json.load(json_file7)
    with open('/home/mithun/promp/examples/python_promp/data_jayanth/data/la01_torquefeedback1.json') as json_file8:
        data8 = json.load(json_file8)
    with open('/home/mithun/promp/examples/python_promp/data_jayanth/data/la01_torquefeedback2.json') as json_file9:
        data9 = json.load(json_file9)

    with open('/home/mithun/promp/examples/python_promp/data_jayanth/data/va01_torquefeedback0.json') as json_file10:
        data10 = json.load(json_file10)
    with open('/home/mithun/promp/examples/python_promp/data_jayanth/data/va01_torquefeedback1.json') as json_file11:
        data11 = json.load(json_file11)
    with open('/home/mithun/promp/examples/python_promp/data_jayanth/data/va01_torquefeedback3.json') as json_file12:
        data12 = json.load(json_file12)

    with open('/home/mithun/promp/examples/python_promp/data_jayanth/data/di01_torquefeedback0.json') as json_file13:
        data13 = json.load(json_file13)
    with open('/home/mithun/promp/examples/python_promp/data_jayanth/data/di01_torquefeedback1.json') as json_file14:
        data14 = json.load(json_file14)
    with open('/home/mithun/promp/examples/python_promp/data_jayanth/data/di01_torquefeedback2.json') as json_file15:
        data15 = json.load(json_file15)

    # cpos1 = np.array(data1['slave_c_pos'])
    # cpos2 = np.array(data2['slave_c_pos'])
    # cpos3 = np.array(data3['slave_c_pos'])
    # cpos4 = np.array(data4['slave_c_pos'])
    # cpos5 = np.array(data5['slave_c_pos'])
    # cpos6 = np.array(data6['slave_c_pos'])
    # cpos7 = np.array(data7['slave_c_pos'])
    # cpos8 = np.array(data8['slave_c_pos'])
    # cpos9 = np.array(data9['slave_c_pos'])
    # cpos10 = np.array(data10['slave_c_pos'])
    # cpos11 = np.array(data11['slave_c_pos'])
    # cpos12 = np.array(data12['slave_c_pos'])
    # cpos13 = np.array(data13['slave_c_pos'])
    # cpos14 = np.array(data14['slave_c_pos'])
    # cpos15 = np.array(data15['slave_c_pos'])

    cpos1 = np.array(data1['mcurr_load'])
    cpos2 = np.array(data2['mcurr_load'])
    cpos3 = np.array(data3['mcurr_load'])
    cpos4 = np.array(data4['mcurr_load'])
    cpos5 = np.array(data5['mcurr_load'])
    cpos6 = np.array(data6['mcurr_load'])
    cpos7 = np.array(data7['mcurr_load'])
    cpos8 = np.array(data8['mcurr_load'])
    cpos9 = np.array(data9['mcurr_load'])
    cpos10 = np.array(data10['mcurr_load'])
    cpos11 = np.array(data11['mcurr_load'])
    cpos12 = np.array(data12['mcurr_load'])
    cpos13 = np.array(data13['mcurr_load'])
    cpos14 = np.array(data14['mcurr_load'])
    cpos15 = np.array(data15['mcurr_load'])

    print(cpos1.shape,'data1',cpos2.shape,'data2',cpos3.shape,'data3')

    [row1, column1] = cpos1.shape
    [row2, column2] = cpos2.shape
    [row3, column3] = cpos3.shape

    [row4, column1] = cpos4.shape
    [row5, column2] = cpos5.shape
    [row6, column3] = cpos6.shape

    [row7, column1] = cpos7.shape
    [row8, column2] = cpos8.shape
    [row9, column3] = cpos9.shape

    [row10, column1] = cpos10.shape
    [row11, column2] = cpos11.shape
    [row12, column3] = cpos12.shape

    [row13, column1] = cpos13.shape
    [row14, column2] = cpos14.shape
    [row15, column3] = cpos15.shape

    t1 = np.linspace(0, 1, row1)
    t2 = np.linspace(0, 1, row2)
    t3 = np.linspace(0, 1, row3)
    t4 = np.linspace(0, 1, row4)
    t5 = np.linspace(0, 1, row5)
    t6 = np.linspace(0, 1, row6)
    t7 = np.linspace(0, 1, row7)
    t8 = np.linspace(0, 1, row8)
    t9 = np.linspace(0, 1, row9)
    t10 = np.linspace(0, 1, row10)
    t11 = np.linspace(0, 1, row11)
    t12 = np.linspace(0, 1, row12)
    t13 = np.linspace(0, 1, row13)
    t14 = np.linspace(0, 1, row14)
    t15 = np.linspace(0, 1, row15)


    # QQ = np.array([cpos1, cpos2, cpos3])
    # time_t = np.array([t1, t2, t3])

    # QQ = np.array([cpos1, cpos3])
    # time_t = np.array([t1, t3])

    QQ = np.array([cpos1, cpos2, cpos3, cpos4, cpos5, cpos6, cpos7, cpos8, cpos9, cpos10, cpos11, cpos12, cpos13, cpos14, cpos15])
    time_t = np.array([t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15])

    print(QQ.shape)
    print(time_t.shape)



    phaseGenerator = phase.LinearPhaseGenerator()
    basisGenerator = basis.NormalizedRBFBasisGenerator(phaseGenerator, numBasis=15, duration=1, basisBandWidthFactor=3, numBasisOutside=1)
    time = np.linspace(0, 1, 100)
    nDof = 7
    # nDof = 3
    proMP = ProMP(basisGenerator, phaseGenerator, nDof)   # 3 argument = nDOF
    # trajectories = proMP.getTrajectorySamples(time, 4)   # 2nd argument is numSamples/Demonstrations/trajectories
    # print(trajectories.shape)
    # meanTraj, covTraj = proMP.getMeanAndCovarianceTrajectory(time)
    plotDof = 5
    # sample_time = np.linspace(0, 30000, 30000)
    # plt.figure()
    # plt.plot(QQ[0][:, 2], color='green')
    # plt.show()
    # plt.figure()
    # plt.plot(time, trajectories[:, 2, :])
    # plt.show()
    # #
    # plt.figure()
    # plt.plot(time, meanTraj[:, 0])

    learnedProMP = ProMP(basisGenerator, phaseGenerator, nDof)
    learner = MAPWeightLearner(learnedProMP)
    # trajectoriesList = []
    # timeList = []
    #
    # for i in range(trajectories.shape[2]):
    #     trajectoriesList.append(trajectories[:, :, i])
    #     timeList.append(time)

    # learner.learnFromData(trajectoriesList, timeList)
    # time = np.linspace(0, 30000, 30000)
    learner.learnFromData(QQ, time_t)
    # trajectories = learnedProMP.getTrajectorySamples(time, 10)
    trajectories = learnedProMP.getTrajectorySamples(time, 10)
    print(trajectories.shape)
    plt.figure()
    plt.plot(time, trajectories[:, plotDof, :])
    # plt.plot(time)
    plt.xlabel('time')
    plt.title('learnedProMP')

    # plt.figure()
    # plt.plot(QQ[0][:, plotDof], color='green')
    # plt.plot(QQ[1][:, plotDof], color='red')
    # # plt.plot(QQ[2][:, plotDof], color='blue')
    plt.figure()
    for i in range(6):
        plt.plot(QQ[i][:, plotDof])
        # plt.plot(QQ[2][:, plotDof], color='blue')

    phaseGeneratorSmooth = phase.SmoothPhaseGenerator(duration = 1)
    proMPSmooth = ProMP(basisGenerator, phaseGeneratorSmooth, nDof)
    proMPSmooth.mu = learnedProMP.mu
    proMPSmooth.covMat = learnedProMP.covMat

    trajectories = proMPSmooth.getTrajectorySamples(time, 10)
    plt.figure()
    plt.plot(time, trajectories[:, plotDof, :], '--')
    plt.title('learnedProMPSmooth')
    plt.show()
################################################################

    # Conditioning in JointSpace
    desiredTheta = np.array([0.5, 0.7, 0.9, 0.2, 0.6, 0.8, 0.1])
    desiredVar = np.eye(len(desiredTheta)) * 0.0001
    newProMP = proMP.jointSpaceConditioning(0.5, desiredTheta=desiredTheta, desiredVar=desiredVar)
    trajectories = newProMP.getTrajectorySamples(time, 4)
    plt.figure()
    plt.plot(time, trajectories[:, plotDof, :])
    plt.xlabel('time')
    plt.title('Joint-Space conditioning')
    # newProMP.plotProMP(time, [3,4])

    plt.show()






