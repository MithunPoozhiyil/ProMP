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

    dataset = load_data.LoadData()
    (nf_slave_c_pos, nf_master_j_pos, nf_master_j_vel, nf_mcurr_load) = dataset.get_no_feedback()
    (tf_slave_c_pos, tf_master_j_pos, tf_master_j_vel, tf_mcurr_load) = dataset.get_torque_feedback()
    (pf_slave_c_pos, pf_master_j_pos, pf_master_j_vel, pf_mcurr_load) = dataset.get_position_feedback()

    time_data = []
    for i in range(len(pf_slave_c_pos)):
        # time_data.append(np.linspace(0, 1, len(nf_slave_c_pos[i])))
        # time_data.append(np.linspace(0, 1, len(tf_slave_c_pos[i])))
        time_data.append(np.linspace(0, 1, len(pf_slave_c_pos[i])))
    # for i in range(12):
    #     time_data.append(np.linspace(0, 1, len(nf_slave_c_pos[i])))

    # QQ = nf_slave_c_pos
    # QQ = tf_slave_c_pos
    QQ = pf_slave_c_pos
    time_t = time_data
    # for i in range(QQ.size):

    # QQ = np.array([cpos1, cpos2, cpos3, cpos4, cpos5, cpos6, cpos7, cpos8, cpos9, cpos10, cpos11, cpos12, cpos13, cpos14, cpos15])
    # time_t = np.array([t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15])

    # print(QQ.shape)
    # print(time_t.shape)



    phaseGenerator = phase.LinearPhaseGenerator()
    basisGenerator = basis.NormalizedRBFBasisGenerator(phaseGenerator, numBasis=15, duration=1, basisBandWidthFactor=3, numBasisOutside=1)
    time = np.linspace(0, 1, 100)
    nDof = 3
    plotDof = 0
    proMP = ProMP(basisGenerator, phaseGenerator, nDof)   # 3 argument = nDOF
    learnedProMP = ProMP(basisGenerator, phaseGenerator, nDof)
    learner = MAPWeightLearner(learnedProMP)
    learner.learnFromData(QQ, time_t)
    trajectories = learnedProMP.getTrajectorySamples(time, 10)
    print(trajectories.shape)
    plt.figure()
    plt.plot(time, trajectories[:, plotDof, :])
    plt.xlabel('time')
    plt.title('learnedProMP')

    plt.figure()
    for i in range(len(nf_slave_c_pos)):
        plt.plot(QQ[i][:, plotDof])
        # plt.plot(QQ[2][:, plotDof], color='blue')
    plt.title('x-pose')
    ################################################################
    phaseGeneratorSmooth = phase.SmoothPhaseGenerator(duration = 1)
    proMPSmooth = ProMP(basisGenerator, phaseGeneratorSmooth, nDof)
    proMPSmooth.mu = learnedProMP.mu
    proMPSmooth.covMat = learnedProMP.covMat

    trajectories = proMPSmooth.getTrajectorySamples(time, 50)
    plt.figure()
    plt.plot(time, trajectories[:, plotDof, :], '--')
    plt.title('learnedProMPSmooth')
    plt.show()
    ################################################################

    # # Conditioning in JointSpace
    # desiredTheta = np.array([0.5, 0.7, 0.9, 0.2, 0.6, 0.8, 0.1])
    # desiredVar = np.eye(len(desiredTheta)) * 0.0001
    # newProMP = proMP.jointSpaceConditioning(0.5, desiredTheta=desiredTheta, desiredVar=desiredVar)
    # trajectories = newProMP.getTrajectorySamples(time, 4)
    # plt.figure()
    # plt.plot(time, trajectories[:, plotDof, :])
    # plt.xlabel('time')
    # plt.title('Joint-Space conditioning')
    # # newProMP.plotProMP(time, [3,4])
    #
    # plt.show()






