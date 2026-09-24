'''
tests.py
Author: Andrew Gaylord

Contains statistical, efficiency, and correctness for kalman filter objects
Following statistical tests notes reference Estimation II by Ian Reid

innovation (V) or residual: difference between a measurement and its prediction at time k
    measures new info provided by adding another measurement in estimation
    can be used to validate a measurement before adding to observation sequence

    innovation tests: test that innovation has zero mean and white with cov S_k
        1) check that it is consistent with its cov/within bounds--checks filter consistency
        2) chi square test for unbiased (normalized innovation squared)
        3) whiteness test (autocorrelation)

'''


'''
one of the ways to check Kalman filters performance is to check for error covariance matrix P 
to be converging. If it converges to + or - standard deviation of the estimated value, 
it can be considered as a stable point. 
calculate square of difference between estimated and real
You can verify that the estimated state converges to the actual state. 
The error covariance, P, must decrease.
'''

import numpy as np
from graphing import *
from scipy.stats import chi2
import scipy.signal as signal


# test #1 for filter consistency
# function that returns true if 95% of the innovations are within 2 standard deviations of the mean
# parameters: innovation arrays, innovation covariance arrays
def innovationTest(innovations, innovationCovs, dim_mes):
    innovationMags = np.array([np.linalg.norm(x) for x in innovations])
    innovationCovMags = innovationCovToStd(innovationCovs, dim_mes)

    upper = innovationMags + 2 * innovationCovMags
    lower = innovationMags - 2 * innovationCovMags

    # only returns true if 95% of innovations are within 2 standard deviations
    return np.sum((innovationMags < upper) & (innovationMags > lower)) / len(innovationMags) > .95


def plotInnovations(innovations, innovationCovs):
    # method for plotting each innovation and their respective standard deviations on separate graphs
    # for i in range(len(innovations[0])):
    #     sds =  np.array([np.sqrt(np.diag(x)) for x in innovationCovs])
    #     plot_multiple_lines(np.array([innovations[:, i], innovations[:, i] + 2 * sds[:, i], innovations[:, i] - 2 * sds[:, i]]), ["innovation magnitude", "upper sd", "lower sd"], "innovation " + str(i+1), 100 + i*50, 100 + i*50)

    # method using magnitude of whole innovation array
    # find magnitudes of innovation arrays
    # innovationMags = np.array([np.linalg.norm(x) for x in ukf.innovations])

    # to get standard deviation, take sqrt of diagonal
    # divide by number of observations to get standard error of mean
    # get magnitude afterwards
    # innovationCovMags = np.array([(np.linalg.norm(y)/ ukf.dim_mes) for y in np.array([np.sqrt(np.diag(x)) for x in ukf.innovationCovs])])

    # find upper and lower bounds of 2 * standard deviation
    # upper = innovationMags + 2 * innovationCovMags
    # lower = innovationMags - 2 * innovationCovMags

    # plot to check whether innovation is centered on 0 and 95% of measurements are consistent with standard deviation
    # plot_multiple_lines(np.array([innovationMags, upper, lower]), ["innovation magnitude", "upper sd", "lower sd"], "innovation", 300, 200)

    # print("Test #1: for innovation consistency, 95% of innovations must be within confidence interval bounds")
    # plt.figure(figsize=(3, 3))
    
    # plot orientation and velocity innovations on separate graphs
    plotVelocityInnovations(innovations, innovationCovs)
    plotOrientationInnovations(innovations, innovationCovs)

# plot the last 3 innovations and their respective standard deviations on 1 graph
def plotVelocityInnovations(innovations, innovationCovs):
    innovations = innovations[1:]
    innovationCovs = innovationCovs[1:]

    sds =  np.array([np.sqrt(np.diag(x)) for x in innovationCovs])
    # text = "95% of innovations should be within confidence interval bounds"
    text = ""
    # NOTE: should 2 * bound be added to mean, or 0??
    # plot_multiple_lines(np.array([innovations[:, 3], innovations[:, 3] + 2 * sds[:, 3], innovations[:, 3] - 2 * sds[:, 3], innovations[:, 4], innovations[:, 4] + 2 * sds[:, 4], innovations[:, 4] - 2 * sds[:, 4], innovations[:, 5], innovations[:, 5] + 2 * sds[:, 5], innovations[:, 5] - 2 * sds[:, 5]]), ["velocity 1", "upper 1", "lower 1", "velocity 2", "upper 2", "lower 2", "velocity 3", "upper 3", "lower 3"], "innovation magnitudes", 900, 100)
    # plot_multiple_lines(np.array([innovations[:, 3], 2 * sds[:, 3], - 2 * sds[:, 3], innovations[:, 4], 2 * sds[:, 4], - 2 * sds[:, 4], innovations[:, 5], 2 * sds[:, 5], - 2 * sds[:, 5]]), ["velocity 1", "upper 1", "lower 1", "velocity 2", "upper 2", "lower 2", "velocity 3", "upper 3", "lower 3"], "velocity innovation magnitudes", 900, 100, "TEST")
    plot_multiple_lines(np.array([innovations[:, 3], 2 * sds[:, 3], - 2 * sds[:, 3], innovations[:, 4], innovations[:, 5]]), ["velocity 1", "upper sd", "lower sd", "velocity 2", "velocity 3"], "Test 1: velocity innovation magnitudes", 900, 100, text, fileName="test1-2.png")

    
    
# plot the first 3 innovations on their respective standard deviations on 1 graph
def plotOrientationInnovations(innovations, innovationCovs):
    innovations = innovations[1:]
    innovationCovs = innovationCovs[1:]

    sds =  np.array([np.sqrt(np.diag(x)) for x in innovationCovs])
    # text = "95% of innovations should be within confidence interval bounds"
    text = ""
    # NOTE: should 2 * bound be added to mean, or 0??
    # plot_multiple_lines(np.array([innovations[:, 0], innovations[:, 0] + 2 * sds[:, 0], innovations[:, 0] - 2 * sds[:, 0], innovations[:, 1], innovations[:, 1] + 2 * sds[:, 1], innovations[:, 1] - 2 * sds[:, 1], innovations[:, 2], innovations[:, 2] + 2 * sds[:, 2], innovations[:, 2] - 2 * sds[:, 2]]), ["orientation 1", "upper 1", "lower 1", "orientation 2", "upper 2", "lower 2", "orientation 3", "upper 3", "lower 3"], "orientation innovation magnitudes", 100, 100)
    plot_multiple_lines(np.array([innovations[:, 0], 2 * sds[:, 0], - 2 * sds[:, 0], innovations[:, 1], 2 * sds[:, 1], - 2 * sds[:, 1], innovations[:, 2], 2 * sds[:, 2], - 2 * sds[:, 2]]), ["orientation 1", "upper 1", "lower 1", "orientation 2", "upper 2", "lower 2", "orientation 3", "upper 3", "lower 3"], "Test 1: orientation innovation magnitudes", 100, 100, text, fileName="test1-1.png")





# test #2 for unbiasedness
# calculates and plots normalised innovation squared, returns whether each innovation fits chi square distribution
def plotInnovationSquared(innovations, innovationCovs):
    # normInnovSquared = np.atleast_2d(innovations[:, 0]).T.conj() * np.linalg.inv(innovationCovs) * innovations[:, 0]
    # normInnovSquared = innovations[:, 0] * np.linalg.inv(innovationCovs) * innovations[:, 0]
    # normInnovSquared = innovations * np.linalg.inv(innovationCovs) * innovations

    print("Test #2: for unbiasedness, the sum of the normalised innovations squared must be within confidence interval bounds")
    print("MORE INFO NEEDED")

    # calculate normalised innovation squared of all innovations combined
    normInnovSquared = np.zeros((len(innovations), 6, 6))

    for i in range(len(innovations)):
        normInnovSquared[i] = innovations[i] * np.linalg.inv(innovationCovs[i]) * innovations[i]

    # normalize the diagonal of the 6x6 matrix
    # normInnovSquared = np.array([np.linalg.norm(np.diag(x)) for x in normInnovSquared])
    normInnovSquared = np.array([np.linalg.norm(x) for x in normInnovSquared])

    # sum of normalised innovation squared combined
    print("combined sum: ", np.sum(normInnovSquared))

    plot_multiple_lines(np.array([normInnovSquared]), ["normalised innovation squared"], "innovation squared combined", 100, 100, fileName="test2-1.png")


    # find normalised innovation squared all innovations separately
    # the innovations squared should each be chi squared distributed with 6 degrees of freedom
    squares = np.zeros((len(innovations), len(innovations[0])))
    for i in range(len(innovations)):
        # which one is correct??
        squares[i] = innovations[i] * np.diag(np.linalg.inv(innovationCovs[i])) * innovations[i]
        # squares[i] = innovations[i] * np.diag(np.linalg.inv(np.diag(np.diag(innovationCovs[i])))) * innovations[i]


    # find sum of normalised innovations squared
    sumInnovSquared = np.sum(squares, axis=0)

    # average of each normalised innovation squared
    aves = np.zeros(len(innovations[0]))
    for i in range(len(innovations[0])):
        aves[i] = np.mean(squares[:, i])
    # aves = np.array([np.mean(x) for x in squares])

    print("Sum of separate squared innovations: ", sumInnovSquared)

    print("Averages of separate squared innovations: ", aves)

    print("sum of seperate combined squared innovations: ", np.sum(sumInnovSquared))

    # plot normalised innovation squared
    plot_multiple_lines(np.array([squares[:, 0]]), ["normalised innovation squared"], "orientation 1", 200, 200, text="sum: " + str(round(sumInnovSquared[0], 3)) + " avrg: " + str(round(aves[0], 3)), fileName="test2-2-1.png")
    plot_multiple_lines(np.array([squares[:, 1]]), ["normalised innovation squared"], "orientation 2", 300, 200, text="sum: " + str(round(sumInnovSquared[1], 3)) + " avrg: " + str(round(aves[1], 3)), fileName="test2-2-2.png")
    plot_multiple_lines(np.array([squares[:, 2]]), ["normalised innovation squared"], "orientation 3", 400, 200, text="sum: " + str(round(sumInnovSquared[2], 3)) + " avrg: " + str(round(aves[2], 3)), fileName="test2-2-3.png")
    plot_multiple_lines(np.array([squares[:, 3]]), ["normalised innovation squared"], "velocity 3", 500, 200, text="sum: " + str(round(sumInnovSquared[3], 3)) + " avrg: " + str(round(aves[3], 3)), fileName="test2-2-4.png")
    plot_multiple_lines(np.array([squares[:, 4]]), ["normalised innovation squared"], "velocity 3", 600, 200, text="sum: " + str(round(sumInnovSquared[4], 3)) + " avrg: " + str(round(aves[4], 3)), fileName="test2-2-5.png")
    plot_multiple_lines(np.array([squares[:, 5]]), ["normalised innovation squared"], "velocity 3", 700, 200, text="sum: " + str(round(sumInnovSquared[5], 3)) + " avrg: " + str(round(aves[5], 3)), fileName="test2-2-6.png")



    # because innovation is ergodic, we can find sample mean as a moving average of 1 run instead using N independent samples 
    # print("Averages of separate squared innovations: ", np.mean(squares, axis=0))

    # find confidence interval for chi square test for unbaisedness
    # TODO: should chi squared interval have 6, 100, or 600 degrees of freedom??
    #   article claims that it should be n * m, where n is number of observations and m is number of measurements

    # interval that sum must be within
    # or could divide by number of observations, and average must be within interval divided by n too
    # (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html)
    interval = chi2.interval(0.95, 100)

    print("interval with " + str(len(innovations)) + " df: ", chi2.interval(0.95, len(innovations)))
    print("interval with n*m = " + str(len(innovations[0])*len(innovations)) + " df: ", chi2.interval(0.95, len(innovations[0])*len(innovations)))
    
    return np.sum(sumInnovSquared)

    # sum for each innovation must be within 95% confidence interval
    #   otherwise, infer whether combined noise is too large/too small

    # if too small (lower end of interval) then the filter is too confident in its measurements (??)
    #       i.e. measurement/process noise combined is too large/overestimated (facts)
    # if too large (upper end of interval), then the filter is not confident enough in its measurements (??)
    #       i.e. measurement/process noise combined is too small/underestimated
    #       smaller noise = larger normalized innovations squared

    # note: if the sum of the normalised innovation squared is within the confidence interval, then the filter is unbiased
    # absolute value of noises may be tuned to pass chi square test

    # tuning may be more sensitive to changes in measurement/process noise if one affects orientation instead of just velocity




# test #3 for whiteness (autocorrelation)
# function that finds and plots normalised autocorrelation for each innovation sequence in a 2D array
def plotAutocorrelation(innovations):

    # analyze for time dependency: can mean over/underestimation of process/measurement noise
    #       white noise = disrtributed around 0 entire time = good fit
    # ideally, autocorrelation for each var should be within 95% of confidence interval
    # smaller noises = smaller distribution = more likely to in bounds
    print("Analyze autocorrelation of innovations:")
    print("if there is time correlation (i.e. not randomly distributed around 0), then test for whiteness fails")
    print("process/measurement noise may be too high or too low")

    # fig.subplots_adjust(bottom=0.3)
    # plt.figure(figsize=(3, 3))
    # fig.text(x, y, "text")

    n = len(innovations)

    correlations = np.zeros((6, n-1))

    # set bounds for 95% confidence interval
    lower = np.array([-2 / np.sqrt(n)] * (n-1))
    upper = np.array([2 / np.sqrt(n)] * (n-1))

    for i in range(len(innovations[0])):
        # find autocorrelation of each innovation
        # basically, autocorrelation represents the correlation between a time series and its own lagged values
        r = np.correlate(innovations[:, i], innovations[:, i], mode='full')
        
        # get second half of autocorration bc it starts at negative instead of 0
        r = r[n:2*n-1]

        # normalize by dividing by the first element of the positive autocorrelation
        r = r / r[0]

        # check if 95% of the values in r fall with the upper and lower bounds
        print("95% of autocorrelation {} with 95% bounds: {}".format(i+1, np.sum((r < upper) & (r > lower)) / len(r) > .95))

        correlations[i] = r

    # plot autocorrelation for orientation innovations
    # plot_multiple_lines(np.array([correlations[0], correlations[1], correlations[2], lower, upper]), ["auto 1", "auto 2", "auto 3", "lower sd", "upper sd"], "orientation autocorrelations", 100, 100, fileName="test3-1")

    # plot autocorrelation for velocity innovations
    # plot_multiple_lines(np.array([correlations[3], correlations[4], correlations[5], lower, upper]), ["auto 4", "auto 5", "auto 6", "lower sd", "upper sd"], "velocity autocorrelations", 600, 100, fileName="test3-2")

    zeros = np.zeros((n-1))

    # plot each autocorrelation on its own graph
    plot_multiple_lines(np.array([correlations[0], zeros]), ["autocorrelation", "zero"], "orientation 1", 200, 200, fileName="test3-1.png")
    plot_multiple_lines(np.array([correlations[1], zeros]), ["autocorrelation", "zero"], "orientation 2", 300, 200, fileName="test3-2.png")
    plot_multiple_lines(np.array([correlations[2], zeros]), ["autocorrelation", "zero"], "orientation 3", 400, 200, fileName="test3-3.png")
    plot_multiple_lines(np.array([correlations[3], zeros]), ["autocorrelation", "zero"], "velocity 1", 500, 200, fileName="test3-4.png")
    plot_multiple_lines(np.array([correlations[4], zeros]), ["autocorrelation", "zero"], "velocity 2", 600, 200, fileName="test3-5.png")
    plot_multiple_lines(np.array([correlations[5], zeros]), ["autocorrelation", "zero"], "velocity 3", 700, 200, fileName="test3-6.png")







# find the normalized autocorrelation for an array of residuals/innovations
def autocorrelation(innovations):
    # find the mean of the innovations
    mean = np.mean(innovations)

    # find the autocorrelation of the innovations
    autocorrelation = np.correlate(innovations - mean, innovations - mean, mode='full') / (np.var(innovations) * len(innovations))

    return autocorrelation

# function that returns true if the innovations are white
def whiteTest(innovations):
    return np.allclose(autocorrelation(innovations), 0, atol=1e-2)


# function that returns true if the error covariance matrix P is converging
def covarianceConvergence(P, threshold):
    return np.allclose(P, threshold, atol=1e-2)

# function that takes in an array of innovation covariances and returns an array of the standard deviations of the innovations
def innovationCovToStd(innovationCovs, dim_mes):
    return np.array([(np.linalg.norm(y)/ dim_mes) for y in np.array([np.sqrt(np.diag(x)) for x in innovationCovs])])







# function that returns true if the innovations are consistent with their covariance
# def consistentTest(innovations, innovationCovs):
    # return innovationTest(innovations, innovationCovs, 6) and whiteTest(innovations) and unbiasedTest(innovations)

# function that finds normalised innovation and moving average
def movingAverage(innovations, window):
    return np.convolve(innovations, np.ones(window), 'valid') / window

# function that performs normalised innovations chi square test 
def chiSquareTest(innovations, innovationCovs):
    return np.sum(innovations ** 2 / innovationCovs) / len(innovations) < 1.5