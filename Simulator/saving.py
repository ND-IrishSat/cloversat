'''
saving.py
Author: Andrew Gaylord

contains the saving functionality for kalman filter visualization
saves graphs to png and then embeds them in a pdf with contextual information

'''

import os
from matplotlib.backends.backend_pdf import PdfPages
import subprocess
from fpdf import FPDF
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

import os
import sys

# import params module from parent directory
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from params import *

def saveFig(fig, fileName):
    '''
    saves fig to a png file in the outputDir directory with the name fileName
    also closes fig
    '''

    # absolute path to current directory
    my_path = os.path.dirname(os.path.abspath(__file__))
    saveDirectory = os.path.join(my_path, OUTPUT_DIR)

    fig.savefig(os.path.join(saveDirectory, fileName))

    plt.close(fig)


def savePDF(outputFile, pngDir, sim):
    '''
    creates a simulation report using FPDF with all PNGs found in pngDir
    Describes the different graphs and their significance

    @params:
        outputFile: name of pdf to be generated
        pngDir: name of folder where graph PNGs are found
        sim: Simulator object with sim info
    '''

    # absolute path to current directory
    my_path = os.path.dirname(os.path.abspath(__file__))
    pngDirectory = os.path.join(my_path, pngDir)

    # create the PDF object
    pdf = FPDF()
    title = "Cubesat Simulation Report"
    pdf.set_author("Andrew Gaylord")
    pdf.set_title(title)
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=11)

    x_offset = 12
    y_pic_offset = 134

    # title and document details
    pdfHeader(pdf, title)

    pdf.image(os.path.join(pngDirectory, "magData.png"), x=x_offset, y=pdf.get_y(), w=180)
    pdf.ln(y_pic_offset)
    pdf.image(os.path.join(pngDirectory, "B_earth.png"), x=x_offset, y=pdf.get_y(), w=180)

    pdf.add_page()
    pdfHeader(pdf, "Orientation and Angular Velocity")

    pdf.image(os.path.join(pngDirectory, "Quaternion.png"), x=x_offset, y=pdf.get_y(), w=180)
    pdf.ln(y_pic_offset)
    pdf.image(os.path.join(pngDirectory, "Velocity.png"), x=x_offset, y=pdf.get_y(), w=180)

    pdf.add_page()
    pdf.image(os.path.join(pngDirectory, "Velocity_Magnitude.png"), x = x_offset, y = pdf.get_y(), w = 180)
    pdf.ln(y_pic_offset)
    if PROTOCOL_MAP["point"] in sim.mode:
        pdf.image(os.path.join(pngDirectory, "Error.png"), x = x_offset, y = pdf.get_y(), w = 180)
    else:
        pdf.image(os.path.join(pngDirectory, "gyroData.png"), x=x_offset, y=pdf.get_y(), w=180)

    pdf.add_page()
    if RUN_UKF:
        pdfHeader(pdf, "Filter Results")

        filterText = f"""Kalman filter estimates our state each time step by combining noisy data and physics EOMs over {sim.n * sim.dt} seconds."""

        pdf.multi_cell(0, 5, filterText, 0, 'L')

        pdf.image(os.path.join(pngDirectory, "filteredQuaternion.png"), x=x_offset, y=pdf.get_y(), w=180)
        pdf.ln(128)
        pdf.image(os.path.join(pngDirectory, "filteredVelocity.png"), x=x_offset, y=pdf.get_y(), w=180)

        pdf.add_page()

    if not np.array_equal(RW_AXES, np.array([0, 0, 0, 0])):
        # If we're simulating any reaction wheels, include their graphs
        pdfHeader(pdf, "Reaction Wheels")

        pdf.image(os.path.join(pngDirectory, "PWM.png"), x = x_offset, y = pdf.get_y(), w = 180)
        pdf.ln(128)
        pdf.image(os.path.join(pngDirectory, "ReactionCurrent.png"), x = x_offset, y = pdf.get_y(), w = 180)

        pdf.add_page()
        pdf.image(os.path.join(pngDirectory, "ReactionSpeeds.png"), x = x_offset, y = pdf.get_y(), w = 180)
        pdf.add_page()

    if not np.array_equal(MAG_AXES, np.array([0, 0, 0])):
        # removed total power for now
        # pdf.image(os.path.join(pngDirectory, "Total_Power_Output.png"), x = x_offset, y = pdf.get_y(), w = 180)

        pdf.image(os.path.join(pngDirectory, "MagVoltages.png"), x = x_offset, y = pdf.get_y(), w = 180)
        pdf.ln(y_pic_offset)
        pdf.image(os.path.join(pngDirectory, "MagCurrents.png"), x = x_offset, y = pdf.get_y(), w = 180)

        pdf.add_page()

        pdf.image(os.path.join(pngDirectory, "Power_Output.png"), x = x_offset, y = pdf.get_y(), w = 180)
        pdf.ln(y_pic_offset)
        pdf.image(os.path.join(pngDirectory, "MagTorques.png"), x = x_offset, y = pdf.get_y(), w = 180)

        pdf.add_page()

    # eulerText = f"""Our filtered orientation represented by Euler Angles (counterclockwise rotation about x, y, z). Can bug out sometimes. Near 180 degrees (pi) is the same as zero. """
    # pdf.multi_cell(0, 5, eulerText, 0, 'L')
    # pdf.image(os.path.join(pngDirectory, "Euler.png"), x=10, y=pdf.get_y(), w=180)

    # Construct text box dynamically based on param:
    magText = ""
    for key, value in PROTOCOL_MAP.items():
        magText += f"'{key}' = {value}\n"
    pdf.multi_cell(0, 5, magText, 0, 'L')
    pdf.image(os.path.join(pngDirectory, "Mode.png"), x = x_offset, y = pdf.get_y(), w = 180)

    pdf.ln(y_pic_offset)

    pdf.add_page()
    pdfHeader(pdf, "General Info")

    # set numpy printing option so that 0's don't have scientific notation
    np.set_printoptions(formatter={'all': lambda x: '{:<11d}'.format(int(x)) if x == 0 else "{:+.2e}".format(x)})
    pdf.set_font("Arial", size=13)

    if ACCURATE_MAG_READINGS:
        magReadingInfo = f"""
Magnetometer readings are taken every {MAG_READING_INTERVAL} seconds.
Magnetorquers are turned off for {TORQUER_OFF_TIME} seconds before each reading.
"""
    else:
        magReadingInfo = ""

    # if (not RUNNING_1D) or (RUNNING_1D and DETUMBLE_1D):
    if PROTOCOL_MAP["detumble"] in sim.mode:
        detumbleMessage = f"""
Hours completed during detumble: {round(sim.finishedTime/3600, 4)} hours.
Orbits completed during detumble: {round(sim.finishedTime/ORBITAL_PERIOD, 4)} orbits.

Power consumed to detumble (Total Energy): {int(sim.energy)} Jules
        """
    else:
        detumbleMessage = f"""Time to complete to reach 90 degrees: {round(sim.finishedTime, 4)} seconds """

    if RUN_UKF:
        filteringMessage = f"""{sim.n} filter iterations were completed in {round(np.sum(sim.times) * 1000, 2)} milliseconds. This kalman filter took {round(np.mean(sim.times) * 1000, 2)} ms per iteration.

Process Noise:

{sim.R}

Measurement Noise:

{sim.Q}"""
    else:
        filteringMessage = ""

    hardwareMessage = ""
    # Display info about which hardware we've simulated
    if not np.array_equal(RW_AXES, np.array([0, 0, 0, 0])):
        hardwareMessage += f"Reaction Wheels present: {RW_AXES}\n\n"
    if not np.array_equal(MAG_AXES, np.array([0, 0, 0])):
        # Place the correct mag_sat.mags here based on MAG_AXES
        hardwareMessage += f"Magnetorquers:\n"
        if MAG_AXES[0] != 0:
            hardwareMessage += f"  X-axis: {sim.mag_sat.mags[0]}\n"
        if MAG_AXES[1] != 0:
            hardwareMessage += f"  Y-axis: {sim.mag_sat.mags[1]}\n"
        if MAG_AXES[2] != 0:
            hardwareMessage += f"  Z-axis: {sim.mag_sat.mags[2]}\n"

    starting_speed = sim.states[0][4:]
    if not DEGREES:
        starting_speed *= 180/np.pi

    infoText = f"""Starting speed: {starting_speed} degrees/s.

Total simulation time: {round(float(sim.n) * sim.dt / 3600, 4)} hours

Timestep: {sim.dt} seconds

Orbits completed during simulation: {round((sim.n * sim.dt)/ORBITAL_PERIOD, 4)} orbits.

Orbital elments: {ORBITAL_ELEMENTS}
    These define our simulated orbit (see sol_sim.py in PySOL for more info)

{detumbleMessage}

{filteringMessage}

B-dot proportional gain: k = {sim.mag_sat.mags[0].k}

PD proportional gain: kp = {sim.mag_sat.kp}
PD derivative gain: kd = {sim.mag_sat.kd}

Satellite info:
{magReadingInfo}
{hardwareMessage}"""

    pdf.multi_cell(0, 5, infoText, 0, 'L')

    if RUN_STATISTICAL_TESTS:

        pdf.add_page()
        pdfHeader(pdf, "Tests")
        testText = f"""We have two metrics for examining our filter: statistical and speed tests.

    Speed tests:

    {sim.n} iterations were completed in {round(np.sum(sim.times) * 1000, 2)} milliseconds. This kalman filter took {round(np.mean(sim.times) * 1000, 2)} ms per iteration.

    The statistical tests are based on Estimation II by Ian Reid. He outlines 3 tests that examine the innovation (or residual) of the filter, which is the difference betwee a measurement and the filter's prediction.

    1) Consistency: the innovations should be randomly distributed about 0 and fall within its covariance bounds.

    2) Unbiasedness: the sum of the normalised innovations squared should fall within a 95% chi square confidence interval.
        If distribution sums are too small (fall below interval), then measurement/process noise is overestimated (too large). Therefore, the combined magnitude of the noises must be decreased.
        Conversely, measurement/process noise can be increased to lower the sums of the normalized innovations squared.

    3) Whiteness: autocorrelation should be distributed around 0 with no time dependecy."""

        pdf.multi_cell(0, 5, testText, 0, 'L')

        pdf.add_page()

        pdfHeader(pdf, "Test 1")

        pdf.multi_cell(0, 5, "Vizually inspect that 95% of innovations fall within confidence interval bounds.", 0, 'L')

        # split into 6 different graphs?
        pdf.image(os.path.join(pngDirectory, "test1-1.png"), x=10, y=pdf.get_y(), w=180)
        pdf.ln(128)
        pdf.image(os.path.join(pngDirectory, "test1-2.png"), x=10, y=pdf.get_y(), w=180)


        # test 2: show 6 graphs + combined? or do no graphs and just numbers?
        pdf.add_page()

        pdfHeader(pdf, "Test 2")

        # pdf.multi_cell(0, 5, "Sum of each innovation must be within chi square bounds " + str([round(x, 3) for x in chi2.interval(0.95, 100)]) + " (df=100)", 0, 'L')
        pdf.multi_cell(0, 5, "Sum of each innovation must be within chi square bounds {} (df={})".format(str([round(x, 3) for x in chi2.interval(0.95, sim.n)]), sim.n), 0, 'L')

        # pdf.multi_cell(0, 5, "Total sum " + str(round(sum, 3)) + " must be within interval " + str([round(x, 3) for x in chi2.interval(0.95, 600)]) + " (df=600)", 0, 'L')
        pdf.multi_cell(0, 5, "Total sum {} must be within 95% interval {} (df={})".format(str(round(sum, 3)), str([round(x, 3) for x in chi2.interval(0.95, sim.n*6)]), sim.n * sim.dim_mes), 0, 'L')

        pdf.multi_cell(0, 5, "If distributions are too small, decrease measurement/process noise (and vice versa)", 0, 'L')

        # split into 6 different graphs
        pdf.image(os.path.join(pngDirectory, "test2-2-1.png"), x=5, y=pdf.get_y(), w=105)
        pdf.image(os.path.join(pngDirectory, "test2-2-2.png"), x=100, y=pdf.get_y(), w=105)
        pdf.ln(80)
        pdf.image(os.path.join(pngDirectory, "test2-2-3.png"), x=5, y=pdf.get_y(), w=105)
        pdf.image(os.path.join(pngDirectory, "test2-2-4.png"), x=100, y=pdf.get_y(), w=105)
        pdf.ln(80)
        pdf.image(os.path.join(pngDirectory, "test2-2-5.png"), x=5, y=pdf.get_y(), w=105)
        pdf.image(os.path.join(pngDirectory, "test2-2-6.png"), x=100, y=pdf.get_y(), w=105)

        pdf.add_page()

        pdfHeader(pdf, "Test 3")

        pdf.multi_cell(0, 5, "Analyze each graph for time dependency. Each autocorrelation should be randomly distributed around 0 the entire time (except for first element).", 0, 'L')

        # split into 6 different graphs
        pdf.image(os.path.join(pngDirectory, "test3-1.png"), x=5, y=pdf.get_y(), w=105)
        pdf.image(os.path.join(pngDirectory, "test3-2.png"), x=100, y=pdf.get_y(), w=105)
        pdf.ln(80)
        pdf.image(os.path.join(pngDirectory, "test3-3.png"), x=5, y=pdf.get_y(), w=105)
        pdf.image(os.path.join(pngDirectory, "test3-4.png"), x=100, y=pdf.get_y(), w=105)
        pdf.ln(80)
        pdf.image(os.path.join(pngDirectory, "test3-5.png"), x=5, y=pdf.get_y(), w=105)
        pdf.image(os.path.join(pngDirectory, "test3-6.png"), x=100, y=pdf.get_y(), w=105)


    if RUNNING_MAYA:
        pdf.add_page()
        pdf.image(os.path.join(pngDirectory, "Pitch_Roll.png"), x = x_offset, y = pdf.get_y(), w = 180)

        pdf.add_page()
        pdf.image(os.path.join(pngDirectory, "Edges1.png"), x = x_offset, y = pdf.get_y(), w = 180)
        pdf.ln(y_pic_offset)
        pdf.image(os.path.join(pngDirectory, "Edges2.png"), x = x_offset, y = pdf.get_y(), w = 180)

    # output the pdf to the outputFile
    if RUNNING_MAYA:
        outputPath = os.path.join(my_path, outputFile)
        pdf.output(outputPath)
    else:
        pdf.output(outputFile)


def pdfHeader(pdf, title):
    '''
    insert a header in the pdf with title
    '''

    pdf.set_font('Arial', 'B', 14)
    # Calculate width of title and position
    w = pdf.get_string_width(title) + 6
    pdf.set_x((210 - w) / 2)
    # Colors of frame, background and text
    pdf.set_draw_color(255, 255, 255)
    pdf.set_fill_color(255, 255, 255)
    # pdf.set_text_color(220, 50, 50)
    # Thickness of frame (1 mm)
    # pdf.set_line_width(1)
    pdf.cell(w, 9, title, 1, 1, 'C', 1)
    # Line break
    pdf.ln(2)

    # return to normal font
    pdf.set_font("Arial", size=11)


def savePNGs(outputDir):
    '''
    saves all currently open plots as PNGs in outputDir and closes them
    '''

    # absolute path to current directory
    my_path = os.path.dirname(os.path.abspath(__file__))
    saveDirectory = os.path.join(my_path, outputDir)

    # get list of all figures
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]

    numPlots = 0
    # iterate over and save all plots tp saveDirectory
    for fig in figs:
        numPlots += 1

        # save and close the current figure
        fig.savefig(os.path.join(saveDirectory, "plot" + str(numPlots) + ".png"))
        # fig.savefig(saveDirectory + "plot" + str(numPlots) + ".png")

        plt.close(fig)


def openFile(outputFile):
    # open the pdf file
    subprocess.Popen([outputFile], shell=True)


def clearDir(outputDir):

    # create the output directory if it doesn't exist
    my_path = os.path.dirname(os.path.abspath(__file__))
    saveDirectory = os.path.join(my_path, outputDir)
    if not os.path.exists(saveDirectory):
        os.makedirs(saveDirectory)

    # removes all files in the output directory
    files = os.listdir(saveDirectory)
    for file in files:
        file_path = os.path.join(saveDirectory, file)
        if os.path.isfile(file_path):
            os.remove(file_path)