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


# declare global var for name out output directory to store plots in
outputDir = "plotOutput"

def saveFig(fig, fileName):
    '''
    saves fig to a png file in the outputDir directory with the name fileName
    also closes fig
    '''

    # absolute path to current directory
    my_path = os.path.dirname(os.path.abspath(__file__)) 
    saveDirectory = os.path.join(my_path, outputDir)

    fig.savefig(os.path.join(saveDirectory, fileName))

    plt.close(fig)



def savePDF(outputFile, pngDir, sim, controller=None, target=[1, 0, 0, 0], sum=0, printTests=False):
    '''
    creates a simulation report using FPDF with all PNGs found in pngDir
    Describes the different graphs and their significance

    @params:
        outputFile: name of pdf to be generated
        pngDir: name of folder where graph PNGs are found
        sim: Simulator object with sim info
        controller: PIDController object with weights info. If = None, controls info is not printed
        target: our target quaternion for this simulation
        sum: statistical tests sum
        printTests: whether statistical tests were run or not (true = print test info)
    '''

    # absolute path to current directory
    my_path = os.path.dirname(os.path.abspath(__file__)) 
    pngDirectory = os.path.join(my_path, pngDir)

    # create the PDF object
    pdf = FPDF()
    title = "Kalman-Testing Simulation Report"
    pdf.set_author("Andrew Gaylord")
    pdf.set_title(title)
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=11)

    # title and document details
    pdfHeader(pdf, title)

    magSD = (140 * 10e-6) * np.sqrt(200)
    gyroSD = 0.0035 * np.sqrt(200)
    dataText = ""

    # if we are running with real data and do not know ideal behavior, leave that page out
    if sim.ideal_known:
        introText = f"""Ideal behavior is dictated by our propogating initial state and reaction wheel info for each step through our Equations of Motion (EOMs) and the true magnetic field ({sim.B_true[0][0]}, {sim.B_true[0][1]}, {sim.B_true[0][2]} microteslas)."""
        
        pdf.multi_cell(0, 5, introText, 0, 'L')

        pdf.image(os.path.join(pngDirectory, "idealQuaternion.png"), x=10, y=pdf.get_y(), w=180)
        pdf.ln(128)
        pdf.image(os.path.join(pngDirectory, "idealVelocity.png"), x=10, y=pdf.get_y(), w=180)

        pdf.add_page()

        pdfHeader(pdf, "Data")

        dataText = f"""Simulate IMU data by adding noise to our ideal states in measurement space. For vn100, magnetometer noise = {round(magSD, 5)} and gyroscope noise = {round(gyroSD, 5)}."""

    else:

        dataText = f"""IMU sensor data in true magnetic field of ({sim.B_true[0]}, {sim.B_true[1]}, {sim.B_true[2]}. For vn100, magnetometer noise = {round(magSD, 5)} and gyroscope noise = {round(gyroSD, 5)}."""

    pdf.multi_cell(0, 5, dataText, 0, 'L')

    pdf.image(os.path.join(pngDirectory, "magData.png"), x=10, y=pdf.get_y(), w=180)
    pdf.ln(128)
    pdf.image(os.path.join(pngDirectory, "gyroData.png"), x=10, y=pdf.get_y(), w=180)

    pdf.add_page()


    pdfHeader(pdf, "Filter Results")

    filterText = f"""Kalman filter estimates our state each time step by combining noisy data and physics EOMs over {sim.n * sim.dt} seconds."""

    pdf.multi_cell(0, 5, filterText, 0, 'L')

    pdf.image(os.path.join(pngDirectory, "filteredQuaternion.png"), x=10, y=pdf.get_y(), w=180)
    pdf.ln(128)
    pdf.image(os.path.join(pngDirectory, "filteredVelocity.png"), x=10, y=pdf.get_y(), w=180)

    pdf.add_page()

    if controller != None:

        pdfHeader(pdf, "Controls")

        pwmText = f"""Our Proportional Integral Derivative controller tells our motors how fast to spin (with PMW signals) to orient us toward our target. The parameters are dependent upon our max PWM and must be finely tuned. 
        This run's target quaternion: {target}
        PID controller parameters: 
            Proportional gain: {controller.kp}
            Intergral gain: {controller.ki}
            Derivative gain: {controller.kd}"""

        pdf.multi_cell(0, 5, pwmText, 0, 'L')
            
        pdf.image(os.path.join(pngDirectory, "PWM.png"), x = 10, y = pdf.get_y(), w = 180)

        pdf.add_page()

    pdfHeader(pdf, "Reaction wheel speeds")

    pdf.image(os.path.join(pngDirectory, "ReactionSpeeds.png"), x = 10, y = pdf.get_y(), w = 180)

    if controller != None:
        # pdfHeader(pdf, "Motor Current")
        pdf.ln(128)
        pdf.image(os.path.join(pngDirectory, "current.png"), x = 10, y = pdf.get_y(), w = 180)

    pdf.add_page()
    eulerText = f"""Our filtered orientation represented by Euler Angles (counterclockwise rotation about x, y, z). Can bug out sometimes. Near 180 degrees (pi) is the same as zero. """
    pdf.multi_cell(0, 5, eulerText, 0, 'L')
    pdf.image(os.path.join(pngDirectory, "Euler.png"), x=10, y=pdf.get_y(), w=180)

    pdf.add_page()

    pdfHeader(pdf, "General Info")

    # set numpy printing option so that 0's don't have scientific notation
    np.set_printoptions(formatter={'all': lambda x: '{:<11d}'.format(int(x)) if x == 0 else "{:+.2e}".format(x)})

    infoText = f"""{sim.n} filter iterations were completed in {round(np.sum(sim.times) * 1000, 2)} milliseconds. This kalman filter took {round(np.mean(sim.times) * 1000, 2)} ms per iteration.
    
Process Noise:

{sim.R}

Measurement Noise:

{sim.Q}"""

    pdf.multi_cell(0, 5, infoText, 0, 'L')

    if printTests:

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

    # # iterate over all PNGs in the directory and add them to the pdf
    # for i, png in enumerate(os.listdir(pngDirectory)):
        
    #     # add title and description
    #     pdf.cell(200, 10, txt="Plot " + str(i+1), ln=1, align='C')
    #     pdf.cell(200, 10, txt="This is a description of the graph and its significance", ln=1, align='L')

    #     # add the PNG to the pdf
    #     pdf.image(os.path.join(pngDirectory, png), x=10, y=pdf.get_y(), w=180)

    #     # add a page break if not the last PNG
    #     if i < len(os.listdir(pngDirectory))-1:
    #         pdf.add_page()

    # output the pdf to the outputFile
    pdf.output(outputFile)


def pdfHeader(pdf, title):
    '''
    insert a header in the pdf with title
    '''

    pdf.set_font('Arial', 'B', 16)
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
    pdf.ln(10)

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
    

def oldPDF(outputFile):
    pass
    # Create the PdfPages object to which we will save the pages:
    # The with statement makes sure that the PdfPages object is closed properly at
    # the end of the block, even if an Exception occurs.
    # with PdfPages(outputFile) as pdf:

    #     fig_nums = plt.get_fignums()
    #     figs = [plt.figure(n) for n in fig_nums]
        
    #     # iterating over the numbers in list 
    #     for fig in figs:  
        
    #         # save and close the current figure
    #         fig.savefig(pdf, format='pdf') 
    #         pdf.attach_note("This is a note")
    #         plt.close(fig)
        
    #     # attach_note(self, text, positionRect=[-100, -100, 0, 0]) 
    #     # - Adds a new note to the page that will be saved next. 
    #     # The optional positionRect specifies the position on the page.

    #     # We can also set the file's metadata via the PdfPages object:
    #     d = pdf.infodict()
    #     d['Title'] = 'Kalman-Testing Output'
    #     d['Author'] = u'Jouni K. Sepp\xe4nen'
    #     d['Subject'] = 'Graphical output of Kalman-Testing simulation'
    #     d['Keywords'] = """IrishSat, UKF, Kalman Filter, CubeSat, Magnetometer, Gyroscope, Quaternion, Angular Velocity, 
    #                     Magnetic Field, Reaction Wheels, EOM, Unscented Kalman Filter, State Estimation, State Space, Measurement Space, 
    #                     Process Noise, Measurement Noise, Magnetic Field, Propagation, Simulation, Testing"""
