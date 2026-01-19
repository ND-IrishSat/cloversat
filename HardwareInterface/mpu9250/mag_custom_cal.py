''' 
mag_custom_cal.py
Author: Juwan Jacobe

this script calibrates to remove hard-iron and soft-iron distortion for mpu magnetometer

'''

#
# wait 5-sec for IMU to connect
import time,sys
sys.path.append("../")
t0 = time.time()
start_bool = False # if IMU start fails - stop calibration
while time.time()-t0<5:
    try: 
        from happy_sensors import mpu
        start_bool = True
        break
    except:
        continue
import numpy as np
import matplotlib.pyplot as plt

time.sleep(2) # wait for mpu to load
# 
#####################################
# Mag Calibration Functions
#####################################
#
def outlier_removal(x_ii, y_ii, z_ii):
    x_diff = np.append(0.0,np.diff(x_ii)) # looking for outliers
    stdev_amt = 5.0 # standard deviation multiplier
    x_outliers = np.where(np.abs(x_diff)>np.abs(np.mean(x_diff))+\
                          (stdev_amt*np.std(x_diff)))
    x_inliers  = np.where(np.abs(x_diff)<np.abs(np.mean(x_diff))+\
                          (stdev_amt*np.std(x_diff)))
    y_diff = np.append(0.0,np.diff(y_ii)) # looking for outliers
    y_outliers = np.where(np.abs(y_diff)>np.abs(np.mean(y_diff))+\
                          (stdev_amt*np.std(y_diff)))
    y_inliers  = np.abs(y_diff)<np.abs(np.mean(y_diff))+\
                 (stdev_amt*np.std(y_diff))

    z_diff = np.append(0.0, np.diff(z_ii)) # looking for outliers
    z_outliers = np.where(np.abs(y_diff) > np.abs(np.mean(z_diff)) + \
                          (stdev_amt*np.std(z_diff)))
    z_inliers = np.abs(z_diff) < np.abs(np.mean(z_diff)) + \
                (stdev_amt*np.std(z_diff))

    if len(x_outliers)!=0:
        x_ii[x_outliers] = np.nan # null outlier
        y_ii[x_outliers] = np.nan # null outlier
        z_ii[x_outliers] = np.nan
    if len(y_outliers)!=0:
        y_ii[y_outliers] = np.nan # null outlier
        x_ii[y_outliers] = np.nan # null outlier
        z_ii[y_outliers] = np.nan
    if len(z_outliers)!=0:
        x_ii[z_outliers] = np.nan
        y_ii[z_outliers] = np.nan
        z_ii[z_outliers] = np.nan

    return x_ii, y_ii, z_ii

def mag_cal():
    ''' Calculates calibration offsets and scaling factors

        Return:
            - offsets (3x1 list), scales (3x1 list), data (N x 3 np.ndarray)
    '''

    print("-"*50)
    print("Magnetometer Calibration")
    mpu.configure()

    # Initialize data array
    data = []

    print("Reading Starting - Move Magnetometer in Figure 8 Shape")
    print("Press Ctrl + C when done")
    while True:
        try:
            # Read magnetometer data
            sample = mpu.readMagnetometerMaster()
            data.append(sample)
            time.sleep(0.1)
        except KeyboardInterrupt:
            break
        except:
            continue

    # Clear buffer, don't use first 20 vals
    data = data[20:]
    
    # Convert to numpy array
    data = np.array(data)
    
    mx = data[:, 0]
    my = data[:, 1]
    mz = data[:, 2]

    # Hard iron calibration
    offset_mx = (np.max(mx) + np.min(mx)) / 2
    offset_my = (np.max(my) + np.min(my)) / 2
    offset_mz = (np.max(mz) + np.min(mz)) / 2

    # Remove hard iron offsets to use 
    # hard iron corrected values for soft-iron calibration
    mx_hard = mx - offset_mx * np.ones(mx.shape)
    my_hard = my - offset_my * np.ones(my.shape)
    mz_hard = mz - offset_mz * np.ones(mz.shape)

    # Soft iron calibration
    avg_delta_mx = (np.max(mx_hard) - np.min(mx_hard))/2
    avg_delta_my = (np.max(my_hard) - np.min(my_hard))/2
    avg_delta_mz = (np.max(mz_hard) - np.min(mz_hard))/2

    avg_delta = (avg_delta_mx + avg_delta_my + avg_delta_mz) / 3

    scale_mx = avg_delta / avg_delta_mx
    scale_my = avg_delta / avg_delta_my
    scale_mz = avg_delta / avg_delta_mz

    return [offset_mx, offset_my, offset_mz], [scale_mx, scale_my, scale_mz], data

def mag_cal_plot(offsets, scales, data):
    '''
    '''
    
    # Grab data columns
    mx = data[:, 0]
    my = data[:, 1]
    mz = data[:, 2]

    # Calculate corrections
    # Hard-iron corrected
    mx_hcorr = (mx - offsets[0]*np.ones(mx.shape))
    my_hcorr = (my - offsets[1]*np.ones(my.shape))
    mz_hcorr = (mz - offsets[2]*np.ones(mz.shape))

    data_hcorr = np.zeros(data.shape)
    data_hcorr[:, 0] = mx_hcorr
    data_hcorr[:, 1] = my_hcorr
    data_hcorr[:, 2] = mz_hcorr

    # Hard-iron and soft-iron corrected
    mx_hscorr = scales[0]*mx_hcorr
    my_hscorr = scales[1]*my_hcorr
    mz_hscorr = scales[2]*mz_hcorr

    data_hscorr = np.zeros(data.shape)
    data_hscorr[:, 0] = mx_hscorr
    data_hscorr[:, 1] = my_hscorr
    data_hscorr[:, 2] = mz_hscorr

    axes_pairs = ['Bx vx By', 'Bx vs Bz', 'By vs Bz']
    axes_pairs_indices = [[0, 1], [0, 2], [1, 2]]

    data_arrs = [data, data_hcorr, data_hscorr]
    plot_titles = ['Raw Magnetometer Data', 'Hard-Iron Corrected Data', 'Hard- and Soft-Iron Corrected Data']

    for i in np.arange(3):
        fig = plt.figure(i)
        ax = plt.axes()

        data_plot = data_arrs[i]
        title = plot_titles[i]

        for j, axes_pair in enumerate(axes_pairs):

            indices_plot = axes_pairs_indices[j]
            #print(data_plot[indices_plot[0]])
            #print(data_plot[indices_plot[1]])
            ax.scatter(data_plot[:, indices_plot[0]], data_plot[:, indices_plot[1]], label = axes_pair)
            ax.set_aspect('equal')
            ax.set_xlim(-100, 100)
            ax.set_ylim(-100, 100)
            ax.set_xlabel('micro-Tesla')
            ax.set_ylabel('micro-Tesla')
            ax.legend()
            ax.set_title(title)
        plt.show() 
if __name__ == '__main__':
    if not start_bool:
        print("IMU not Started - Check Wiring and I2C")
    else:
        
        mpu.calibrate()
        #
        ###################################
        # Magnetometer Calibration
        ###################################
        #
        #mcoeffs, mscales, data = mag_cal()
        #print(mcoeffs)
        #print(mscales)
        #print(data)
        #

        # Save mag calibration files to text file
        mcoeffs, mscales, data = mag_cal()
        print(mcoeffs)
        print(mscales)
        #fname=input("Enter file name to store the data")
        #fout=open(fname,"w")
        fout=open("calvals.txt","w")
        if(fout): #idk if you want it printed out like this but eh i cant test it so
            for i in range(3):
                fout.write(str(mcoeffs[i])+" "+str(mscales[i])+"\n")
            fout.close()
        #print(data)
        #
        ###################################
        # Plot with and without offsets
        ###################################
        #
        mag_cal_plot(mcoeffs, mscales, data) # plot un-calibrated and calibrated results
        #
        
