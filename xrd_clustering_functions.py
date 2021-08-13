#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 18:26:18 2021

@author: kaleighcurtis
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.cluster import KMeans, AgglomerativeClustering
from itertools import cycle
import tarfile

def extract_from_tar(tar_file):
    """
    Extracts diffraction data from tar file.
    
    Extracts diffraction data from tar file. Splits data into three lists 
    containing the name of the file, the q-space that this reading was taken
    in, and the intensity data from the reading. 
    
    Parameters
    ----------
    tar_file : tarfile.TarFile
        A tar file. Each line in the file gives the name, q-space data, and
        intensity data for a specific point.
        
    Returns
    -------
    tar0 : list
        List of strings, with each string being identifying information for
        the corresponding data point.
    tar1 : list
        List of np.ndarray. Each array contains the q-space data for each
        corresponding data point
    tar2 : list
        List of np.ndarray. Each array contains the intensity data for each
        corresponding data point. 
    """
    listoftar = []
    for thing in tar_file:
        dataset = tar_file.extractfile(thing)
        nameoffile = thing.name
        x,y = np.genfromtxt(dataset, skip_header=7, delimiter=' ', unpack=True)
        listoftar.append([nameoffile,x,y])
    listoftar = sorted(listoftar, key=lambda x: x[0])

    tar0 = []
    tar1 = []
    tar2 = []

    for i in range(len(listoftar)):
        obja, objb, objc = listoftar[i]
        tar0.append(obja)
        tar1.append(objb)
        tar2.append(objc)
    return tar0, tar1, tar2

def get_intensity(i_data):
    """
    Calculates the sum of all intensities.
    
    For each data point, all intensity data is summed to get a total intensity
    value for that data point. 
    
    Parameters
    ----------
    i_data : list of np.ndarray
        List of np.ndarray. Each array contains the intensity data for each
        corresponding data point. 
    
    Returns
    -------
    intensitylist : list
        List of the summed intensity values for each data point. 
    """
    intensitylist = []
    for i in i_data:
        summation = sum(i)
        intensitylist.append(summation)
    return intensitylist

def get_coordinates(coordinate_list):
    """
    Gets coordinates for data points.
    
    Takes coordinate file and splits it to get the cartesian coordinates for
    each data point.
    
    Parameters
    ----------
    coordinate_list : np.ndarray
        An array where each line contains a timestamp, the x coordinate, 
        and the y coordinate for that data point. 
        
    Returns
    -------
    x_coords : np.ndarray
        An array containing the x coordinates for every data point in order. 
    y_coords : np.ndarray
        An array containing the y coordinates for every data point in order. 
    """
    x_coords = np.hsplit(coordinate_list,3)[1]
    y_coords = np.hsplit(coordinate_list,3)[2]
    return x_coords, y_coords

def create_point_dict(x_coords, y_coords, i_data):
    """
    Maps cartesian coordinates to intensity data.
    
    Creates a dictionary that maps each x and y coordinate to its respective
    diffraction pattern. It creates a nested dictionary, with the first 
    layer being each y value, and the second layer being a dictionary of each
    x datapoint on the corresponding y plane and the intensity data at that
    coordinate.
    
    Parameters
    ----------
    x_coords : np.ndarray
        An array containing the x coordinates for every data point in order. 
    y_coords : np.ndarray
        An array containing the y coordinates for every data point in order.
    i_data : list of np.ndarray
        List of np.ndarray. Each array contains the intensity data for each
        corresponding data point. 
    
    Returns
    -------
    point_dict : dictionary
        Dictionary that maps each cartesian coordinate to its diffraction
        pattern. 
    xlist : list
        Turns x coordinates into a list with dimensions nxn, with n being
        the number of datapoints
    ylist : list
        Turns y coordinates into a list with dimensions nxn, with n being
        the number of datapoints
    """
    xlist = []
    speclist = []
    ylist = []
    point_dict = {}
    for i in y_coords:
        internallist = []
        specinternal = []
        yinternal = []
        for j in range(len(y_coords)):
            if y_coords[j] ==i:
                internallist.append(x_coords[j])
                specinternal.append(i_data[j])
                yinternal.append(y_coords[j])
        xlist.append(internallist)
        speclist.append(specinternal)
        ylist.append(yinternal)
    for i in range(len(y_coords)):
        yposition = y_coords[i][0]
        internaldict = {}

        for j in range(len(xlist[i])):
            
            xposition = xlist[i][j][0]

            internaldict[xposition] = speclist[i][j]

        point_dict[yposition] = internaldict
    return point_dict, xlist, ylist

def normalize_intensity(intensity_list):
    """
    Normalizes intensity data.
    
    Normalizes intensity data to be between 0 and 1. 
    
    Parameters
    ----------
    intensity_list : list
        List of the summed intensity values for each data point. 
        
    Returns
    -------
    intensity_list : list
        List of the summed intensity values for each data point, but 
        normalized. 
    """
    intensity_list -= intensity_list.min()
    intensity_list /= intensity_list.max()
    return intensity_list

def down_the_line(point_dictionary, y_testing_val, x_coords, i_data):
    """
    Gets x positions and intensity data. 
    
    This function compresses the process of collecting intensity data and
    getting the x range by calling the corresponding functions in this 
    function instead.
    
    Parameters
    ----------
    point_dictionary : dictionary
        Dictionary that maps each cartesian coordinate to its diffraction
        pattern.
    y_testing_val : float
        Initial y position that the rough scan is done across.
    x_coords : np.ndarray
        An array containing the x coordinates for every data point in order. 
    i_data : list of np.ndarray
        List of np.ndarray. Each array contains the intensity data for each
        corresponding data point. 
    
    Returns
    -------
    x_range : np.ndarray
        Array of all unique x coordinates
    intensitylist : list
        List of the summed intensity values for each data point. 
    """
    x_range = np.unique(x_coords)
    

    intensitylist = get_intensity(i_data)
    intensitylist = normalize_intensity(np.asarray(intensitylist))
    intensitylist = np.ndarray.flatten(intensitylist)

    return  x_range, intensitylist

def _identify_peaks_scan_shifter_pos(
    x, y, num_samples=0, min_height=0.02, min_dist=5, peak_rad=8, open_new_plot=True
):
    """
    Finds sample centers and plots sample locations. 
    
    This function first finds the sample centers by finding the maximum of the
    intensity values. It then cuts the data so that it can fit a Gaussian 
    curve to each sample center. The maximum of the Gaussian curves are
    returned as the true sample centers. 
    
    Parameters
    ----------
    x : np.ndarray
        Array of all unique x coordinates.
    y : list
        List of the summed intensity values for each data point. 
    num_samples : int
        Number of samples being tested.
    min_height : int
        Minimum height required for a peak to be identified as a local 
        maximum.
    min_dist : int
        Required horizontal distance in between peaks. 
    peak_rad : int
        Limits for the Gaussian curve fit. 
    open_new_plot : boolean
        Asks if you want to open a new plot for the plots created in this
        function.
        
    Returns
    -------
    boolean : boolean
        Returns False if you chose not to continue with the found peaks and 
        True if you did choose to continue. Returns as the boolean.
    fit_peak_cen_list : list
        List of the fitted peak centers. Returns flipped.
    """
    from scipy.signal import find_peaks
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    import numpy as np
    

    if open_new_plot:
        print("making new figure")
        plt.figure()
    else:
        print("clearing figure")
        this_fig = plt.gcf()
        this_fig.clf()
        plt.pause(0.01)

    def yn_question(q):
        return input(q).lower().strip()[0] == "y"

    #Normalizes data
    y -= y.min()
    y /= y.max()
    print("ymax is " + str(max(y)))
    print("ymin is " + str(min(y)))

    def cut_data(qt, sqt, qmin, qmax):
        #Arrays to collect the cut x and y data
        qcut = []
        sqcut = []
        for i in range(len(qt)):
            #If the x value is greater than the minimum x around a guessed 
            #peak and less than the max around the same guessed peak, keep it
            if qt[i] >= qmin and qt[i] <= qmax:
                qcut.append(qt[i])
                sqcut.append(sqt[i])
        qcut = np.array(qcut)
        sqcut = np.array(sqcut)
        return qcut, sqcut

    # initial guess of position peaks
    print("Finding peaks")
    peaks, _ = find_peaks(y, height=min_height, distance=min_dist)

    if num_samples == 0:
        print("I found " + str(len(peaks)) + " peaks.")
    elif num_samples == len(peaks):
        print("I think I found all " + str(num_samples) + " samples you expected.")
    else:
        print("WARNING: I saw " + str(len(peaks)) + " samples!")
    print("doing a thing")
    
    this_fig = plt.gcf()
    this_fig.clf()

    #Plotting intensity versus position and plotting the identified peaks
    plt.plot(x, y)
    plt.plot(x[peaks], y[peaks], "kx")
    plt.title('Dan\'s peak finder')
    plt.xlabel('Position')
    plt.ylabel('Normalized Intensity')
    plt.show()
    print("done")
    
    # Checking with you to see if you want to continue
    plt.pause(0.01)
    if not yn_question("Go on? [y/n] "):
        return False, []

    # now refine positions
    peak_cen_guess_list = x[peaks]
    peak_amp_guess_list = y[peaks]

    #create zeroes arrays to put real values in later
    fit_peak_cen_list = np.zeros(len(peaks))
    fit_peak_amp_list = np.zeros(len(peaks))
    fit_peak_bgd_list = np.zeros(len(peaks))
    fit_peak_wid_list = np.zeros(len(peaks))

    #Defining the Gaussian function to fit to
    def this_func(x, c, w, a, b):

        return a * np.exp(-((x - c) ** 2.0) / (2.0 * (w ** 2))) + b

    this_fig = plt.gcf()
    this_fig.clf()
    
    cut_x_list = []
    cut_y_list = []

    for i in range(len(peaks)):
        #Get the cut data for just one peak
        cut_x, cut_y = cut_data(
            x, y, peak_cen_guess_list[i] - peak_rad, peak_cen_guess_list[i] + peak_rad
        )        
        plt.plot(cut_x, cut_y)
        
        cut_x_list.append(cut_x)
        cut_y_list.append(cut_y)
        
        #Defining a guess, the lower limit, andthe upper limit for the curve_fit 
        this_guess = [peak_cen_guess_list[i], 1, peak_amp_guess_list[i], 0]
        low_limits = [peak_cen_guess_list[i] - peak_rad, 0.05, 0.0, 0.0]
        high_limits = [peak_cen_guess_list[i] + peak_rad, 3, 1.5, 0.5]

        popt, _ = curve_fit(
            this_func, cut_x, cut_y, p0=this_guess, bounds=(low_limits, high_limits)
        )
        
        #Plotting each identified peak
        plt.plot(cut_x, this_func(cut_x, *popt), "k--")

        #Putting data from this peak into a list
        fit_peak_amp_list[i] = popt[2]
        fit_peak_wid_list[i] = popt[1]
        fit_peak_cen_list[i] = popt[0]
        fit_peak_bgd_list[i] = popt[3]
    plt.title('Dan\'s cut data')
    plt.xlabel('Position')
    plt.ylabel('Normalized Intensity')
    plt.show()
    plt.pause(0.01)

    # finally, return this as a numpy list
    return True, fit_peak_cen_list[::-1]  # return flipped

def find_sample_centers(point_dictionary, ycoordinates, xcoordinates, i_data,  num_samples_guess = 0, min_height = 0.02, open_new_plot = True):
    """
    Finds sample centers. 
    
    Large function encompassing other functions that finds sample centers
    and plots them. 
    
    Parameters
    ---------
    point_dictionary : dictionary
        Dictionary that maps each cartesian coordinate to its diffraction
        pattern. 
    ycoordinates : np.ndarray
        An array containing the y coordinates for every data point in order.
    xcoordinates : np.ndarray
        An array containing the x coordinates for every data point in order.
    i_data : list of np.ndarray
        List of np.ndarray. Each array contains the intensity data for each
        corresponding data point. 
    num_samples_guess : int
        Number of samples being tested, or a guess of the number of samples.
    min_height : int
        Minimum height required for a peak to be identified as a local 
        maximum.
    open_new_plot : boolean
        Asks if you want to open a new plot for the plots created in this
        function.
        
    Returns
    ------
    centerlist : list
        List of the fitted peak centers. Returns flipped.
    intensitylist : list
        List of the summed intensity values for each data point. 
    """

    
    x_range, intensitylist= down_the_line(point_dictionary, ycoordinates, xcoordinates, i_data)
    x_range = np.asarray(x_range)
    intensitylist= np.asarray(intensitylist)
    print(x_range.shape)
    print(intensitylist.shape)
    _, centerlist = _identify_peaks_scan_shifter_pos(x_range, intensitylist)
    return centerlist, intensitylist

def vertical_scan(center_spectra_list, sample_centers, range_of_yvals):
    """
    Collects y data for each x sample center. 
    
    Returns a dictionary containing dictionaries. Each sub-dictionary contains
    the y-positions as keys, and the corresponding values are the diffraction
    patterns at those y-positions. These dictionaries are the values for the
    overall dictionary, and their keys are the sample centers. 
    
    Parameters
    ----------
    center_spectra_list : list
        Spectra for all y positions on each sample center.
    sample_centers : np.ndarray
        Array of calculated sample centers. 
    range_of_yvals : np.ndarray
        Array of all y coordinates in  order.
        
    Returns
    -------
    spec_dict : dictionary
        Dictionary that, for every sample center, contains the spectra for 
        every y position.
    adjusted_y : list
        List of the y values for each sample center. 
    """
    spec_dict = {}
    adjusted_y = []
    print('Scanning vertically')
    for i in range(len(sample_centers)):
        cen = sample_centers[i]
        print(cen)
        print(i)
        spec_dict[cen[0]] = center_spectra_list[i]
        
        length_y = len(center_spectra_list[i])
        actual_y = range_of_yvals[:length_y]
        adjusted_y.append(actual_y)
    print('Done with vertical scan')
    return spec_dict, adjusted_y

def clustering_1D(model, adjusted_centers, x, y, xlist, ylist, qlist, ilist, colors = None):
    """
    Clusters diffraction data.
    
    Clusters the whole diffraction dataset and returns the data points that 
    corresponds to each cluster. Also gives the label that the data at each 
    coordinate is assigned to.
    
    Parameters
    ----------
    model : sklearn.cluster
        Model from scikit-learn's clustering module, with any wanted 
        parameters included.
    adjusted_centers : np.ndarray
        Array of horizontal positions of sample centers. Must be 2D array. 
    x : np.ndarray
        Array of the horizontal, defined as 'x' here, coordinates. Includes
        all instances of this 'x' position in the dataset. 
    y : np.ndarray
        Array of the vertical, defined as 'y' here, coordinates. Includes
        all instances of this 'y' position in the dataset.  
    xlist : np.ndarray
        Array of all x coordinates in a slimmed-down dataset that only 
        includes datapoints that fall on a determined sample center. Meaning,
        this array includes all instances of each sample center.
    ylist : np.ndarray
        Array of all y coordinates in a slimmed-down dataset that only 
        includes datapoints that fall on a determined sample center. Meaning, 
        this array includes all y positions that fall on a sample center. 
     qlist : np.ndarray
        Array of all values in q-space for a slimmed-down dataset that only 
        includes datapoints that fall on a determined sample center. Meaning, 
        for each point that falls on a sample center, this array collects
        the values in q-space for which diffraction data was collected.
     ilist : np.ndarray
        Array of all intensity values for a slimmed-down dataset that only 
        includes datapoints that fall on a determined sample center. Meaning, 
        for each point that falls on a sample center, this array collects
        the intensity values along q-space for which diffraction data was 
        collected.
    colors : itertools.cycle of list of strings
        An itertools function that loops through a list of matplotlib colors. 
        Matplotlib colors are written as strings. 
        
    Returns
    -------
    mymemlist : list of np.ndarrays and string
        A list where each element contains the values in q-space, intensity
        for each value in q-space, and a string containing the color that 
        cluster was assigned to
    labels : np.ndarray
        Array of cluster labels that were assigned to each point by the 
        clustering algorithm. Array is flattened.
    """

    if colors == None:
        colors = cycle(['b', 'orchid', 'teal', 'maroon', 'g', 'peru', 'lightpink', 
                        'goldenrod', 'navy', 'thistle', 'yellow', 'rosybrown', 
                        'cadetblue', 'grey', 'turquoise', 'papayawhip', 
                        'lightpink', 'mediumslateblue', 'aquamarine', 'orange',
                        'chartreuse', 'deeppink', 'black', 'cornflowerblue'])

    
    model = model

    fitted_model = model.fit(ilist)

    labels = fitted_model.labels_

    labels_unique = np.unique(labels)
    n_clusters = len(labels_unique)
    print('For position', len(adjusted_centers),  'there are', n_clusters, 'estimated regions.')
    
    mymemlist = []
    ymymemlist = []
    
    fig, ax = plt.subplots( dpi = 300)
    for k, col in zip(range(n_clusters), colors):
        my_members = labels == k
 
        qlist = np.asarray(qlist)
        ilist = np.asarray(ilist)
        xlist = np.asarray(xlist).reshape(-1,1)
        ylist = np.asarray(ylist).reshape(-1,1)
        
        q_mymem = qlist[my_members]
        i_mymem = ilist[my_members]
        mymemlist.append([q_mymem,i_mymem, col])
        ymymemlist.append(i_mymem)

        plt.plot(xlist[my_members],ylist[my_members], '.', color = col)
        plt.title('Number of Clusters: %d' %  len(labels_unique))       
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    return mymemlist, labels

def find_nearest(array,value):
    """
    Finds nearest number from a list.
    
    Finds the nearest value in an array to an input value.
    
    Parameters
    ----------
    array : np.ndarray
        Array of values.
    value : float
        Value for which you want to find the nearest value to in the array. 
        
    Returns
    -------
    idx : int
        Index of the closest number to value in array. 
    """
    idx = (np.abs(array-value)).argmin()
    return idx #array[idx]

def adjust_centers(sample_centers, x_list):
    """
    Adjusts the sample centers to real values.
    
    Shifts the sample centers to real positions, since the maximum of the 
    Gaussian fit may not be a value for which data was collected. 
    
    Parameters
    ----------
    sample_centers : np.ndarray
        Array of sample centers from the Gaussian fit. 
    x_list : list
        X coordinates in a list with dimensions nxn, with n being
        the number of datapoints.
        
    Returns
    -------
    adjusted_centers : np.ndarray
        Array of sample centers adjusted to be on the nearest x positions.
        Returns sorted. 
    """
    adjusted_centers = []
    for i in range(len(sample_centers)):
        my_x_cen = x_list[0][find_nearest(x_list[0], sample_centers[i])]
        adjusted_centers.append(my_x_cen)
    return sorted(adjusted_centers)

def index_adjust_centers(nearest_centers, x_values):
    """
    Finds the index of the adjusted centers. 
    
    Returns the index of the adjusted center value in the list of x positions. 
    
    Parameters
    ---------
    nearest_centers : np.ndarray
        Sorted array of sample centers adjusted to be on the nearest x 
        positions.
    x_values : list
        X coordinates in a list with dimensions nxn, with n being
        the number of datapoints.
    
    Returns
    -------
    coordindexlist : list
        List of indexes for each sample center.
    """
    coordindexlist = []
    for j in nearest_centers:
        indices = [i for i, x in enumerate(x_values) if x == j]
        coordindexlist.append(indices)
    return coordindexlist

def order_sample_center_values(coordinate_index_list, xvalues, yvalues, qvalues, ivalues):
    """
    Reshapes data on sample centers.
    
    Reshapes all x positions, y positions, q-space, and intensity values for
    datapoints on a sample center.
    
    Parameters
    ---------
    coordinate_index_list : list
        List of indexes for each sample center.
    xvalues : np.ndarray
        An array containing the x coordinates for every data point in order. 
    yvalues : np.ndarray
        An array containing the y coordinates for every data point in order. 
    qvalues : list
        List of q-space values for each data point.
    ivalues : list
        List of all intensity data for each data point.
    
    Returns
    -------
    chop_x : list
        List of all x positions for each data point on a sample center.
    chop_y : list
        List of all y positions for each data point on a sample center.
    chop_q : list
        List of q-space values for each data point on a sample center.
    chop_i: list
        List of all intensity data for each data point on a sample center. 
    """
    chop_q = []
    chop_i = []
    chop_x = []
    chop_y = []
    
    idict = {}
    for i in coordinate_index_list:
        sublist_q = []
        sublist_i = []
        sublist_x = []
        sublist_y = []
        
        subdict_i = {}
        for j in i:
            
            index_no = j
            sub_q = qvalues[index_no]
            sub_i = ivalues[index_no]
            sub_x = xvalues[index_no]
            sub_y = yvalues[index_no]
        
            sublist_q.append(sub_q)
            sublist_i.append(sub_i)
            sublist_x.append(sub_x)
            sublist_y.append(sub_y)
        
            subdict_i[sub_y[0]] = sub_i
        chop_q.append(sublist_q)
        chop_i.append(sublist_i)
        chop_x.append(sublist_x)
        chop_y.append(sublist_y)
    
        idict[sublist_x[0][0]] = subdict_i
    return chop_x, chop_y, chop_q, chop_i

def stacked_sample_center_values(chopped_x, chopped_y, chopped_q, chopped_i):
    """
    Stacks smaller data set for points on sample center. 
    
    Stacks chopped data for data points on sample centers in a consistent
    format. 
    
    Parameters
    ----------
    chopped_x : list
        List of all x positions for each data point on a sample center.
    chopped_y : list
        List of all y positions for each data point on a sample center.
    chopped_q : list
        List of q-space values for each data point on a sample center.
    chopped_i: list
        List of all intensity data for each data point on a sample center. 
        
    Returns
    -------
    large_x : list
        Unnested list of all x positions for each data point on a sample
        center.
    large_y : list
        Unnested list of all y positions for each data point on a sample 
        center.
    large_q : list
        Unnested list of q-space values for each data point on a sample 
        center.
    large_i : list
        Unnested list of all intensity data for each data point on a sample 
        center. 
    """
    large_q = []
    large_x = []
    large_y = []
    large_i = []
    for line in chopped_q:
        for value in line:
            large_q.append(value)
    for line in chopped_x:
        for value in line:
            large_x.append(value)
    for line in chopped_y:
        for value in line:
            large_y.append(value)
    for line in chopped_i:
        for value in line:
            large_i.append(value)
        
    large_x = np.asarray(large_x)
    large_y = np.asarray(large_y)
    large_q = np.asarray(large_q)
    large_i = np.asarray(large_i)
    
    return large_x, large_y, large_q, large_i

def plot_standard_deviation(member_list, q_data):
    """
    Plots standard deviation for each cluster. 
    
    Plots the standard deviation for each spectra in their corresponding 
    clusters. 
    
    Parameters
    ---------
    member_list : list of np.ndarrays and string
        A list where each element contains the values in q-space, intensity
        for each value in q-space, and color associated with that cluster.
    q_data : list
        List of q-range data.
        
    Returns
    -------
    plt.show : matplotlib object
        Returns the plot of standard deviation at return. 
    """
    for i, (x_position, y_position, color) in enumerate(member_list):
        stdev = np.std(member_list[i][1], axis=0).reshape(-1,1)
        plt.plot(q_data[0], stdev,  marker = '.', color = color, alpha = 0.3)
  
    plt.title('Standard Deviation of the Sample')
    plt.xlabel('Q')
    plt.ylabel('Intensity')
    return plt.show()

def find_cluster_centers(member_list):
    """
    Finds cluster centers. 
    
    When a model does not return the cluster centers for each cluster, this
    function can return those cluster centers. 
    
    Parameters
    ----------
    member_list : list
        A list where each element contains the values in q-space, intensity
        for each value in q-space, and color associated with that cluster.
    
    Returns
    -------
    cluster_center_list : list
        List of cluster centers for each cluster. 
    """
    cluster_center_list = []
    for i in member_list:
        q_values = i[0]
        i_values = i[1]
        
        saved_q = q_values[0]
        saved_i = np.mean(i_values, axis = 0)

        
        cluster_center_list.append([saved_q, saved_i])
    return cluster_center_list

def plot_cluster_centers(cluster_centers, my_member_list):
    """
    Plots cluster centers.
    
    Plots the cluster center for each found cluster. 
    
    Parameters
    ----------
    cluster_centers : list
        List of cluster centers for each cluster. 
    my_member_list : list
        A list where each element contains the values in q-space, intensity
        for each value in q-space, and color associated with that cluster.
    
    Returns
    -------
    plt.show() : matplotlib object
        Returns the plot of cluster centers. 
    """
    # creates waterfall plot, change multiplicative of i to change that
    for i in range(len(cluster_centers)):
        plt.plot(cluster_centers[i][0], cluster_centers[i][1] + i*2000, marker = '.',  alpha = 0.3, color = my_member_list[i][2])
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Cluster Centers')
    return plt.show() 

def onclick(event, clicklist):
    """
    Collects the coordinates for each click on a plot.
    
    Collects the data coordinates for each click on an interactive matplotlib
    plot.
    
    Parameters
    ----------
    event : event object
        An event that occurs when a plot is clicked. 
    clicklist : list
        An empty list.
    Returns
    -------
    Nothing. 
    """
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))
    x = event.xdata
    y = event.ydata
    clicklist.append([x,y])

def real_resid(target, data, absolute = False):
    """
    Calculates and returns residual.
    
    Calculates the residual for the entire dataset. Can also return the 
    absolute value of each residual.
    
    Parameters
    ----------
    target : np.ndarray
        Array containing the data of the target. 
    data : np.ndarray
        Array containing the data for the entire dataset. 
    absolute : boolean
        Boolean to determine if the absolute value of the residual should be 
        returned.
    
    Returns
    -------
    abs(target-data) : np.ndarray
        Returns the absolute value of a residual.
    target-data : np.ndarray
            Returns residual.
    """
    if absolute:
        return abs(target-data)
    else:
        return target-data

def give_me_data(x, y, data, x_coords, y_coords):
    """
    Returns data at a point.
    
    Finds the nearest datapoints to an input coordinate and returns the 
    data at that point. 

    Parameters
    ----------
    x : float
        Horizontal position of datapoint being looked at. 
    y : float
        Vertical position of datapoint being looked at. 
    data : dictionary
        Dictionary that maps each cartesian coordinate to its diffraction
        pattern. 
    x_coords : np.ndarray
        An array containing the x coordinates for every data point in order. 
    y_coords : np.ndarray
        An array containing the y coordinates for every data point in order. 
        
    Returns
    -------
    data[my_real_y][my_real_x] : np.ndarray
        Returns the diffraction data for the desired point. 
    """
    idx = find_nearest(x_coords, x)
    idy = find_nearest(y_coords, y)
    
    my_real_x = x_coords[idx][0]
    my_real_y = y_coords[idy][0]

    return data[my_real_y][my_real_x]

def residual_map(target, data, norm = False):
    """
    Calculates residuals. 
    
    Calculates the residuals of a dataset compared to a target. 
   
    Parameters
    ----------
    target : np.ndarray
        Array containing the data of the target. 
    data : np.ndarray
        Array containing the data for the entire dataset. 
    norm : boolean
        Boolean to determine if the absolute value of the residual should be 
        returned.
        
    Return
    ------
    residual_list_map : list
        List of each residual in order.
    resid_x : list
        List of each x coordinate corresponding to the residuals in 
        residual_list_map
    resid_y : list
        List of each y coordinate corresponding to the residuals in 
        residual_list_map
    """
    residual_list_map = []
    resid_x = []
    resid_y = []
    for line in data:
        xvalue = line

        for value in data[line]:
            yvalue = value
            this_res = sum(real_resid(target, data[line][value], absolute = True))
            residual_list_map.append(this_res)
            resid_x.append(xvalue)
            resid_y.append(yvalue)
    if norm:
        residual_list_map = (residual_list_map - min(residual_list_map))/ (max(residual_list_map)- min(residual_list_map))
    return residual_list_map, resid_x, resid_y

def plot_my_residuals(tx, ty, data, xcoords, ycoords, clicklist):
    """
    Plots residuals.
    
    Plots the residuals of a dataset compared to a target. 
    
    Parameters
    ----------
    tx : float
        Horizontal position of datapoint being looked at. 
    ty : float
        Vertical position of datapoint being looked at. 
    data : dictionary
        Dictionary that maps each cartesian coordinate to its diffraction
        pattern.
    xcoords : np.ndarray
        An array containing the x coordinates for every data point in order. 
    ycoords : np.ndarray
        An array containing the y coordinates for every data point in order. 
    clicklist : list
        List containing the data coordinates for each point the user clicked
        on the interactive plot.
        
    Returns
    -------
    sc : matplotlib object
        Returns a plot of the residuals.
    """
    
    fig, ax = plt.subplots(figsize = (10.1, 0.9), dpi = 100, constrained_layout = True)
    my_target = give_me_data(tx, ty, data, xcoords, ycoords)
    resid_map, resid_x, resid_y = residual_map(my_target, data, norm = True)
    
    
    sc = ax.scatter(resid_y, resid_x, cmap = 'viridis', c = resid_map, marker = 's', alpha = 0.9, s = 5)
    ax.scatter(tx, ty, color = 'k', s = 30, marker = 'x')
    fig.colorbar(sc)
    plt.show()
    cid = fig.canvas.mpl_connect('button_press_event', onclick(clicklist))
    return sc   
    
def plot_clicks(click_list, point_dictionary, x_coords, y_coords, return_click_list = False):
    """
    Returns residuals for selected points. 
    
    Always returns a residual plot where the target's coordinates are where 
    the user clicked. Can also return where the user clicked. 
    
    Parameters
    ----------
    click_list : list
        List containing the data coordinates for each point the user clicked
        on the interactive plot.
    point_dictionary : dictionary
        Dictionary that maps each cartesian coordinate to its diffraction
        pattern.  
    x_coords : np.ndarray
        An array containing the x coordinates for every data point in order. 
    y_coords : np.ndarray
        An array containing the y coordinates for every data point in order.    
    return_click_list : boolean
        Returns the list of points clicked if true. 
        
    Returns
    ------
    click_list : list
        List containing the data coordinates for each point the user clicked
        on the interactive plot.    
    """
    
    
    
    for group in click_list:
        #fig, ax = plt.subplots(figsize = (10.1, 0.9), dpi = 100, constrained_layout = True)
        tx, ty = group
        sc = plot_my_residuals(tx, ty, point_dictionary, x_coords, y_coords, click_list)

    if return_click_list:
        return click_list

def get_resid_coords(residual_dict):
    """
    Gets coordinates for residual.
    
    Gets the coordinates for each residual point.
    
    Parameters
    ----------
    residual_dict : dictionary
        Dictionary that maps each cartesian coordinate to its residual.
        
    Returns
    -------
    x_coord_residual : np.ndarray
        Array of x coordinates.
    y_coord_residual : np.ndarray
        Array of y coordinates.
    """
    y_coord_residual = np.asarray(list(residual_dict.keys()))
    x_coord_residual = np.asarray(list(residual_dict[y_coord_residual[0]].keys()))
    return x_coord_residual, y_coord_residual

def get_cluster_data_from_label(clustercenters, labels):
    """
    Makes list of cluster data by label order. 
    
    Makes a list of cluster centers in the same order as their corresponding 
    labels. 
    
    Parameters
    ----------
    clustercenters : list
        List of cluster centers for each cluster. Same length as the number
        of clusters. 
    labels : np.ndarray
        Array of cluster labels that were assigned to each point by the 
        clustering algorithm. Array is flattened.
    
    Returns
    -------
    cluster_label_list : list
        List of the cluster centers in order of label. 
    """
    cluster_label_list = []
    for i in range(len(labels)):
        clus_index = labels[i]
        internal_cluster_spectra = clustercenters[clus_index][1]
        cluster_label_list.append(internal_cluster_spectra)
    return cluster_label_list

def give_y_bounds(x_list, y_list, label_list, click_list):
    """
    Returns vertical bounds for a cluster. 
    
    Returns the vertical bounds for the cluster that the user clicked on. 
    
    Parameters
    ----------
    x_list : list
        Unnested list of all x positions for each data point on a sample
        center.
    y_list : list
        Unnested list of all y positions for each data point on a sample
        center.
    label_list : np.ndarray
        Array of cluster labels that were assigned to each point by the 
        clustering algorithm. Array is flattened.
    click_list : list
        List containing the data coordinates for each point the user clicked
        on the interactive plot.
        
    Returns
    -------
    bound_list : list
        List of vertical bounds for each sample the user clicked. 
    """
    xylabel_list = np.concatenate((np.asarray(x_list), np.asarray(y_list), np.asarray(label_list).reshape(-1,1)), axis = 1)
    
    selected_x_coords = []
    selected_y_coords = []
    
    selected_labels = []
    selected_index = []
    
    for group in click_list:
        x0, y0 = group
        x = x_list[find_nearest(x_list, x0)][0]
        y = y_list[find_nearest(y_list, y0)][0]
        
        selected_x_coords.append(x)
        selected_y_coords.append(y)
        
        for i, j in enumerate(xylabel_list):
            if j[0] ==x and j[1] ==y:
                print('Cluster:', label_list[i])
                selected_labels.append(label_list[i])
                selected_index.append(i)
    selected_combined_list = np.concatenate((np.asarray(selected_x_coords).reshape(-1,1), np.asarray(selected_y_coords).reshape(-1,1), np.asarray(selected_labels).reshape(-1,1), np.asarray(selected_index).reshape(-1,1)), axis = 1)

    sublist2 = []
    
    for index, value in enumerate(selected_x_coords):
        sublist1 = []
        for i, j in enumerate(xylabel_list):
            if j[0] == value:
                sublist1.append(j)
        sublist2.append(sublist1)
    external_y_save = []
    
    for index, points in enumerate(sublist2):
        internal_y_save = []
        cluster_group = selected_combined_list[index][2]
        
        for this_point, next_point in zip(points, points[1:]):
            if this_point[2] == next_point[2] == cluster_group:
                internal_y_save.append(this_point)
        external_y_save.append(internal_y_save)
    
    bound_list = []
    for index, thing in enumerate(external_y_save):
        print('The y bounds for cluster', selected_combined_list[index][2], 'at x=', selected_combined_list[index][0], 'are', thing[0][1], 'to', thing[-1][1])
        bound_list.append([thing[0][1], thing[-1][1]])
    return bound_list









