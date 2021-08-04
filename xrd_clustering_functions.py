#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 18:26:18 2021

@author: kaleighcurtis
"""

'''As of August 4 2021- Currently working on docstrings'''


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.cluster import KMeans, AgglomerativeClustering
from itertools import cycle
import tarfile

#tar_file and coordinate_list are imported files

def extract_from_tar(tar_file):
    listoftar = []
    for thing in tar_file:
        dataset = tar_file.extractfile(thing)
        nameoffile = thing.name
        #x is the position on Q, y is the intensity at that position
        x,y = np.genfromtxt(dataset, skip_header=7, delimiter=' ', unpack=True)
        listoftar.append([nameoffile,x,y])
    #Sorting tar file alphabetically according to file name 
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
    intensitylist = []
    for i in i_data:
        summation = sum(i)
        intensitylist.append(summation)
    return intensitylist

def get_coordinates(coordinate_list):
    x_coords = np.hsplit(coordinate_list,3)[1]
    y_coords = np.hsplit(coordinate_list,3)[2]
    return x_coords, y_coords

def create_point_dict(x_coords, y_coords, i_data):
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

def make_random_peaks(
    x, xmin=None, xmax=None, peak_chance=0.1, return_pristine_peaks=False):

    # select boundaries for peaks
    if xmin is None:
        xmin = np.percentile(x, 10)
    if xmax is None:
        xmax = np.percentile(x, 90)

    y = np.zeros(len(x))

    # make peak positions
    peak_pos = np.array(np.random.random(len(x)) < peak_chance)
    peak_pos[x < xmin] = False
    peak_pos[x > xmax] = False

    for peak_idex in [i for i, x in enumerate(peak_pos) if x]:
        y += gaussian(x, c=x[peak_idex], sig=0.1, amp=(1 / x[peak_idex]) ** 0.5)

    # now for any diffuse low-Q component
    y += gaussian(x, c=0, sig=3, amp=0.1)
    
    return y


def gaussian(x, c=0, sig=1, amp=None): #Used for making random peaks
    if amp is None:
        return (
            1.0
            / (np.sqrt(2.0 * np.pi) * sig)
            * np.exp(-np.power((x - c) / sig, 2.0) / 2)
        )
    else:
        return amp * np.exp(-np.power((x - c) / sig, 2.0) / 2)

#Normalizes the intensity data
def normalize_intensity(intensity_list):
    intensity_list -= intensity_list.min()
    intensity_list /= intensity_list.max()
    return intensity_list


#Collects spectra for each point in an even spread of points in a range
def down_the_line(point_dictionary, y_testing_val, x_coords, testing_spectra):
    x_range = np.unique(x_coords)
    spectra_collection_list = testing_spectra

    intensitylist = get_intensity(spectra_collection_list, x_range)
    intensitylist = normalize_intensity(np.asarray(intensitylist))
    intensitylist = np.ndarray.flatten(intensitylist)

    return spectra_collection_list, x_range, intensitylist

#Sums the intensities for each spectra and plots the summed spectra against position
def get_intensity(spectra_collection_list, x_range):
    intensitylist = []
    for i in spectra_collection_list:
        summation = sum(i)
        intensitylist.append(summation)
    plt.plot(x_range, intensitylist, color = 'r')
    plt.title('Intensities versus x range')
    plt.show()
    return intensitylist

#Finds the peaks that a relavant to you and plots them
def relevant_peaks(x_range, intensitylist, min_height, min_distance, ):
    rel_peaks = find_peaks(intensitylist, height = 0.02, distance = 5 )
    y_intensity_value = []
    x_peak_value = []
    for i in range(len(rel_peaks[0])):
        #Finds the x and y coordinates of the peaks
        indexvalue = rel_peaks[0][i]
        intensitypoint = intensitylist[indexvalue]
        xpoint = x_range[indexvalue]
        y_intensity_value.append(intensitypoint)
        x_peak_value.append(xpoint)
    
    #Plots the intensity versus x distance and the relevant peaks
    plt.scatter(x_peak_value, y_intensity_value, color = 'b')

    plt.plot(x_range, intensitylist, color='r')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Data points & peaks')
    plt.show()
    return x_peak_value, y_intensity_value, rel_peaks

#Peak radius needs to be double the delta x value from peak creation 
def _identify_peaks_scan_shifter_pos(
    x, y, num_samples=0, min_height=0.02, min_dist=5, peak_rad=8, open_new_plot=True
):
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

#gauss_curve_fit_and_cut = _identify_peaks_scan_shifter_pos(np.asarray(x_range), np.asarray(intensitylist))


def find_sample_centers( deltax, peak_chance, xmin, xmax,  peak_radius, point_dictionary, ycoordinates, xcoordinates, num_samples_guess = 0, min_height = 0.02, open_new_plot = True):
  
    if peak_radius <2* deltax:
        print('Error! Your peak radius needs to be at least 2*delta x!')
    
    listspectra, x_range, intensitylist= down_the_line(point_dictionary, ycoordinates, xcoordinates)
    x_range = np.asarray(x_range)
    intensitylist= np.asarray(intensitylist)
    print(x_range.shape)
    print(intensitylist.shape)
    _, centerlist = _identify_peaks_scan_shifter_pos(x_range, intensitylist)
    return centerlist, intensitylist



def vertical_scan(center_spectra_list, sample_centers, range_of_yvals):
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

def oneD_clustering(oneD_dict,centers, spectra_list, ylist, adjusted_y):
    model =  KMeans()
    print('Clustering')
    for i in range(len(oneD_dict)):
        print(i)
        cen = centers[i]
        
        collection_endpoint = len(spectra_list[i])
        oneD_spectra = np.asarray(oneD_dict[cen])
    
        oneD_yvals = np.asarray(adjusted_y[i])
    

        oneD_stacked = (np.hstack(oneD_spectra))
        oneD_stacked = oneD_stacked.reshape(-1,1)
        #oneD_stacked= np.rot90(oneD_stacked, k = 1)

        
        #cos_similarity = cosine_similarity(oneD_stacked)
        fitted_model = model.fit(oneD_stacked)


        labels = fitted_model.labels_
        labels_unique = np.unique(labels)
        print(len(labels))
        n_clusters = len(labels_unique)
        print('For sample', len(oneD_dict)-i, ', centered at', centers[i], 'there are', n_clusters, 'estimated regions.')

        adjusted_y_slimmed = adjusted_y[i]
        colors = cycle(['b', 'orchid', 'teal', 'maroon', 'g', 'peru', 'lightpink', 'goldenrod'])
        xvals = np.full((collection_endpoint,1), fill_value = i)
        for k, col in zip(range(n_clusters), colors):
            my_members = labels == k
            xvals = np.full((collection_endpoint, 1), fill_value = centers[i])
            print(xvals.shape)
            print(adjusted_y_slimmed.shape)
            print(my_members.shape)
            plt.plot(xvals[my_members], adjusted_y_slimmed[my_members] + i*10, 'o', color = col)
            plt.title('Number of Samples: %d' %  len(centers))        
    plt.xlabel('Sample Center')
    plt.ylabel('Vertical Range')
    plt.show()
    return oneD_spectra, oneD_yvals, oneD_stacked

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx #array[idx]

def adjust_centers(sample_centers, x_list):
    adjusted_centers = []
    for i in range(len(sample_centers)):
        my_x_cen = x_list[0][find_nearest(x_list[0], sample_centers[i])]
        adjusted_centers.append(my_x_cen)
    return sorted(adjusted_centers)

def index_adjust_centers(nearest_centers, x_values):
    #Finds the index for the points that are at the defined sample centers
    coordindexlist = []
    for j in nearest_centers:
        indices = [i for i, x in enumerate(x_values) if x == j]
        coordindexlist.append(indices)
    #sorts and stacks the indices so it is all one unnested list
    return coordindexlist

def order_sample_center_values(coordinate_index_list, xvalues, yvalues, qvalues, ivalues):
    #puts the necessary values into a nice order
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

'''need to change model_name so its actually used'''
def clustering_1D(model_name, adjusted_centers, x, y, xlist, ylist, qlist, ilist):
#model =  AgglomerativeClustering(n_clusters = 8)
    model = KMeans()

    fitted_model = model.fit(ilist)


    labels = fitted_model.labels_
    print(len(labels))
    labels_unique = np.unique(labels)
    n_clusters = len(labels_unique)
    print('For position', len(adjusted_centers),  'there are', n_clusters, 'estimated regions.')
    
    colors = cycle(['b', 'orchid', 'teal', 'maroon', 'g', 'peru', 'lightpink', 'goldenrod'])

    mymemlist = []
    ymymemlist = []
    for k, col in zip(range(n_clusters), colors):
        my_members = labels == k
 

    
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

def plot_standard_deviation(member_list, q_data):
    for i, (x_position, y_position, color) in enumerate(member_list):
        stdev = np.std(member_list[i][1], axis=0).reshape(-1,1)
        plt.plot(q_data[0][:300], stdev[:300] ,  marker = '.', color = color, alpha = 0.3)
  
    plt.title('Standard Deviation of the Sample')
    plt.xlabel('Q')
    plt.ylabel('Intensity')
    return plt.show()

def find_cluster_centers(member_list):
    #finding cluster centers when the model doesn't give it to me
    cluster_center_list = []
    for i in member_list:
        q_values = i[0]
        i_values = i[1]
        
        saved_q = q_values[0]
        saved_i = np.mean(i_values, axis = 0)

        
        cluster_center_list.append([saved_q, saved_i])
    return cluster_center_list

def plot_cluster_centers(cluster_centers, my_member_list):
    # creates waterfall plot, change multiplicative of i to change that
    for i in range(len(cluster_centers)):
        plt.plot(cluster_centers[i][0], cluster_centers[i][1] + i*2000, marker = '.',  alpha = 0.3, color = my_member_list[i][2])
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Cluster Centers')
    return plt.show() 

def onclick(event, clicklist):
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))
    x = event.xdata
    y = event.ydata
    clicklist.append([x,y])

def real_resid(target, data, absolute = False):
    if absolute:
        return abs(target-data)
    else:
        return target-data

def give_me_data(x, y, data, x_coords, y_coords):
    idx = find_nearest(x_coords, x)
    idy = find_nearest(y_coords, y)
    
    my_real_x = x_coords[idx][0]
    my_real_y = y_coords[idy][0]

    return data[my_real_y][my_real_x]

def residual_map(target, data, norm = False):
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
    for group in click_list:
        #fig, ax = plt.subplots(figsize = (10.1, 0.9), dpi = 100, constrained_layout = True)
        tx, ty = group
        sc = plot_my_residuals(tx, ty, point_dictionary, x_coords, y_coords, click_list)

    if return_click_list:
        return click_list

def get_resid_coords(residual_dict):
    y_coord_residual = np.asarray(list(residual_dict.keys()))
    x_coord_residual = np.asarray(list(residual_dict[y_coord_residual[0]].keys()))
    return x_coord_residual, y_coord_residual

def get_cluster_data_from_label(clustercenters, labels):
    cluster_label_list = []
    for i in range(len(labels)):
        clus_index = labels[i]
        internal_cluster_spectra = clustercenters[clus_index][1]
        cluster_label_list.append(internal_cluster_spectra)
    return cluster_label_list

def give_y_bounds(x_list, y_list, label_list, click_list):
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









