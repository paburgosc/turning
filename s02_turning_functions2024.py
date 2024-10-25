# -*- coding: utf-8 -*-
"""
@author: burgosp

detect onsets and offset of each pause beetwen turns

"""



from matplotlib import animation
from scipy.interpolate import interp1d
# import imufusion
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt,argrelextrema, argrelmax, find_peaks
import scipy
   
from sklearn.cluster import KMeans   
from itertools import groupby   
                     
                   
#%% onsets offsets
### VALERIO USE LENGHT OF THE SIGNAL IN MAXLAGS
### GET THE NUMBER OF LAGS
def xcorr(x, y, normed=True, detrend=False, maxlags=10):
    # Cross correlation of two signals of equal length
    # Returns the coefficients when normed=True
    # Returns inner products when normed=False
    # Usage: lags, c = xcorr(x,y,maxlags=len(x)-1)
    # Optional detrending e.g. mlab.detrend_mean

    Nx = len(x)
    if Nx != len(y):
        raise ValueError('x and y must be equal length')
    
    if detrend:
        import matplotlib.mlab as mlab
        x = mlab.detrend_mean(np.asarray(x)) # can set your preferences here
        y = mlab.detrend_mean(np.asarray(y))
    # x,y=accGb[:,0],accGb[:,1]
    c = scipy.signal.fftconvolve(x,y[::-1], mode='full')
    
    #### USE numpy here
    # c = np.correlate(x, y, mode='full')

    if normed:
        n = np.sqrt(np.dot(x, x) * np.dot(y, y)) # this is the transformation function
        c = np.true_divide(c,n)

    if maxlags is None:
        maxlags = Nx - 1

    if maxlags >= Nx or maxlags < 1:
        raise ValueError('maglags must be None or strictly '
                         'positive < %d' % Nx)

    lags = np.arange(-maxlags, maxlags + 1)
    c = c[Nx - 1 - maxlags:Nx + maxlags]
    return lags, c

def butter_lowpass_filter(data, cutoff, fs, order):

    # Filter requirements.
    # T = 5.0         # Sample Period
    # fs = 30.0       # sample rate, Hz
    # cutoff = 2      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
    nyq = 0.5 * fs  # Nyquist Frequency
    # order = 2       # sin wave can be approx represented as quadratic
    # n = int(T * fs)  # total number of samples
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def butter_highpass_filter(data, cutoff, fs, order):

    # Filter requirements.
    # T = 5.0         # Sample Period
    # fs = 30.0       # sample rate, Hz
    # cutoff = 2      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
    nyq = 0.5 * fs  # Nyquist Frequency
    # order = 2       # sin wave can be approx represented as quadratic
    # n = int(T * fs)  # total number of samples
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, data)
    return y

def phase_locking_value(theta1, theta2):
    complex_phase_diff = np.exp(complex(0,1)*(theta1 - theta2))
    plv = np.abs(np.sum(complex_phase_diff))/len(theta1)
    return plv

def periodic_corr_np(x, y):
    """Periodic correlation, implemented using np.correlate.

    x and y must be real sequences with the same length.
    """
    return np.correlate(x, np.hstack((y[1:], y)), mode='valid')

def corrcoef(x, y, deg=True, test=False):
    '''Circular correlation coefficient of two angle data(default to degree)
    Set `test=True` to perform a significance test.
    '''
    sx=x
    sy=y
    # convert = np.pi / 180.0 if deg else 1
    # sx = np.frompyfunc(np.sin, 1, 1)((x - np.mean(x, deg)) * convert)
    # sy = np.frompyfunc(np.sin, 1, 1)((y - np.mean(y, deg)) * convert)
    r = (sx * sy).sum() / np.sqrt((sx ** 2).sum() * (sy ** 2).sum())

    if test:
        l20, l02, l22 = (sx ** 2).sum(),(sy ** 2).sum(), ((sx ** 2) * (sy ** 2)).sum()
        test_stat = r * np.sqrt(l20 * l02 / l22)
        p_value = 2 * (1 - stats.norm.cdf(abs(test_stat)))
        return tuple(round(v, 7) for v in (r, test_stat, p_value))
    return round(r, 7)

def onsets3(x,t,plot):
    # x = df2["Head"].values
    # t = df2.time.values
    #test onsets
    # cutoff, fs, order = 0.1,128,4

    # x = s2.butter_highpass_filter(x, cutoff, fs, order)
    #datpbfin2 = s2.butter_lowpass_filter(datpbfin, cutoff2, fs, order)
    
    x = x-(np.mean(x[0:2]))
    half = min(x)+((max(x)-min(x))/2)
    ons3, propo = find_peaks(x,height=half, distance=200, prominence=100,threshold=0.001)
    ends, prope = find_peaks(-x,height=half, distance=200, prominence=100,threshold=-0.001)

    pre = ons3<ends[0]
    post = ons3> ends[-1]
    
    if len(np.argwhere(pre))>1:
        ons3 = ons3[(len(np.argwhere(pre))-1):]
        
    if len(np.argwhere(post))>1:
        ons3 = ons3[0:-(len(np.argwhere(post))-1)]
    
    velpre = (np.diff(x[0:ends[0]])/np.diff(t[0:ends[0]]))[0:-100]
    velpost= (np.diff(x[ends[-1]:])/np.diff(t[ends[-1]:]))[100:]
    tpre = t[0:ends[0]-1][0:-100]
    tpost = t[ends[-1]:-1][100:]
    velpre2 = np.convolve(velpre,np.ones(128),mode="same")
    velpost2 = np.convolve(velpost,np.ones(128),mode="same")
    # plt.plot(tpost,velpost)
    # plt.plot(tpost,velpost2)
    pausepre = np.abs(velpre2)<100
    pausepost = np.abs(velpost2)<100
    
    # plt.figure()
    # plt.plot(tpre,velpre2)
    # plt.plot(tpre[pausepre],velpre2[pausepre],'ro')
    # plt.figure()
    # plt.plot(tpost,velpost2)
    # plt.plot(tpost[pausepost],velpost2[pausepost],'ro')
    
    ons3pre = np.argwhere(pausepre)[-1]
    ons3post = np.argwhere(pausepost)[0]
    ons3 = np.hstack([ons3pre,ons3])
    if len(ons3post)>0:
        ons3 = np.hstack([ons3,ends[-1]+ons3post+100])
    
    ofset = np.mean(x[ons3])
    
    x = x-ofset
        
    

    if plot:
        plt.figure()
        plt.plot(t,x)
        plt.plot(t[ons3],x[ons3],"ro")
        plt.plot(t[ends],x[ends],"go")
    
    # plt.plot(np.diff(x[0:ends[0]]))    
    return ons3,ends,x
    

def onsets(x,t,plot): #numon -1 or 0, substract onsets or not
    x2 = x.copy()
    plt.plot(x)
    cutoff, fs, order = 3,128,4
    cutoff2=0.1
    x = butter_highpass_filter(x, cutoff2, fs, order)
    # x = butter_lowpass_filter(x, cutoff, fs, order)
    plt.plot(x)
    # plt.figure()
    
    numon = -1
    #siempre usar luego de obtener los datos actualizdos de conducta, mrkers y cop
    
    # dato = self.dataf1c7
    # dato = a.dataf1c7
    # x = np.array(dato.iloc[:,0]).astype("f")
    # y = np.array(dato.iloc[:,1]).astype("f")
    # z = np.array(dato.iloc[:,2]).astype("f")
    # t = np.array(dato.time).astype("f")
    
    p=argrelextrema(-x, np.greater_equal)[0]
    u=(np.nanmean(-x)+np.nanstd(-x))
    pf= -x[p] > u
    ends= p[pf]
    
    #get peaks:
    p2=argrelmax(x, order=1)[0]
    u2=np.nanmean(x)
    pf2= x[p2] > u2
    ons= p2[pf2]
    
    # p3=argrelextrema(x, np.greater_equal)[0]
    # u3=(np.nanmean(x)+np.nanstd(x))
    # pf3= x[p3] > u3
    # ons3= p3[pf3]
    
    ons3, _ = find_peaks(x,height=5, distance=200)
    # If you want more information about the peaks, you can access properties
# such as peak heights, left and right thresholds, etc.
    # properties = find_peaks(x, height=5, distance=100)
    # print(properties)
    # print("Peak heights:", properties["distance"])

    
    zero_crossings = np.where(np.diff(np.sign(x2)))[0] # could be x or x2

    
    th = 300
    # th3 = 0
    # ends = np.delete(ends, np.argwhere(np.ediff1d(ends) <= th) )
    
    
    #getting ends = toching the screen
    ends = np.delete(ends, np.argwhere(np.ediff1d(ends) <= th) + 1)
    onss = ons3
    # onss = np.delete(ons3, np.argwhere(np.ediff1d(ons3) <= th3) + 1)
    
    for ie in range(len(ends)):
        lie = -x[ends[ie]:ends[ie]+200]==np.nanmax(-x[ends[ie]:ends[ie]+200])
        ie2 = np.where(lie)[0][0]
        # print(ie2)
        ends[ie] = ends[ie] + ie2
        
    #getting onsets and endss = starting from thorax and coming back to thorax   
    onsf = np.zeros(np.shape(ends)).astype("i")

    ends2 = np.zeros(np.shape(ends)).astype("i")
    
    for i in range(len(ends)):
        if ends[i] == 0:
            continue
        alt = ons < ends[i]
        alt2 = ons > ends[i]
        onsf[i]=ons[alt][numon]
        if all(alt2==False):
            ends2[i]=x[~np.isnan(x)][-1]
        else:
            ends2[i]=ons[alt2][0]
    
    print(onsf[1:])
    
    if onsf[0]==0 and ends2[0]==0:
        onsf = onsf[1:]
        ends2 = ends2[1:]
        ends = ends[1:]

    starts = np.vstack((onsf,ends,ends2))
    
    minlim = min(min(t[onsf]),min(t[ends]))
    maxlim = max(max(t[onsf]),max(t[ends]))
    zlo1 = np.where(t[zero_crossings] < minlim) 
    zlo2 = np.where(t[zero_crossings] > maxlim)
    # zlo3 = np.arange(zlo1[0][-1],zlo2[0][0]+1)  #PB 6/1/2023
    # zlo3 = np.arange(zlo2[0][0])
    

    
    # zero_crossings = zero_crossings[zlo3]
    
    # plot = plott
    x = x2
    if plot:
        plt.figure()
        
        plt.plot(t,-x)
        plt.plot(t[ends],-x[ends],"ro")
        plt.plot(t[onsf],-x[onsf],"g^")
        plt.plot(t[zero_crossings],-x[zero_crossings],"y^")
        

        # plt.figure()
        
        # plt.plot(t,-x)
        # plt.plot(t[onss],-x[onss],"g^")
    
    # find additional peaks
    #     print(t[onss[-1]])
    #     print(t[onsf[-1]])
    # if t[onss[-1]]>t[onsf[-1]]:
    #     print(len(onsf))
    #     onsf= np.hstack([onsf,onss[-1]])
    #     print(len(onsf))
    
    return onsf,ends, zero_crossings



# signals = [0,0,0,0,1,1,0,0,0,1,1,1,0,0,0,1,0,0,1,1,1,1,0]

def count_event(signals,event_value=1):
    """Counts no of events and counts the duration of the events

    Args:
        signal : list of signals
        event_value : value of the siganl

    Returns:
        list of tuples of index and duration of events
    """
    signals = list(signals)
    event_duration = []
    index = 0
    for key,g in (groupby(signals)):
        length = len(list(g))
        if key == event_value:
            event_duration.append((index,length-1))
        index += length
    return event_duration



def onsets2(data,time,n_clusters,n_clusters2,up_down,plot):
    if up_down == 1:
        data = data
    elif up_down ==0:
        data=-data
    else:
        print('up_down should be 1 or 0')
    # data = df2.Lumbar.values
    # time = df2.time.values
    # n_clusters=8
    # n_clusters2=7
    
    kmeans = KMeans(n_clusters)
    kmeans.fit(data.reshape(-1,1))
    label_clusters = kmeans.labels_  # marca cada punto con un cluster
    center_clusters = np.squeeze(kmeans.cluster_centers_)  # centro de los clusters
    sorted_clusters = np.argsort(center_clusters) # lower to greater
    # np.where(center_clusters == center_clusters[sorted_clusters[-1]]
    signal1 = label_clusters == sorted_clusters[-1] ##SELECT HIGHER CLUSTER
    # signal1 = label_clusters == sorted_clusters[0] ##SELECT LOWER CLUSTER
    event_duration=count_event(list(signal1),1)
    event_duration = np.array(event_duration)

    if plot:
        
        plt.figure()
        plt.plot(time,signal1)
        plt.plot(time[event_duration[:,0]],signal1[event_duration[:,0]],'xg')
        plt.plot(time[np.sum(event_duration,1)],signal1[np.sum(event_duration,1)],'or')
        
        
        plt.figure()
        plt.scatter(time, data, c=label_clusters)
    
    onsoff =[] 
    onsoff2 =[] 
    for ons,off in zip(event_duration[:,0],np.sum(event_duration,1)):
        #ons= event_duration[0,0]
        #off= np.sum(event_duration,1)[0]

        short_angle= data[ons:off]
        short_data = np.diff(data[ons:off])/np.diff(time[ons:off])
        short_time = time[ons:off-1]
        # short_data = np.diff(data[0:580])/np.diff(time[0:580])
        # short_time = time[0:579]

        # n_clusters2=5
        kmeans2 = KMeans(n_clusters2)
        kmeans2.fit(short_data.reshape(-1,1))
        label_clusters2 = kmeans2.labels_
        center_clusters2 = np.squeeze(kmeans2.cluster_centers_)  # centro de los clusters
        higher_cluster2 = []
        short_angle2= short_angle[0:-1]
        for i in range(n_clusters2):
            higher_cluster2.append(np.max(short_angle2[label_clusters2==i]))
        higher_cluster2 = np.array(higher_cluster2) 
        
            
        sorted_clusters2 = np.argsort(np.abs(center_clusters2)) # minor to mayor
        sorted_clusters3 = np.argsort(-higher_cluster2) # minor to mayor
        sorting_cluster = np.vstack([sorted_clusters2,sorted_clusters3])
        if sorting_cluster[0,0] == sorting_cluster[1,0]:
            final_clust = sorting_cluster[1,0]
        else:
            final_clust = sorting_cluster[0,0]
        
        signal2 = label_clusters2 == final_clust
        event_duration2 =[]
        event_duration2=count_event(list(signal2),1)
        event_duration2b = np.array(event_duration2)
        
        
        if plot==2:
            plt.figure()
            plt.scatter(short_time, short_data, c=label_clusters2)
            
            plt.title("angular velocity")
            
            plt.figure()
            plt.scatter(short_time, short_angle2, c=label_clusters2)
            plt.scatter(short_time[signal2], short_angle2[signal2], marker='+',c='red')
            plt.title("angle")
        # try:
        #     checkonset= onsets(short_time,short_angle2,1)
        # except:
        #     print("probably no picks")
        
        event_duration2b[event_duration2b==0]=1
        offevent_duration2= np.sum(event_duration2b,1)
        
        short_a2_list=[]
        for i in range(len(event_duration2b)):
            short_a2_list.append(np.max(short_angle2[event_duration2b[i,0]:offevent_duration2[i]]))
        
        if len(event_duration2)>1:
            posclu= np.where(short_a2_list == np.max(short_a2_list))[-1][-1]
            # print(posclu)
            maxloc1 = short_angle2[event_duration2b[posclu,0]:offevent_duration2[posclu]]
            
            
            maxloc2 = short_angle2[event_duration2b[-1,0]:offevent_duration2[-1]] #ultimo
            maxloc3 = short_angle2[event_duration2b[0,0]:offevent_duration2[0]] #primero
            # on1= event_duration[0,0]+ offevent_duration2[0]#event_duration2b[0,0] + #len(maxloc1) #np.where(maxloc1==np.max(maxloc1))[-1][-1]
            # print(event_duration2)
            # print(maxloc1)
            # print(maxloc2)
            # if np.max(maxloc1) > np.max(maxloc2):
            on1= event_duration2b[posclu,0]+np.where(maxloc1==np.max(maxloc1))[-1][-1]
            # on2= on1    
                # on2= event_duration2b[-1,0] + offevent_duration2[-1]#event_duration2b[-1,0] + #len(maxloc2) #np.where(maxloc2==np.max(maxloc2))[-1][-1]
            on2= event_duration2b[-1,0] + len(maxloc2)#np.where(maxloc2==np.max(maxloc2))[-1][-1] #ultimo
            on3= event_duration2b[0,0] + np.where(maxloc3==np.max(maxloc3))[-1][-1] #primero
            # else:
            #     on2= event_duration[0,0]+np.where(maxloc1==np.max(maxloc1))[-1][-1]
                
            #     # on2= event_duration2b[-1,0] + offevent_duration2[-1]#event_duration2b[-1,0] + #len(maxloc2) #np.where(maxloc2==np.max(maxloc2))[-1][-1]
            #     on1= event_duration2b[-1,0] + np.where(maxloc2==np.max(maxloc2))[-1][-1]
        #ons= event_duration[0,0]
        #off= np.sum(event_duration,1)[0]
        else:
            maxloc1 = short_angle2[event_duration2b[0,0]:offevent_duration2[0]]
            # on1= event_duration2b[0,0] + offevent_duration2[0]#event_duration2b[0,0] + #len(maxloc1) #np.where(maxloc1==np.max(maxloc1))[-1][-1]
            # on2 = event_duration2b[0,0]
            on1= event_duration2b[0,0] + np.where(maxloc1==np.max(maxloc1))[-1][-1]#event_duration2b[0,0] + #len(maxloc1) #np.where(maxloc1==np.max(maxloc1))[-1][-1]
            on2 = event_duration2b[0,0] + len(maxloc1)
            on3 = event_duration2b[0,0] 
        onsoff.append([ons+on1,ons+on2])  
        onsoff2.append(ons+on3)
        # onsoff.append([ons+on1+1,ons+on2+1])  
    onsoffb = np.array(onsoff)
    onsoffc = onsoffb[:,0]
    onsoffc[0]= onsoffb[0,1]#onsoff2[0]
    # if abs(onsoffb[-1,0]-onsoffb[-1,1])>128:
    onsoffc[-1]= onsoff2[-1]
        
    
    return onsoffc,maxloc1,event_duration      


      
###Using event dutarion from Lumbar sensor        
def onsets2B(data,time,event_duration,n_clusters2,up_down,plot):
    if up_down == 1:
        data = data
    elif up_down ==0:
        data=-data
    else:
        print('up_down should be 1 or 0')
    # data = df2.Lumbar.values
    # time = df2.time.values
    # n_clusters=8
    # n_clusters2=7
    
    # kmeans = KMeans(n_clusters)
    # kmeans.fit(data.reshape(-1,1))
    # label_clusters = kmeans.labels_  # marca cada punto con un cluster
    # center_clusters = np.squeeze(kmeans.cluster_centers_)  # centro de los clusters
    # sorted_clusters = np.argsort(center_clusters) # lower to greater
    # # np.where(center_clusters == center_clusters[sorted_clusters[-1]]
    # signal1 = label_clusters == sorted_clusters[-1] ##SELECT HIGHER CLUSTER
    # # signal1 = label_clusters == sorted_clusters[0] ##SELECT LOWER CLUSTER
    # event_duration=count_event(list(signal1),1)
    # event_duration = np.array(event_duration)

    # if plot:
        
    #     plt.figure()
    #     plt.plot(time,signal1)
    #     plt.plot(time[event_duration[:,0]],signal1[event_duration[:,0]],'xg')
    #     plt.plot(time[np.sum(event_duration,1)],signal1[np.sum(event_duration,1)],'or')
        
        
    #     plt.figure()
    #     plt.scatter(time, data, c=label_clusters)
    
    onsoff =[] 
    onsoff2 =[] 
    idxonsoff=0
    for ons,off in zip(event_duration[:,0],np.sum(event_duration,1)):
        print("ons")
        print(idxonsoff)
        idxonsoff+=1
        #ons= event_duration[0,0]
        #off= np.sum(event_duration,1)[0]

        short_angle= data[ons:off]
        
        
        ##VALERIO FILTER
        # plt.figure()
        # plt.plot(short_angle)
        # short_angle_filt = butter_highpass_filter(np.asarray(short_angle),0.01,128,4)
        # plt.plot(short_angle_filt)
        
        short_data = np.diff(data[ons:off])/np.diff(time[ons:off])
        short_time = time[ons:off-1]
        # short_data = np.diff(data[0:580])/np.diff(time[0:580])
        # short_time = time[0:579]

        # n_clusters2=5
        kmeans2 = KMeans(n_clusters2)
        kmeans2.fit(short_data.reshape(-1,1))
        label_clusters2 = kmeans2.labels_
        center_clusters2 = np.squeeze(kmeans2.cluster_centers_)  # centro de los clusters vel
        higher_cluster2 = []
        short_angle2= short_angle[0:-1]
        for i in range(n_clusters2):
            higher_cluster2.append(np.max(short_angle2[label_clusters2==i]))
        higher_cluster2 = np.array(higher_cluster2) 
        
            
        sorted_clusters2 = np.argsort(np.abs(center_clusters2)) # minor to mayor vel
        sorted_clusters3 = np.argsort(-higher_cluster2) # minor to mayor angle
        sorting_cluster = np.vstack([sorted_clusters2,sorted_clusters3])

        if sorting_cluster[0,0] == sorting_cluster[1,0]:
            final_clust = sorting_cluster[1,0]
        else:
            final_clust = sorting_cluster[1,0]
        
        signal2 = label_clusters2 == final_clust
        event_duration2 =[]
        event_duration2=count_event(list(signal2),1)
        event_duration2b = np.array(event_duration2)

        
        if plot==2:
            plt.figure()
            plt.scatter(short_time, short_data, c=label_clusters2)
            
            plt.title("angular velocity")
            
            plt.figure()
            plt.scatter(short_time, short_angle2, c=label_clusters2)
            plt.scatter(short_time[signal2], short_angle2[signal2], marker='+',c='red')
            plt.title("angle")
        # try:
        #     checkonset= onsets(short_time,short_angle2,1)
        # except:
        #     print("probably no picks")
        
        event_duration2b[event_duration2b==0]=1
        offevent_duration2= np.sum(event_duration2b,1)
        print('event_duration2b')
        print(event_duration2b)
        short_a2_list=[]
        for i in range(len(event_duration2b)):
            short_a2_list.append(np.max(short_angle2[event_duration2b[i,0]:offevent_duration2[i]]))
        
        
        print('short_a2_list')
        print(short_a2_list)
        if len(event_duration2)>1:
            posclu= np.where(short_a2_list == np.max(short_a2_list))[-1][-1]
            print('posclu')
            print(posclu)
            maxloc1 = short_angle2[event_duration2b[posclu,0]:offevent_duration2[posclu]]
            
            
            maxloc2 = short_angle2[event_duration2b[-1,0]:offevent_duration2[-1]] #ultimo
            maxloc3 = short_angle2[event_duration2b[0,0]:offevent_duration2[0]] #primero
            # on1= event_duration[0,0]+ offevent_duration2[0]#event_duration2b[0,0] + #len(maxloc1) #np.where(maxloc1==np.max(maxloc1))[-1][-1]
            # print(event_duration2)
            # print(maxloc1)
            # print(maxloc2)
            # if np.max(maxloc1) > np.max(maxloc2):
            on1= event_duration2b[posclu,0]+np.where(maxloc1==np.max(maxloc1))[-1][-1]
            # on2= on1    
                # on2= event_duration2b[-1,0] + offevent_duration2[-1]#event_duration2b[-1,0] + #len(maxloc2) #np.where(maxloc2==np.max(maxloc2))[-1][-1]
            on2= event_duration2b[-1,0] + len(maxloc2)#np.where(maxloc2==np.max(maxloc2))[-1][-1] #ultimo
            on3= event_duration2b[0,0] + np.where(maxloc3==np.max(maxloc3))[-1][-1] #primero
            # else:
            #     on2= event_duration[0,0]+np.where(maxloc1==np.max(maxloc1))[-1][-1]
                
            #     # on2= event_duration2b[-1,0] + offevent_duration2[-1]#event_duration2b[-1,0] + #len(maxloc2) #np.where(maxloc2==np.max(maxloc2))[-1][-1]
            #     on1= event_duration2b[-1,0] + np.where(maxloc2==np.max(maxloc2))[-1][-1]
        #ons= event_duration[0,0]
        #off= np.sum(event_duration,1)[0]
        else:
            maxloc1 = short_angle2[event_duration2b[0,0]:offevent_duration2[0]]
            # on1= event_duration2b[0,0] + offevent_duration2[0]#event_duration2b[0,0] + #len(maxloc1) #np.where(maxloc1==np.max(maxloc1))[-1][-1]
            # on2 = event_duration2b[0,0]
            on1= event_duration2b[0,0] + np.where(maxloc1==np.max(maxloc1))[-1][-1]#event_duration2b[0,0] + #len(maxloc1) #np.where(maxloc1==np.max(maxloc1))[-1][-1]
            on2 = event_duration2b[0,0] + len(maxloc1)
            on3 = event_duration2b[0,0] 
        onsoff.append([ons+on1,ons+on2])  
        onsoff2.append(ons+on3)
        # onsoff.append([ons+on1+1,ons+on2+1])  

    onsoffb = np.array(onsoff)
    print('onsoffb')
    print(onsoffb)
    onsoffc = onsoffb[:,0]
    if up_down:
        onsoffc[0]= onsoffb[0,1]#onsoff2[0]  ###PB 6 7 2024
    # if abs(onsoffb[-1,0]-onsoffb[-1,1])>128:
    #     onsoffc[-1]= onsoff2[-1]      ###PB 6 7 2024
        
    
    return onsoffc,maxloc1             
    
#### delete otliers turn by angle or time, use the boxplot or distribution criteria    


