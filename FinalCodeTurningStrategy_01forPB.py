# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 13:16:02 2024


1. use onsets
2. use area

1. check angles and use scipy detrend
2. compare videos and signals, PB
3. model for video*(skeleton), MA
4. check CCorr (VA)


validate time onset with giroscope
validate the angles with video(2d or 3d)
validate angle with known positions(45,90,180)
verification with mecanical system


Manuscript



@author: burgosp
"""
#%% libraries
import s01_read_h5_func2024 as s1
import s02_turning_functions2024 as s2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks



#%% onsets & enbloc all effectors all subjects option 1


clust1=7
clust2=8 
fofig = "./figures5/FigOnsets_"

files = pd.read_excel('data/files3.xlsx')
# files = pd.read_excel('data/files2.xlsx')
for col in ["folder", "filename"]:
    files[col] = files[col].replace("\s+", " ", regex=True).str.strip()


files2 = files[(files.good==1)&(files.onsets==1)]
files2 = files2.reset_index()
# files2 = files.copy()
files2 = files2.loc[0:,:]## ALL or some participants

onsets_all_R = []
enbloc_all_R= []
onsets_all_L = []
enbloc_all_L= []
turninfo_all = []
cc_all_R = []
cc_all_L = []

        
turninfo_all.append(["participant",
                     "number_of_turns",
                     "duration_R(mean)",
                     "duration_L(mean)",
                     "angle_R(mean)",
                     "angle_L(mean)"])

errorsfi = []
errorsfi2 = []
userinput = []
for fo,fi,pa,off12,left,corre in zip(files2.folder,files2.filename,files2.participant,files2.off1_off2,files2.left,files2.correction):
    filename = fo+fi
    
    off12b = off12.split("_") 
    off1= off12b[0]
    off2 = off12b[1] 
    if fo == "./data/Longitudinal2/":
        turnfile = 0
        diag = 0
    else: # it is possible to improve with a seconf offset
        turnfile = 1
        diag =0
    
    if pa in ['OHSU_Turn_105','OHSU_Turn_137','OHSU_Turn_140','OHSU_Turn_516','OHSU_Turn_525','OHSU_Turn_602']:
        diag =1
        
    if corre == '7_8':
        clust1=7
        clust2=8
    else:
        clust1=5
        clust2=10
        
    
    
     
    if 'Long_003' in pa:
        clust1=3
    #     clust2=8
    # else:
    #     clust1=5
    #     clust2=10
    # print(filename)
    #get acc gyr angles
    # trypb =1
    # if trypb:
    try:
        data,data2,timetotal,t,tan = s1.s01b(filename,4,0,int(off1),int(off2),turnfile)
        
        df1 = pd.concat([pd.DataFrame(data[x]) for x in data], keys=data.keys(), axis=1)
        df1.columns = ['{}_{}'.format(x[0], x[1]) for x in df1.columns]
        df1.to_excel(filename+"_imu.xlsx")
        
        df2 = pd.concat([pd.DataFrame(data2[x]) for x in data2], keys=data2.keys(), axis=1)
        df2.columns = ['{}_{}'.format(x[0], x[1]) for x in df2.columns]
        df2["time"]=tan
        if np.shape(df2)[1]>8:
            df2 = df2[['Right Foot_0', 'Left Foot_0',
                   'Sternum_0', 'Head_0', 'Right Wrist_0', 'Lumbar_0', 'Left Wrist_0',
                   'time']]
        df2.columns = ['Right_Wrist', 'Left_Wrist', 'Sternum', 'Head', 'Lumbar',
               'Right_Foot', 'Left_Foot', 'time']
        
        if diag:
            df2.Lumbar = s1.diagonal_correction(df2.Lumbar, 2, 15, 128)
            df2.Sternum  = s1.diagonal_correction(df2.Sternum , 2, 15, 128)
            df2.Head = s1.diagonal_correction(df2.Head, 2, 15, 128)
            
        df2.Sternum = df2.Sternum - (df2.Sternum.mean()-df2.Lumbar.mean())
            
        df2.Head = df2.Head - (df2.Head.mean()-df2.Lumbar.mean())
        
        df2.to_excel(filename+"_angles.xlsx")
        
        
        # correction diagonal
        
        # plt.figure()
        # plt.plot(df2.Lumbar)
        # #detrend
        # import matplotlib.mlab as mlab
        
        # df2.Lumbar = mlab.detrend_mean(np.asarray(df2.Lumbar))
        # df2.Sternum = mlab.detrend_mean(np.asarray(df2.Sternum))
        # df2.Head = mlab.detrend_mean(np.asarray(df2.Head))
        # plt.plot(df2.Lumbar)
        # plt.legend(["pre","post"])
        
        # butter_highpass_filter(data, cutoff, fs, order)
        
        #filtering
        # plt.figure()
        # plt.plot(df2.Lumbar)
        # #detrend
                
        # df2.Lumbar = s2.butter_highpass_filter(np.asarray(df2.Lumbar),0.01,128,4)
        # df2.Sternum = s2.butter_highpass_filter(np.asarray(df2.Sternum),0.01,128,4)
        # df2.Head = s2.butter_highpass_filter(np.asarray(df2.Head),0.01,128,4)
        # plt.plot(df2.Lumbar)
        # plt.legend(["pre","post"])        
        
        #detect onset offsets
        plot_a= 2  # 2 show clusters
        # clust1 = 3
        # onoff = s2.onsets(df2.Lumbar.values, df2.time.values,1)
        if left:
            onoff2,m1,clust_indexes1 =s2.onsets2(-df2.Lumbar.values, df2.time.values,clust1,clust2,1,plot_a) # 8 and 7 cluster the best approach
            onoff3,m2,clust_indexes2 =s2.onsets2(-df2.Lumbar.values, df2.time.values,clust1,clust2,0,plot_a)
            onoff4,onoff5,xval =s2.onsets3(-df2.Lumbar.values, df2.time.values,plot_a)        
        else:
            onoff2,m1,clust_indexes1 =s2.onsets2(df2.Lumbar.values, df2.time.values,clust1,clust2,1,plot_a) # 8 and 7 cluster the best approach
            onoff3,m2,clust_indexes2 =s2.onsets2(df2.Lumbar.values, df2.time.values,clust1,clust2,0,plot_a)
            onoff4,onoff5,xval =s2.onsets3(df2.Lumbar.values, df2.time.values,plot_a)
        # onoff2,onoff3,zero_crossings = s2.onsets(df2.Lumbar.values, df2.time.values,plot_a)
        
        if onoff2[0]-onoff4[0]>200:
            onoff2 = np.hstack([onoff4[0],onoff2])
        else:
            onoff2[0] = onoff4[0]

        if onoff5[-1]-onoff3[-1]>200:
            onoff2 = np.hstack([onoff3,onoff5[-1]])
        else:
            onoff3[-1] = onoff5[-1]
            
        # for oo2 in onoff2:            
        #     oo2dist = np.abs(onoff4-oo2)
        #     for ood2 in range(len(oo2dist)):
        #         if oo2dist[ood2] <200:
        #             onoff2[ood2] = onoff4[ood2]
                    
                    
        
        plt.figure()
        plt.plot(df2.time.values,df2.Lumbar.values)
        plt.plot(df2.time.values[onoff2],df2.Lumbar.values[onoff2],'ro')
        plt.plot(df2.time.values[onoff3],df2.Lumbar.values[onoff3],'go')
        plt.title(pa)
        # # plt.show()
        plt.savefig(fofig+pa+".png")
        # # plt.show()
        # general stats
        
        plt.figure()
        plt.plot(np.diff(df2.Lumbar.values))
        
        if len(onoff2)>len(onoff3):
            number_of_turns = len(onoff2) - 1 + len(onoff3)
            turn_durationsR =  df2.time.values[onoff2[0:-1]] - df2.time.values[onoff3]
            turn_durationsL =  df2.time.values[onoff3] - df2.time.values[onoff2[1:]]
            turn_anglesR =  df2.Lumbar.values[onoff2[0:-1]] - df2.Lumbar.values[onoff3]
            turn_anglesL =  df2.Lumbar.values[onoff3] - df2.Lumbar.values[onoff2[1:]]
        elif len(onoff2)==len(onoff3):
            number_of_turns = len(onoff2) - 1 + len(onoff3)
            turn_durationsR =  df2.time.values[onoff2] - df2.time.values[onoff3]
            turn_durationsL =  df2.time.values[onoff3[0:-1]] - df2.time.values[onoff2[1:]]
            turn_anglesR =  df2.Lumbar.values[onoff2] - df2.Lumbar.values[onoff3]
            turn_anglesL =  df2.Lumbar.values[onoff3[0:-1]] - df2.Lumbar.values[onoff2[1:]]
            
            
        
        print("duration R(mean) = "+ str(np.mean(turn_durationsR)))
        print("duration L(mean) = "+ str(np.mean(turn_durationsL)))
        print("angle R(mean) = "+ str(np.mean(turn_anglesR)))
        print("angle L(mean) = "+ str(np.mean(turn_anglesL)))        
        turninfo_all.append([pa,
                             number_of_turns,
                             np.mean(turn_durationsR),
                             np.mean(turn_durationsL),
                             np.mean(turn_anglesR),
                             np.mean(turn_anglesL)])
    except:
        print("error part 1 "+pa)
        errorsfi.append(pa)
        continue
        
       
# continue next part 2
    # if trypb:
    try:

        # check = input("how it is look like ")


        
        
        
        
        

    # except:
    #     errorsfi.append(pa)
    #     continue
        
        # onsets all effectors
        
        onsetsR = pd.DataFrame()
        onsetsL = pd.DataFrame()
        enblocR = pd.DataFrame()
        enblocL= pd.DataFrame()
        plt.figure()
        # for s in [ 'Sternum'] :
        for s in ['Head', 'Sternum', 'Lumbar'] :
            print(s)
            plotb = 0
            # s2.onsets(df2[s].values, df2.time.values,1)
            
            if left:
                onoff2,m1 =s2.onsets2B(-df2[s].values, df2.time.values,clust_indexes1,clust2,1,plotb) # 8 and 7 cluster the best approach
                onoff3,m2 =s2.onsets2B(-df2[s].values, df2.time.values,clust_indexes2,clust2,0,plotb) # clu,clu2,updown,plot
                # onoff2,onoff3,zero_crossings = s2.onsets(df2[s].values, df2.time.values,plotb)
            else:
                onoff2,m1 =s2.onsets2B(df2[s].values, df2.time.values,clust_indexes1,clust2,1,plotb) # 8 and 7 cluster the best approach
                onoff3,m2 =s2.onsets2B(df2[s].values, df2.time.values,clust_indexes2,clust2,0,plotb) # clu,clu2,updown,plot
                # onoff2,onoff3,zero_crossings = s2.onsets(df2[s].values, df2.time.values,plotb)
            onsetsR[s]= df2.time.values[onoff2]
            onsetsL[s]= df2.time.values[onoff3]
            
            plt.figure()
            plt.plot(df2.time.values,df2[s].values)
            plt.plot(df2.time.values[onoff2],df2[s].values[onoff2],'ro')
            plt.plot(df2.time.values[onoff3],df2[s].values[onoff3],'go')
            plt.xlabel('time')
            plt.ylabel(pa)
            plt.title(s)
            
           
        plt.legend(['Head','HR','HL', 
                    'Sternum', 'SR','SL', 
                    'Lumbar','LR','LL'
                    ])
        
        plt.savefig("./figures2/Fig01_"+pa+".png")
        
        
        plt.show()
        
        
        # good = input("good or bad detection, enter 0 if it is bad, 1 if the first is bad, and R or Lif it is the last one is wrong to the Right, or Left  : ")
        good ='1'
        userinput.append(good + " " + fi)
        if good =='0':
            errorsfi.append('bad peak detection '+ pa)
            continue
        
        
        plt.figure()
        n = 2
        oRmin = pd.concat([onsetsR.min(axis=1)] * (n+1), axis=1, ignore_index=True)
        plt.pcolor(onsetsR.values-oRmin.values,vmin=0,vmax=0.5)
        plt.xticks([0.5,1.5,2.5],["Head","Sternum","Lumbar"])
        plt.ylabel("Turns")
        plt.xlabel("Onset Preference")
        plt.title('R turns '+pa)
        plt.colorbar()
        plt.savefig("./figures2/Fig02_"+pa+".png")
        
        onsets_all_R.append((onsetsR.values-oRmin.values))
        
        
        plt.figure()
        n = 2
        oLmin = pd.concat([onsetsL.min(axis=1)] * (n+1), axis=1, ignore_index=True)
        plt.pcolor(onsetsL.values-oLmin.values,vmin=0,vmax=0.5)
        plt.xticks([0.5,1.5,2.5],["Head","Sternum","Lumbar"])
        plt.ylabel("Turns")
        plt.xlabel("Onset Preference")
        plt.title('L turns '+pa)
        plt.colorbar()
        plt.savefig("./figures2/Fig03_"+pa+".png")
        
        onsets_all_L.append((onsetsL.values-oLmin.values))
                       
        # enbloc all effectors
        
        areaR = pd.DataFrame()
        areaR['index2']=np.arange(0,len(onoff3))
        areaR['Sub_ID']=pa
        areaL = pd.DataFrame()
        areaL['index2']=np.arange(0,len(onoff3))
        areaL['Sub_ID']=pa
        areaRb = pd.DataFrame()
        areaRb['index2']=np.arange(0,len(onoff3))
        areaRb['Sub_ID']=pa
        areaLb = pd.DataFrame()
        areaLb['index2']=np.arange(0,len(onoff3))
        areaLb['Sub_ID']=pa
        
        crossccR = pd.DataFrame()
        crossccR['index2']=np.arange(0,len(onoff3))
        crossccR['Sub_ID']=pa
        crossccL = pd.DataFrame()
        crossccL['index2']=np.arange(0,len(onoff3))
        crossccL['Sub_ID']=pa
        
      
        
        for se1 in ['Head', 'Lumbar'] :
            for se2 in ['Head', 'Sternum', 'Lumbar'] :
                areaR[se1+'_'+se2] = np.nan #arange(0,len(onoff3))
                areaRb[se1+'_'+se2] = np.nan #arange(0,len(onoff3))
                areaL[se1+'_'+se2] = np.nan #arange(0,len(onoff3)) 
                areaLb[se1+'_'+se2] = np.nan #arange(0,len(onoff3))
                crossccR[se1+'_'+se2] = np.nan 
                crossccL[se1+'_'+se2] = np.nan 
                idx = 0
                idxb = 0
                if len(onoff2)>len(onoff3):
                    for tu1,tu2 in zip(onoff2[0:-1],onoff3):    
                        areaR[se1+'_'+se2][idx] = np.mean(np.abs((df2[se1].values[tu1:tu2]) - (df2[se2].values[tu1:tu2])))
                        areaRb[se1+'_'+se2][idx] = np.mean((df2[se1].values[tu1:tu2]) - (df2[se2].values[tu1:tu2]))
                        crossccR[se1+'_'+se2][idx] = s2.xcorr((df2[se1].values[tu1:tu2]),(df2[se2].values[tu1:tu2]))[1][10]
                        idx += 1
 
                    
                    for tu1b,tu2b in zip(onoff3,onoff2[1:]):
                        areaL[se1+'_'+se2][idxb] = np.mean(np.abs((df2[se1].values[tu1b:tu2b]) - (df2[se2].values[tu1b:tu2b])))
                        areaLb[se1+'_'+se2][idxb] = np.mean((df2[se1].values[tu1b:tu2b]) - (df2[se2].values[tu1b:tu2b]))
                        crossccL[se1+'_'+se2][idxb] = s2.xcorr((df2[se1].values[tu1b:tu2b]),(df2[se2].values[tu1b:tu2b]))[1][10]
                        idxb += 1
                elif len(onoff2)==len(onoff3):      
                    for tu1,tu2 in zip(onoff2,onoff3):    
                        areaR[se1+'_'+se2][idx] = np.mean(np.abs((df2[se1].values[tu1:tu2]) - (df2[se2].values[tu1:tu2])))
                        areaRb[se1+'_'+se2][idx] = np.mean((df2[se1].values[tu1:tu2]) - (df2[se2].values[tu1:tu2]))
                        crossccR[se1+'_'+se2][idx] = s2.xcorr((df2[se1].values[tu1:tu2]) , (df2[se2].values[tu1:tu2]))[1][10]
                        idx += 1
 
                    
                    for tu1b,tu2b in zip(onoff3[0:-1],onoff2[1:]):
                        areaL[se1+'_'+se2][idxb] = np.mean(np.abs((df2[se1].values[tu1b:tu2b]) - (df2[se2].values[tu1b:tu2b])))
                        areaLb[se1+'_'+se2][idxb] = np.mean((df2[se1].values[tu1b:tu2b]) - (df2[se2].values[tu1b:tu2b]))
                        crossccL[se1+'_'+se2][idxb] = s2.xcorr((df2[se1].values[tu1b:tu2b]) , (df2[se2].values[tu1b:tu2b]))[1][10]
                        idxb += 1
                        
        plt.figure()
        plt.pcolor(areaL[["Head_Sternum","Head_Lumbar","Lumbar_Sternum"]],vmin=0,vmax=35)
        plt.xticks([0.5,1.5,2.5],["Head_Sternum","Head_Lumbar","Lumbar_Sternum"])
        plt.ylabel("Turns")
        plt.xlabel("Area")
        plt.title('L turns ' + pa)
        plt.colorbar()
        
        plt.figure()
        plt.pcolor(areaR[["Head_Sternum","Head_Lumbar","Lumbar_Sternum"]],vmin=0,vmax=35)
        plt.xticks([0.5,1.5,2.5],["Head_Sternum","Head_Lumbar","Lumbar_Sternum"])
        plt.ylabel("Turns")
        plt.xlabel("Area")
        plt.title('R turns ' + pa)
        plt.colorbar()
        
        plt.figure()
        plt.pcolor(crossccL[["Head_Sternum","Head_Lumbar","Lumbar_Sternum"]],vmin=0.99,vmax=1)
        plt.xticks([0.5,1.5,2.5],["Head_Sternum","Head_Lumbar","Lumbar_Sternum"])
        plt.ylabel("Turns")
        plt.xlabel("CrossCorrelation")
        plt.title('L turns ' + pa)
        plt.colorbar()
        
        plt.figure()
        plt.pcolor(crossccR[["Head_Sternum","Head_Lumbar","Lumbar_Sternum"]],vmin=0.99,vmax=1)
        plt.xticks([0.5,1.5,2.5],["Head_Sternum","Head_Lumbar","Lumbar_Sternum"])
        plt.ylabel("Turns")
        plt.xlabel("CrossCorrelation")
        plt.title('R turns ' + pa)
        plt.colorbar()
        
        
        enbloc_all_R.append(areaR[1:-1])
        enbloc_all_L.append(areaL[1:-1])
        cc_all_R.append(crossccR[1:-1])
        cc_all_L.append(crossccL[1:-1])        

        
        # plt.show()
        # inp=input("Do you like the results, yes or no, 1 or 0 :  ")
        # userinput.append(inp)    
    # exceptpb =1
    # if exceptpb:
    #     print('error '+pa)
    #     errorsfi.append(pa)
        
    except:
        print("error part 2 " +pa)
        errorsfi2.append(pa)
        continue
        
#%save dataframes

info=pd.DataFrame(turninfo_all[1:],columns=turninfo_all[0])
infoenblockR = pd.DataFrame(np.concatenate(enbloc_all_R))
infoenblockR.columns= enbloc_all_R[0].columns
infoenblockL = pd.DataFrame(np.concatenate(enbloc_all_L))
infoenblockL.columns= enbloc_all_R[0].columns

infoccR = pd.DataFrame(np.concatenate(cc_all_R))
infoccR.columns= cc_all_R[0].columns
infoccL = pd.DataFrame(np.concatenate(cc_all_L))
infoccL.columns= cc_all_R[0].columns

info_onsetsR=[]
info_onsetsL=[]
for idx,pa in zip(range(len(info)),info.participant) :
    # print(idx)
    info_onsetsR2=[]
    info_onsetsR2=pd.DataFrame(onsets_all_R[idx],columns=["Head","Sternum","Lumbar"])
    info_onsetsR2["Sub_ID"] = info.participant[idx]
    info_onsetsR.append(info_onsetsR2)
    
    info_onsetsL2=[]
    info_onsetsL2=pd.DataFrame(onsets_all_L[idx],columns=["Head","Sternum","Lumbar"])
    info_onsetsL2["Sub_ID"] = info.participant[idx]
    info_onsetsL.append(info_onsetsL2) 
    
info_onsetsRf = pd.DataFrame(np.concatenate(info_onsetsR))
info_onsetsRf.columns=["Head","Sternum","Lumbar","Sub_ID"]
info_onsetsLf = pd.DataFrame(np.concatenate(info_onsetsL))
info_onsetsLf.columns=["Head","Sternum","Lumbar","Sub_ID"]   
    
  
info.to_excel("info1.xlsx")    
infoenblockR.to_excel("infoenblockR1.xlsx")
infoenblockL.to_excel("infoenblockL1.xlsx")
info_onsetsRf.to_excel("info_onsetsR1.xlsx")
info_onsetsLf.to_excel("info_onsetsL1.xlsx")
infoccR.to_excel("infoCC_R.xlsx")
infoccL.to_excel("infoCC_L.xlsx")

#%% stats
import seaborn as sns
info_onsetsRf["group"]= "PD"
info_onsetsRf.group[info_onsetsRf.Sub_ID.str.contains('Long_2')] = "HC"

info_onsetsLf["group"]= "PD"
info_onsetsLf.group[info_onsetsLf.Sub_ID.str.contains('Long_2')] = "HC"

a = info_onsetsRf.groupby(["group","Sub_ID"],as_index=False).mean()
# plt.figure()
# sns.histplot(data=a, hue="group", x="Sternum")
# plt.ylim([0,3])
# plt.figure(figsize=(6, 4))
# plt.hist(v1, bins=bins, density=True, label="Robust")
# plt.hist(v2, bins=bins, density=True, label="Prefrail")
plt.figure()
sns.histplot(x=a.Head,hue=a.group)
plt.figure()
sns.histplot(x=a.Sternum,hue=a.group)
plt.figure()
sns.histplot(x=a.Lumbar,hue=a.group)



summaryonsets= info_onsetsRf==0



#### relate one variable with clinical features
#### rigidity, 
## get weight, 
# updrs
# # weight + height
# excel table
# cross correlation

