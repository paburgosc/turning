# -*- coding: utf-8 -*-
"""
Created on Tue May 21 04:29:59 2024

@author: burgosp
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg


files = pd.read_excel('./data/files3.xlsx')
info= pd.read_excel("./summary/info1.xlsx")    
infoenblockR= pd.read_excel("./summary/infoenblockR1.xlsx")
infoenblockL= pd.read_excel("./summary/infoenblockL1.xlsx")
info_onsetsRf= pd.read_excel("./summary/info_onsetsR1.xlsx")
info_onsetsLf= pd.read_excel("./summary/info_onsetsL1.xlsx")
info_CCR= pd.read_excel("./summary/infoCC_R.xlsx")
info_CCL = pd.read_excel("./summary/infoCC_L.xlsx")

info_onsetsRf = info_onsetsRf.iloc[:,1:]
info_onsetsLf = info_onsetsLf.iloc[:,1:]

info_onsetsLf['trial']=info_onsetsLf.groupby("Sub_ID",sort=False).cumcount()
info_onsetsRf['trial']=info_onsetsRf.groupby("Sub_ID",sort=False).cumcount()

rateonsetR = info_onsetsRf==0
rateonsetR.Sub_ID = info_onsetsRf.Sub_ID
rateonsetR.trial = info_onsetsRf.trial

rateonsetRF=rateonsetR.groupby(["Sub_ID"],sort=False).sum() 
rateonsetRF = rateonsetRF.iloc[:,0:3]

for cols in range(0,3):
    rateonsetRF.iloc[:,cols] = rateonsetRF.iloc[:,cols] / (rateonsetR.groupby("Sub_ID",sort=False).trial.max() +1)



## INFOENBLOCK
idxexcludeR = infoenblockR.groupby("Sub_ID",sort=False).index2.idxmax()
idxexcludeL = infoenblockL.groupby("Sub_ID",sort=False).index2.idxmax()

infoenblockR.iloc[idxexcludeR]= np.nan
infoenblockL.iloc[idxexcludeL]= np.nan

enblocR=infoenblockR.groupby("Sub_ID",sort=False,as_index=False)[['Head_Sternum','Head_Lumbar','Lumbar_Sternum']].mean()
enblocL=infoenblockL.groupby("Sub_ID",sort=False,as_index=False)[['Head_Sternum','Head_Lumbar','Lumbar_Sternum']].mean()
enblocR=enblocR.rename(columns={"Sub_ID": "participant"})
enblocL=enblocL.rename(columns={"Sub_ID": "participant"})

## INFOCROSSCORR
idxCCexcludeR = info_CCR.groupby("Sub_ID",sort=False).index2.idxmax()
idxCCexcludeL = info_CCL.groupby("Sub_ID",sort=False).index2.idxmax()

info_CCR.iloc[idxCCexcludeR]= np.nan
info_CCL.iloc[idxCCexcludeL]= np.nan

CCR=info_CCR.groupby("Sub_ID",sort=False,as_index=False)[['Head_Sternum','Head_Lumbar','Lumbar_Sternum']].mean()
CCL=info_CCL.groupby("Sub_ID",sort=False,as_index=False)[['Head_Sternum','Head_Lumbar','Lumbar_Sternum']].mean()
CCR=CCR.rename(columns={"Sub_ID": "participant"})
CCL=CCL.rename(columns={"Sub_ID": "participant"})

## INFOTIME
idxTIexcludeR = info_onsetsRf.groupby("Sub_ID",sort=False).trial.idxmax()
idxTIexcludeL = info_onsetsLf.groupby("Sub_ID",sort=False).trial.idxmax()

info_TIR=info_onsetsRf.copy()
info_TIL=info_onsetsLf.copy()

info_TIR.iloc[idxTIexcludeR]= np.nan
info_TIL.iloc[idxTIexcludeL]= np.nan

info_TIR['Head_Sternum'] = np.abs(info_TIR.Head - info_TIR.Sternum)
info_TIR['Head_Lumbar'] = np.abs(info_TIR.Head - info_TIR.Lumbar)
info_TIR['Lumbar_Sternum'] = np.abs(info_TIR.Lumbar - info_TIR.Sternum)

info_TIL['Head_Sternum'] = np.abs(info_TIL.Head - info_TIL.Sternum)
info_TIL['Head_Lumbar'] = np.abs(info_TIL.Head - info_TIL.Lumbar)
info_TIL['Lumbar_Sternum'] = np.abs(info_TIL.Lumbar - info_TIL.Sternum)


TIR=info_TIR.groupby("Sub_ID",sort=False,as_index=False)[['Head_Sternum','Head_Lumbar','Lumbar_Sternum']].mean()
TIL=info_TIL.groupby("Sub_ID",sort=False,as_index=False)[['Head_Sternum','Head_Lumbar','Lumbar_Sternum']].mean()
TIR=TIR.rename(columns={"Sub_ID": "participant"})
TIL=TIL.rename(columns={"Sub_ID": "participant"})


###BE  CAREFULL WITH THE AUTOMATIC SORTING OF GROUPBY    
    
rateonsetL = info_onsetsLf==0
rateonsetL.Sub_ID = info_onsetsLf.Sub_ID  
rateonsetL.trial = info_onsetsLf.trial
  
rateonsetLF=rateonsetL.groupby(["Sub_ID"],sort=False).sum() 
rateonsetLF = rateonsetLF.iloc[:,0:3]

for cols in range(0,3):
    rateonsetLF.iloc[:,cols] = rateonsetLF.iloc[:,cols] / (rateonsetL.groupby("Sub_ID",sort=False).trial.max() +1)

rateonsetRF["Head_L"] = rateonsetLF["Head"]
rateonsetRF["Sternum_L"] = rateonsetLF["Sternum"]
rateonsetRF["Lumbar_L"] = rateonsetLF["Lumbar"]

rateonsetRF = rateonsetRF.reset_index()
rateonsetRF.columns=['participant', 'Head_R', 'Sternum_R', 'Lumbar_R', 'Head_L', 'Sternum_L',  'Lumbar_L']

# final = pd.merge(info,files,on ='participant',how ="left") # put larget sub id at left
# final['duration_R(mean)'] = -final['duration_R(mean)']
# final['duration_L(mean)'] = -final['duration_L(mean)']
# final['angle_L(mean)'] = final['angle_L(mean)'].abs()
# final['angle_R(mean)'] = final['angle_R(mean)'].abs()
# final["duration_max"] = final[['duration_R(mean)',
#        'duration_L(mean)']].max(axis=1)




final = pd.merge(info,rateonsetRF,on ='participant',how ="left") # put larget sub id at left
final.iloc[:,3:7]= final.iloc[:,3:7].abs()


files =  files[(files.condition =="st")&(files.visit ==0)]
files2 = files[['participant','diagnosis']]


final = pd.merge(final,files2,on='participant',how='left')



# Remove last turn
# try with the first
#%% head corr
co1 =final[['duration_R(mean)','Head_R']]
co1.columns = ['duration','head1']
co2 =final[['duration_L(mean)','Head_L']]
co2.columns = ['duration','head1']


corrvar = pd.concat([co1,co2], axis=0)
x2=corrvar.duration.values
y2=corrvar.head1.values
# sns.regplot(x=x2,y=y2)
# plt.xlabel('duration')
# plt.ylabel('head_rate')

finalHC = final[final.diagnosis=="HC"]

co1 =finalHC[['duration_R(mean)','Head_R']]
co1.columns = ['duration','head1']
co2 =finalHC[['duration_L(mean)','Head_L']]
co2.columns = ['duration','head1']
corrvar = pd.concat([co1,co2], axis=0)
x2=np.array(corrvar.duration)
y2=np.array(corrvar.head1)
sns.regplot(x=x2,y=y2)
plt.xlabel('duration')
plt.ylabel('head_rate')
plt.legend(['HC'])


finalPD = final[final.diagnosis=="PD"]

co1 =finalPD[['duration_R(mean)','Head_R']]
co1.columns = ['duration','head1']
co2 =finalPD[['duration_L(mean)','Head_L']]
co2.columns = ['duration','head1']
corrvar = pd.concat([co1,co2], axis=0)
x2=np.array(corrvar.duration)
y2=np.array(corrvar.head1)
sns.regplot(x=x2,y=y2)
plt.xlabel('duration')
plt.ylabel('head_rate')
plt.legend(['PD'])


#%% lumbar corr
co1 =final[['duration_R(mean)','Lumbar_R']]
co1.columns = ['duration','Lumbar']
co2 =final[['duration_L(mean)','Lumbar_L']]
co2.columns = ['duration','Lumbar']
corrvar = pd.concat([co1,co2], axis=0)
x2=np.array(corrvar.duration)
y2=np.array(corrvar.Lumbar)
sns.regplot(x=x2,y=y2)
plt.xlabel('duration')
plt.ylabel('lumbar_rate')

finalHC = final[final.diagnosis=="HC"]

co1 =finalHC[['duration_R(mean)','Lumbar_R']]
co1.columns = ['duration','Lumbar']
co2 =finalHC[['duration_L(mean)','Lumbar_L']]
co2.columns = ['duration','Lumbar']
corrvar = pd.concat([co1,co2], axis=0)
x2=np.array(corrvar.duration)
y2=np.array(corrvar.Lumbar)
sns.regplot(x=x2,y=y2)
plt.xlabel('duration')
plt.ylabel('lumbar_rate')



finalPD = final[final.diagnosis=="PD"]

co1 =finalPD[['duration_R(mean)','Lumbar_R']]
co1.columns = ['duration','Lumbar']
co2 =finalPD[['duration_L(mean)','Lumbar_L']]
co2.columns = ['duration','Lumbar']
corrvar = pd.concat([co1,co2], axis=0)
x2=np.array(corrvar.duration)
y2=np.array(corrvar.Lumbar)
sns.regplot(x=x2,y=y2)
plt.xlabel('duration')
plt.ylabel('lumbar_rate')

#%% sternum corr
co1 =final[['duration_R(mean)','Sternum_R']]
co1.columns = ['duration','Sternum']
co2 =final[['duration_L(mean)','Sternum_L']]
co2.columns = ['duration','Sternum']
corrvar = pd.concat([co1,co2], axis=0)
x2=np.array(corrvar.duration)
y2=np.array(corrvar.Sternum)
sns.regplot(x=x2,y=y2)
plt.xlabel('duration')
plt.ylabel('Sternum_rate')

finalHC = final[final.diagnosis=="HC"]

co1 =finalHC[['duration_R(mean)','Sternum_R']]
co1.columns = ['duration','Sternum']
co2 =finalHC[['duration_L(mean)','Sternum_L']]
co2.columns = ['duration','Sternum']
corrvar = pd.concat([co1,co2], axis=0)
x2=np.array(corrvar.duration)
y2=np.array(corrvar.Sternum)
sns.regplot(x=x2,y=y2)
plt.xlabel('duration')
plt.ylabel('Sternum_rate')



finalPD = final[final.diagnosis=="PD"]

co1 =finalPD[['duration_R(mean)','Sternum_R']]
co1.columns = ['duration','Sternum']
co2 =finalPD[['duration_L(mean)','Sternum_L']]
co2.columns = ['duration','Sternum']
corrvar = pd.concat([co1,co2], axis=0)
x2=np.array(corrvar.duration)
y2=np.array(corrvar.Sternum)
sns.regplot(x=x2,y=y2)
plt.xlabel('duration')
plt.ylabel('Sternum_rate')




#%% t test



final["angle_min"] = final[['angle_R(mean)',
        'angle_L(mean)']].min(axis=1)

final["angle_min"]= final["angle_min"]-10



# final["angle_min_side"] = np.nan
# final["angle_min_side"][final['angle_R(mean)']==final["angle_min"]] ="R"
# final["angle_min_side"][final['angle_L(mean)']==final["angle_min"]] ="L"

final["duration_max"] = final[['duration_R(mean)',
        'duration_L(mean)']].max(axis=1)

final["duration_min"] = final[['duration_R(mean)',
        'duration_L(mean)']].min(axis=1)

final["duration_max_side"] = np.nan
final["duration_max_side"][final['duration_R(mean)']==final["duration_max"]] ="R"
final["duration_max_side"][final['duration_L(mean)']==final["duration_max"]] ="L"



Rrate= final[['Head_R','Sternum_R','Lumbar_R']]
Lrate= final[['Head_L','Sternum_L','Lumbar_L']]

final["preferenceR"] = Rrate.idxmax(axis=1)
final["preferenceL"] = Lrate.idxmax(axis=1)

final["preferenceSlower"] = np.nan
final["preferenceSlower"][final["duration_max_side"]=='R']= final.preferenceR[final["duration_max_side"]=='R']
final["preferenceSlower"][final["duration_max_side"]=='L']= final.preferenceL[final["duration_max_side"]=='L']
final["preferenceSlower"]=final["preferenceSlower"].str[0:-2]

final["preferenceFaster"] = np.nan
final["preferenceFaster"][final["duration_max_side"]=='R']= final.preferenceL[final["duration_max_side"]=='R']
final["preferenceFaster"][final["duration_max_side"]=='L']= final.preferenceR[final["duration_max_side"]=='L']
final["preferenceFaster"]=final["preferenceFaster"].str[0:-2]

#ENBLOC AREA
finalenbloc = pd.merge(final,enblocR,on='participant',how='right')
finalenbloc.Head_Sternum[final.duration_max_side=='L'] = enblocL.Head_Sternum[final.duration_max_side=='L']
finalenbloc.Head_Lumbar[final.duration_max_side=='L'] = enblocL.Head_Lumbar[final.duration_max_side=='L']
finalenbloc.Lumbar_Sternum[final.duration_max_side=='L'] = enblocL.Lumbar_Sternum[final.duration_max_side=='L']


final = pd.merge(final,finalenbloc[['participant','Head_Sternum','Head_Lumbar','Lumbar_Sternum']], on='participant',how='right')
#normalize by time
final.Head_Sternum = final.Head_Sternum/final.duration_max
final.Head_Lumbar = final.Head_Lumbar/final.duration_max
final.Lumbar_Sternum = final.Lumbar_Sternum/final.duration_max

#CC
finalCC = pd.merge(final,CCR,on='participant',how='right')
finalCC.Head_Sternum_y[finalCC.duration_max_side=='L'] = CCL.Head_Sternum[finalCC.duration_max_side=='L']
finalCC.Head_Lumbar_y[finalCC.duration_max_side=='L'] = CCL.Head_Lumbar[finalCC.duration_max_side=='L']
finalCC.Lumbar_Sternum_y[finalCC.duration_max_side=='L'] = CCL.Lumbar_Sternum[finalCC.duration_max_side=='L']


#Time
finalTI = pd.merge(final,TIR,on='participant',how='left')
finalTI2 = pd.merge(final,TIL,on='participant',how='left')
finalTI.Head_Sternum_y[finalTI.duration_max_side=='L'] = finalTI2.Head_Sternum_y[finalTI.duration_max_side=='L']
finalTI.Head_Lumbar_y[finalTI.duration_max_side=='L'] = finalTI2.Head_Lumbar_y[finalTI.duration_max_side=='L']
finalTI.Lumbar_Sternum_y[finalTI.duration_max_side=='L'] = finalTI2.Lumbar_Sternum_y[finalTI.duration_max_side=='L']

finalA = pd.merge(final,finalCC[['participant','Head_Sternum_y','Head_Lumbar_y','Lumbar_Sternum_y']], on='participant',how='right')



finalB = pd.merge(final,finalTI[['participant','Head_Sternum_y','Head_Lumbar_y','Lumbar_Sternum_y']], on='participant',how='right')

final['Head_Sternum_cc']=finalA['Head_Sternum_y']
final['Head_Lumbar_cc']=finalA['Head_Lumbar_y']
final['Lumbar_Sternum_cc']=finalA['Lumbar_Sternum_y']


final['Head_Sternum_t']=finalB['Head_Sternum_y']
final['Head_Lumbar_t']=finalB['Head_Lumbar_y']
final['Lumbar_Sternum_t']=finalB['Lumbar_Sternum_y']


# final.Head_Sternum_t = final.Head_Sternum_t/final.duration_max
# final.Head_Lumbar_t = final.Head_Lumbar_t/final.duration_max
# final.Lumbar_Sternum_t = final.Lumbar_Sternum_t/final.duration_max



##normality
pg.normality(final, method='normaltest').round(3)

varsp = ["number_of_turns","angle_min"] 
varspt = ['number of turns','angle (deg)']





sns.set(style="whitegrid", font_scale=1.5 )
for var,vart in zip(varsp,varspt):
    plt.figure()
    tt1=pg.ttest(final[var][final.diagnosis=="PD"],
             final[var][final.diagnosis=="HC"],
             paired = False)
    g1=sns.boxplot(data=final,x="diagnosis",y=var)
    g1=sns.swarmplot(data=final,x="diagnosis",y=var)
    g1.set(xlabel=None)
    g1.set(ylabel=vart)
    plt.title("p = "+ str(tt1["p-val"][0])[0:7])
    
    
varsp = ["duration_max","duration_min",'Head_Sternum','Head_Lumbar','Lumbar_Sternum','Head_Sternum_cc','Head_Lumbar_cc','Lumbar_Sternum_cc','Head_Sternum_t','Head_Lumbar_t','Lumbar_Sternum_t']
varspt = ['duration maximum (s)', 'duration minimum (s)', 'Area Head-Sternum (deg s) ','Area Head-Lumbar (deg s)','Area Sternum-Lumbar (deg s)','Crosscor Head-Sternum (deg s) ','Crosscor Head-Lumbar (deg s)','Crosscor Sternum-Lumbar (deg s)',
          'Delay Head-Sternum (s)','Delay Head-Lumbar(s)','Delay Sternum-Lumbar (s)']

sns.set(style="whitegrid", font_scale=1.5 )
for var,vart in zip(varsp,varspt):
    plt.figure()
    tt1=pg.mwu(final[var][final.diagnosis=="PD"],
             final[var][final.diagnosis=="HC"])
    g1=sns.boxplot(data=final,x="diagnosis",y=var)
    g1=sns.swarmplot(data=final,x="diagnosis",y=var)
    g1.set(xlabel=None)
    g1.set(ylabel=vart)
    plt.title("p = "+ str(tt1["p-val"][0])[0:7])
    if var in ["duration_max","duration_min"]:
        plt.ylim([0,9.1])
    # plt.ylim([0,10])
    


#%% corr HC turn duration and head onset

last = pd.read_excel('./summary/FINAL2024.xlsx')
lastcols = ['participant','enrollmentage','moca', 'gender','pd_durac','falls_year', 'fog','updrs3_rigidity_subscore','updrs3_brady_subscore',
            'updrs3_totalscore','updrs4_subscore','mds_updrs_total_score', 'mb_totalscore']


final2 = pd.merge(finalPD,last[['participant','pd_durac']],on='participant',how='left')








#%% plot preference Slower per group

Rrate= final[['Head_R','Sternum_R','Lumbar_R']]
Lrate= final[['Head_L','Sternum_L','Lumbar_L']]
Diag = final.diagnosis


preference = np.zeros(np.shape(Rrate))
preference[final.preferenceSlower=="Head",0]=1
preference[final.preferenceSlower=="Sternum",1]=1
preference[final.preferenceSlower=="Lumbar",2]=1

#size letter
plt.rcParams.update({'font.size': 16})

#general
plt.figure()
plt.pcolor(preference)
plt.title("Preference All")

#PD
plt.figure()
preferencePD = preference[Diag=="PD",:]
plt.pcolor(preferencePD)
plt.title("Preference PD")

#HC
plt.figure()
preferenceHC = preference[Diag=="HC",:]
plt.pcolor(preferenceHC)
plt.title("Preference HC")

plt.figure()
sns.barplot(preferencePD,errorbar=("se"),color='b')
# plt.bar([0,1,2],np.sum(preferencePD,axis=0))
plt.xticks([0,1,2],["Head","Sternum","Lumbar"])
plt.ylabel("average rate")
plt.ylim([0,0.6])
# plt.title("Preference PD")


plt.figure()
# plt.bar([0,1,2],np.sum(preferenceHC,axis=0))
sns.barplot(preferenceHC,errorbar=("se"),color='b')
plt.xticks([0,1,2],["Head","Sternum","Lumbar"])
plt.ylabel("average rate")
plt.ylim([0,0.6])
# plt.title("Preference HC")

#%% plot preference Faster per group


Rrate= final[['Head_R','Sternum_R','Lumbar_R']]
Lrate= final[['Head_L','Sternum_L','Lumbar_L']]
Diag = final.diagnosis


preference = np.zeros(np.shape(Rrate))
preference[final.preferenceFaster=="Head",0]=1
preference[final.preferenceFaster=="Sternum",1]=1
preference[final.preferenceFaster=="Lumbar",2]=1



plt.figure()
plt.bar([0,1,2],np.sum(preferencePD,axis=0))
plt.xticks([0,1,2],["Head","Sternum","Lumbar"])
plt.ylabel("Participants (number)")
plt.title("Preference PD")


plt.figure()
plt.bar([0,1,2],np.sum(preferenceHC,axis=0))
plt.xticks([0,1,2],["Head","Sternum","Lumbar"])
plt.ylabel("Participants (number)")
plt.title("Preference HC")


#%% plot RATE preference Slower per group

Rrate= final[['Head_R','Sternum_R','Lumbar_R']]
Lrate= final[['Head_L','Sternum_L','Lumbar_L']]
Diag = final.diagnosis

Side= final.duration_max_side


preference = Rrate.copy()


preference[Side=="L"]= Lrate[Side=="L"]


#size letter
plt.rcParams.update({'font.size': 16})
plt.rcParams['image.cmap'] = 'viridis' 

#general
plt.figure()
plt.pcolor(preference.values)
plt.title("Preference All")

#PD

plt.figure()
preferencePD = preference[Diag=="PD"]
plt.pcolor(preferencePD)
# plt.colorbar()
# plt.title("Preference PD")
plt.xticks([0.5,1.5,2.5],["Head","Sternum","Lumbar"])
cbar = plt.colorbar(label= 'rate of preference')
plt.ylabel("Participant")
# Set the title of the colorbar

#HC
plt.figure()
preferenceHC = preference[Diag=="HC"]
plt.pcolor(preferenceHC)
# plt.colorbar()
cbar = plt.colorbar(label= 'rate of preference')
# plt.title("Preference HC")
plt.xticks([0.5,1.5,2.5],["Head","Sternum","Lumbar"])
plt.ylabel("Participant")

plt.figure()
plt.bar([0,1,2],np.mean(preferencePD,axis=0))
plt.xticks([0,1,2],["Head","Sternum","Lumbar"])
plt.ylabel("rate")
# plt.title("Preference PD")


plt.figure()
plt.bar([0,1,2],np.mean(preferenceHC,axis=0))
plt.xticks([0,1,2],["Head","Sternum","Lumbar"])
plt.ylabel("rate")
# plt.title("Preference HC")

#%% correlations clinical
import pingouin as pg
met = ['pearson','spearman','kendall','bicor','percbend','shepherd','skipped']
cor = ['bonf','sidak','holm','fdr_bh', 'fdr_by']


m = met[5]
c= cor[3]

last = pd.read_excel('./summary/FINAL2024.xlsx')
lastcols = ['participant','enrollmentage','moca', 'gender','pd_durac','falls_year', 'fog',
            'updrs3_totalscore', 'mb_totalscore']
lastcols2 = ['enrollmentage','moca','pd_durac','falls_year',
            'updrs3_totalscore', 'mb_totalscore']
lastcols3 = ['enrollmentage','moca','mb_totalscore']
lastcols4 = ['angle_min', 'duration_max', 'duration_min']

colsturn=['Head_rate','Sternum_rate','Lumbar_rate']
colsturn2=['Head_Sternum', 'Head_Lumbar', 'Lumbar_Sternum', 'Head_Sternum_cc']
colsturn3=[ 'Head_Sternum_cc',
'Head_Lumbar_cc', 'Lumbar_Sternum_cc' ]
colsturn4=['Head_Sternum_t',
'Head_Lumbar_t', 'Lumbar_Sternum_t']

last2 = last[lastcols]

final2 = pd.merge(final,last2,on='participant',how='left')
final2["Head_rate"] = np.nan
final2["Head_rate"][final2.duration_max_side=='R']= final2["Head_R"][final2.duration_max_side=='R']
final2["Head_rate"][final2.duration_max_side=='L']= final2["Head_L"][final2.duration_max_side=='L']

final2["Sternum_rate"] = np.nan
final2["Sternum_rate"][final2.duration_max_side=='R']= final2["Sternum_R"][final2.duration_max_side=='R']
final2["Sternum_rate"][final2.duration_max_side=='L']= final2["Sternum_L"][final2.duration_max_side=='L']

final2["Lumbar_rate"] = np.nan
final2["Lumbar_rate"][final2.duration_max_side=='R']= final2["Lumbar_R"][final2.duration_max_side=='R']
final2["Lumbar_rate"][final2.duration_max_side=='L']= final2["Lumbar_L"][final2.duration_max_side=='L']

final2PD = final2[final2.diagnosis=="PD"]
final2HC = final2[final2.diagnosis=="HC"]


dfr0=pg.pairwise_corr(final2PD,columns=[lastcols2, colsturn], method=m, alternative='two-sided', padjust=c).round(3)
dfr1=pg.pairwise_corr(final2,columns=[lastcols3, colsturn], method=m, alternative='two-sided', padjust=c).round(3)
dfr2=pg.pairwise_corr(final2,columns=[lastcols3, colsturn], method=m, alternative='two-sided', padjust=c).round(3)


dfr3=pg.pairwise_corr(final2,columns=[lastcols3, colsturn2], method=m, alternative='two-sided', padjust=c).round(3)
dfr4=pg.pairwise_corr(final2PD,columns=[lastcols2, colsturn2], method=m, alternative='two-sided', padjust=c).round(3)
dfr5=pg.pairwise_corr(final2PD,columns=[lastcols2, colsturn3], method=m, alternative='two-sided', padjust=c).round(3)
dfr6=pg.pairwise_corr(final2PD,columns=[lastcols2, colsturn4], method=m, alternative='two-sided', padjust=c).round(3)

sns.jointplot(data=final2PD, x="pd_durac", y="Head_rate", kind="reg")
plt.ylabel('Head onset (rate)')
plt.xlabel('PD duration (years)')


sns.jointplot(data=final2PD, x="pd_durac", y="Sternum_rate", kind="reg")
plt.ylabel('Sternum onset (rate)')
plt.xlabel('PD duration (years)')

sns.jointplot(data=final2PD, x="mb_totalscore", y="Head_rate", kind="reg")
plt.ylabel('Head onset (rate)')
plt.xlabel('Mini BESTest (score)')

sns.jointplot(data=final2PD, x="pd_durac", y="Sternum_rate", kind="reg")
sns.jointplot(data=final2PD, x="pd_durac", y="Lumbar_rate", kind="reg")

sns.jointplot(data=final2PD, x="moca", y="Sternum_rate", kind="reg")

sns.jointplot(data=final2PD, x="updrs3_totalscore", y="Sternum_rate", kind="reg")

sns.jointplot(data=final2PD, x="updrs3_totalscore", y="Lumbar_rate", kind="reg")



sns.jointplot(data=final2PD, x="duration_max", y="Head_rate", kind="reg")


