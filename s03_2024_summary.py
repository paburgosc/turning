# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 11:11:24 2024

@author: burgosp


Manuscript

Objective: Understand the onset of turning and the relation with balance impairments(MiniBest, UPDRS. Disease DUration)
cephalo-caudal, caudo-cephalic, or “en bloc” strategy)

Masoud, Figures and Stats from final DataFrame
Valerio, Correlation, Long Walk Long and Turning
Pablo, Long Walk Automaticity, send email about Martina opinion



Clinical data

'UPDRS3'
UPDRS 
DISEASE DURATION
FIRST VISIT
MINIBEST
AGE
MOCA
FOG
FALLS
NFoG....

Gait Speed from "2 minutes walk in 7mt" ST and DT
Gait variables
Turning Variables


CHECK TURN DURATION ESTIMATION

ADD HEAD ONSET %, LUMBAR OBSET %, MIN AREA, MAX CrossCorr


"""

import pandas as pd
import numpy as np


#%% IMPORT TURNING INFO COORDINATION
df1 = pd.read_excel("./data/files3.xlsx") #general
df1 = df1.rename(columns={"participant": "SubID"})


dfinfoturns = pd.read_excel("info1.xlsx")
dfinfoturns = dfinfoturns.rename(columns={"participant": "SubID"})
# dfinfoturns[['duration_R(mean)',
#        'duration_L(mean)', 'angle_R(mean)', 'angle_L(mean)']] = infoturns[['duration_R(mean)',
#               'duration_L(mean)', 'angle_R(mean)', 'angle_L(mean)']].abs()

dfonsetsL = pd.read_excel('info_onsetsL1.xlsx')
dfonsetsR = pd.read_excel('info_onsetsR1.xlsx')

dfenblockL = pd.read_excel('infoenblockL1.xlsx')
dfenblockR = pd.read_excel('infoenblockR1.xlsx')

dfCC_L = pd.read_excel('infoCC_L.xlsx')
dfCC_R = pd.read_excel('infoCC_R.xlsx')


dfonsetsL.columns = ['NOL', 'Onsets_L_Head', 'Onsets_L_Sternum', 'Onsets_L_Lumbar', 'SubID']
dfonsetsR.columns = ['NOR', 'Onsets_R_Head', 'Onsets_R_Sternum', 'Onsets_R_Lumbar', 'SubID']

dfenblockL.columns = ['NEL', 'index2', 'SubID', 'Area_L_Head_Head', 'Area_L_Head_Sternum',
       'Area_L_Head_Lumbar', 'Area_L_Lumbar_Head', 'Area_L_Lumbar_Sternum', 'Area_L_Lumbar_Lumbar']

dfenblockR.columns = ['NER', 'index2', 'SubID', 'Area_R_Head_Head', 'Area_R_Head_Sternum',
       'Area_R_Head_Lumbar', 'Area_R_Lumbar_Head', 'Area_R_Lumbar_Sternum', 'Area_R_Lumbar_Lumbar']

dfonsetsL2 = dfonsetsL.groupby(["SubID"],as_index=False).mean()
dfonsetsR2 = dfonsetsR.groupby(["SubID"],as_index=False).mean()

dfenblockL2 = dfenblockL.groupby(["SubID"],as_index=False).mean()
dfenblockR2 = dfenblockR.groupby(["SubID"],as_index=False).mean()

#%%  ADD TURNING INFO
moca = pd.read_excel("./data/moca2.xlsx", header=0)
demo = pd.read_excel("./data/demo.xlsx", header=0)
yoe =  pd.read_excel("./data/turning_yoe.xlsx", header=0)
clin = pd.read_excel("./data/clinical2.xlsx", header=0)
final = pd.merge(dfinfoturns,clin,on ='SubID',how ="left")
final = pd.merge(final,yoe,on ='SubID',how ="left")
final = pd.merge(final,moca,on ='SubID',how ="left")
final = pd.merge(final,demo,on ='SubID',how ="left")

final = pd.merge(final,dfonsetsL2,on ='SubID',how ="left")
final = pd.merge(final,dfonsetsR2,on ='SubID',how ="left")
final = pd.merge(final,dfenblockL2,on ='SubID',how ="left")
final = pd.merge(final,dfenblockR2,on ='SubID',how ="left")
# final = pd.merge(final,dfinfoturns,on ='SubID',how ="left")

final["fall"]=np.NAN
final["fall"][final.falls_year_x>0]=1
final["fall"][final.falls_year_x<=0]=0

# final.fog[final.fog==1]= "FoG"
# final.fog[final.fog==0]= "noFoG"
final["pd_durac"] = final.yoe -final.pd_dur 


final.to_excel("FINALTURNING.xlsx")

#%% ADD LONGITUDINAL INFO

long_redcap = pd.read_csv('./data/LongitudinalRedcap091124.csv')

long_columns = ["record_id","redcap_event_name","dob","visitdate","visitage","gender",
                "falls_12month","pd_dx","pd_dur_symp","pd_laterality_dx",
                "fog" ,"gender_hc","visitage_hc","falls_12month_hc","moca_total_3d18c4",
                "updrs3_totalscore","mds_updrs_total_score","mb_ant_subscore",
                "mb_react_subscore","mb_sens_subscore","mb_dyn_subscore","mb_totalscore",
                "nfogq_score","pa_bmicalc","pdq_39_score","abc_total","festotal"  
    ]

long_redcap2 = long_redcap[long_columns]
long_redcap3pd = long_redcap2[(long_redcap2.redcap_event_name=="baseline_arm_1")]
long_redcap3hc = long_redcap2[(long_redcap2.redcap_event_name=="baseline_arm_2")]

long_redcap3 = long_redcap2[(long_redcap2.redcap_event_name=="baseline_arm_1")|(long_redcap2.redcap_event_name=="baseline_arm_2")]
long_redcap4 = long_redcap3.groupby("record_id").first()

long_redcap4pd = long_redcap3pd.groupby("record_id").first()
long_redcap4hc = long_redcap3hc.groupby("record_id").first()

long_redcap4["gender"][long_redcap4.redcap_event_name=="baseline_arm_2"] = long_redcap4["gender_hc"][long_redcap4.redcap_event_name=="baseline_arm_2"]
long_redcap4["visitage"][long_redcap4.redcap_event_name=="baseline_arm_2"] = long_redcap4["visitage_hc"][long_redcap4.redcap_event_name=="baseline_arm_2"]
long_redcap4["falls_12month"][long_redcap4.redcap_event_name=="baseline_arm_2"] = long_redcap4["falls_12month_hc"][long_redcap4.redcap_event_name=="baseline_arm_2"]
long_redcap4["pd_duration"]=(pd.to_datetime(long_redcap4["visitdate"])-pd.to_datetime(long_redcap4["pd_dx"])).dt.days/365.25

long_redcap4.to_excel("LONG.xlsx")
#%% ADD AUTOMATICITY INFO

auto_redcap = pd.read_csv('./data/AutomaticityRedcap091124.csv')

auto_columns = [
'record_id','redcap_event_name','doe','levodopa_group','gender','dob','enrollmentage','falls_year',
'pd_dur','pd_dur_symp','pd_laterality_dx','fog','visit_datetime','on_off','updrs_rest_trem_subscore',
'updrs_action_trem_subscore','updrs_post_trem_subscore','updrs3_rigidity_subscore','updrs3_brady_subscore',
'updrs3_pigd_subscore','updrs3_totalscore','updrs3_dysk_a','updrs3_dysk_b','updrs3_hy','updrs4_subscore',
'mds_updrs_total_score','mb_ant_subscore','mb_react_subscore','mb_sens_subscore','mb_dyn_subscore',
'mb_totalscore','nfogq_score','pdq_39_score','moca_total_3d18c4'   
    ]

auto_redcap2 = auto_redcap[auto_columns]


auto_redcap2.rename(columns={"record_id": "SubID"})


test = pd.merge(final,auto_redcap2,on ='SubID',how ="left")

auto_redcap2.to_excel("AUTO.xlsx")


# MANUAL MERGE  USE FINAL2024,   FINAL2024Clinical to update turning metrics
#%% plot distribution HC vs PD

import seaborn as sns
import matplotlib.pyplot as plt


# plt.figure()
# sns.histplot(data=a, hue="group", x="Sternum")
# plt.ylim([0,3])
# plt.figure(figsize=(6, 4))
# plt.hist(v1, bins=bins, density=True, label="Robust")
# plt.hist(v2, bins=bins, density=True, label="Prefrail")
feats=['Onsets_L_Head',
       'Onsets_L_Sternum', 'Onsets_L_Lumbar',  'Onsets_R_Head',
       'Onsets_R_Sternum', 'Onsets_R_Lumbar',  
       'Area_L_Head_Sternum', 'Area_L_Head_Lumbar',
       'Area_L_Lumbar_Head', 'Area_L_Lumbar_Sternum', 
       'Area_R_Head_Sternum',
       'Area_R_Head_Lumbar', 'Area_R_Lumbar_Head', 'Area_R_Lumbar_Sternum',
       'number_of_turns',
       'duration_R(mean)', 'duration_L(mean)', 'angle_R(mean)',
       'angle_L(mean)'  ]

for feat in feats:
    plt.figure()
    # sns.histplot(x=final[feat],hue=final.diagnosis,bins=50)
    sns.boxplot(data=final,y=feat,x='diagnosis')



# summaryonsets= info_onsetsRf==0

