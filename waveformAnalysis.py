#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 13:51:54 2023

@author: danielvalmassei
"""

import mplhep as hep
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main(runNum,events,triggerThreshold,nBins):
    plt.style.use(hep.style.LHCb2)
    
    run = runNum
    
    
    print('Loading data. This could take a couple minutes. Please wait...')

    df0 = pd.read_csv(run + "/TR_0_0.txt", header = None)
    df1 = pd.read_csv(run + "/wave_0.txt", header = None)
    df2 = pd.read_csv(run + "/wave_1.txt", header = None)
    
    print('Data loaded!')

    trigger = np.array(df0[0])
    wave0 = np.array(df1[0])
    wave1 = np.array(df2[0])

    
    
    plt.plot(trigger)
    plt.plot(wave0)
    plt.plot(wave1)
    
    plt.show()
    
    print('Integrating signals. This could take a moment. Please wait...')
    
    #integrate waveforms and record energies in an array
    triggerOffset = trigger[0]
    
    wave0Offset = wave0[0]
    wave1Offset = wave1[0]
    
    length = len(trigger)
    
    nEvents = 0
    nBigEvents = 0
    
    q0 = np.empty(events)
    q1 = np.empty(events)
    
    q0Big = np.empty(10266)
    q1Big = np.empty(10266)
    
    i = 0
    
    while i < length:
        if trigger[i] > triggerOffset - triggerThreshold:
            i += 1
        
        else: #trigger[i] < triggerOffset - triggerThreshold:
            j = i + 1
            
            while trigger[j] < triggerOffset - triggerThreshold:
                j += 1
                
                
            q0[nEvents] = sum(wave0Offset - wave0[i:j])
            q1[nEvents] = sum(wave1Offset - wave1[i:j])
            
            if (max(wave0Offset - wave0[i:j])> 1000.0) & (max(wave1Offset - wave1[i:j]) > 1000.0):

                
                
                q0Big[nBigEvents] = sum(wave0Offset - wave0[i:j])
                q1Big[nBigEvents] = sum(wave1Offset - wave1[i:j])
            
                nBigEvents += 1
                
            i += (j - i)
            nEvents +=1
            
    print('Finished processing signals!')
    
    print(nEvents)
    print(nBigEvents)
    #print(q0)
    #print(q1)
    
    
    #plt.hist(q0,bins=300,histtype='step')
    #plt.hist(q1,bins=300,histtype='step')
    plt.hist(q0,bins=128,histtype='step')
    plt.hist(q1,bins=128,histtype='step')
    plt.yscale('log')
    #plt.xlim((-10000,250000))
    plt.show()
    
    
    hist0,bins0 = np.histogram(q0,bins=nBins,range=(-10000,150000))
    hist1,bins1 = np.histogram(q1,bins=nBins,range=(-10000,150000))
    
    plt.step(bins0[:-1],hist0)
    plt.step(bins1[:-1],hist1)
    plt.yscale('log')
    #plt.xlim((-10000,250000))
    plt.show()
    
    
    plt.step(bins0[:int(nBins/2) ],hist0[:int(nBins/2)])
    plt.step(bins1[:int(nBins/2) ],hist1[:int(nBins/2)])
    #plt.xlim((-10000,250000))
    plt.show()
    
    
    mean0 = np.mean(q0)
    mean1 = np.mean(q1)
    
    rms0 = np.sqrt(np.mean(q0**2))
    rms1 = np.sqrt(np.mean(q1**2))
    
    
    
    print(mean0,rms0, rms0/mean0, np.std(q0), np.std(q0)/mean0)
    print(mean1,rms1, rms1/mean1, np.std(q1), np.std(q1)/mean1)
    
    
    outputdf0 = pd.DataFrame(columns = ['histogram'])
    outputdf1 = pd.DataFrame(columns = ['histogram'])
    
    for i in range(nBins):
        outputdf0.loc[i] = hist0[i]
        outputdf1.loc[i] = hist1[i]

    #write dataframe to .csv
    outputdf0.to_csv(runNum + '/ch_0_hist.csv', index=False)
    outputdf1.to_csv(runNum + '/ch_1_hist.csv', index=False)

    

        
if __name__ == '__main__':
    main('R375_spe_022924_0',4661,300,512)
