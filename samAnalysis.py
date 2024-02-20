#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 15:50:51 2024

@author: danielvalmassei
"""

#import mplhep as hep
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import pylandau #pyLandau provided at https://pypi.org/project/pylandau/ based on
                    #https://www.sciencedirect.com/science/article/pii/0010465584900857?via%3Dihub


def gauss(x, A, x0, sigma): 
    return A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def gaussFit(x, y, A, x0, sigma):
    
    coeff, pcov = curve_fit(gauss, x, y,
                            absolute_sigma=True,
                            p0=(A, x0, sigma))
        
    return coeff, pcov

def langauFit(x, y, mpv, sigma, eta, A):

    coeff, pcov = curve_fit(pylandau.langau, x, y,
                        absolute_sigma=True,
                        p0=(mpv, sigma, eta, A),                    
                        bounds=(-2.0,1000.0))
    
    return coeff, pcov

def main(runs,events,atten,triggerThreshold,secondThreshold,nBins,histEndpoint):
    '''
    Parameters
    ----------
    runs : list of strings
        Defines runs for analysis
    events : int
        Number of events in combined runs. Must be greater than the total 
        number of events
    atten : int
        Attenuation factor. atten=1 for no gain/attenuation.
    triggerThreshold : int
        Threshold for analysis trigger. Typically 500
    secondThreshold : int
        Secondary threshold. # of digitizer bins above offset for secondary trigger
    nBins : int
        # of bins for histograms
    histEndpoint : int/double
        endpoint for histogram in pC

    Returns
    -------
    None.

    '''
    
    #plt.style.use(hep.style.ALICE)
    
    triggerThreshold = triggerThreshold/4096/50 #convert digitizer bins to current
    
    print('Loading data. This could take a couple minutes. Please wait...')

    df0 = pd.read_csv(runs[0] + "/TR_0_0.txt", header = None)
    df1 = pd.read_csv(runs[0] + "/wave_0.txt", header = None)
    df2 = pd.read_csv(runs[0] + "/wave_1.txt", header = None)
    

    
    if len(runs) > 1:
        for i in range(len(runs)-1):
            
            df0 = pd.concat([df0, pd.read_csv(runs[i+1] + '/TR_0_0.txt', header = None)])
            df1 = pd.concat([df1, pd.read_csv(runs[i+1] + '/wave_0.txt', header = None)])
            df2 = pd.concat([df2, pd.read_csv(runs[i+1] + '/wave_1.txt', header = None)])
           


    
    print('Data loaded!')

    ###### Convert  digitizer bins to current #####  
    trigger = np.array(df0[0])/4096/50
    wave0 = np.array(df1[0])/4096/50
    ped0 = np.array(df2[0])/4096/50



    
    ##### Plot signals #####
    plt.plot(trigger,color='Blue',linewidth=1)
    plt.plot(wave0,color='Black',linewidth=1)
    plt.plot(ped0,color='Green',linewidth=1)

    plt.ylabel('current [A]')
    plt.xlabel('Sample # @ 5GS/s')
    
    plt.show()
    
    print('Integrating signals. This could take a moment. Please wait...')
    
    ##### get offsets for each signal #####
    
    triggerOffset = np.mean(trigger[0:50])
    
    wave0Offset = np.mean(wave0[-100:])
    ped0Offset = np.mean(ped0[-100:])

    
    
    print(wave0Offset)
    length = len(trigger)
    
    ##### set up some variables and arrays to store charge for each event #####
    nEvents = 0
    nBigEvents = 0
    
    q0 = np.zeros(events)
    pedQ = np.zeros(events)

    
    q0Big = np.zeros(events)
    pedQBig = np.zeros(events)

    ##### Integrate signals #####
    i = 0
    
    while i < length: #are we still within the uploaded signal?
        if trigger[i] > triggerOffset - triggerThreshold: #did the trigger go above threshold?
            i += 1
        
        else: 
            j = i + 1
            
            while trigger[j] < triggerOffset - triggerThreshold: 
                j += 1
                
            # a descrete integral is just a sum...   
            q0[nEvents] = sum(wave0Offset - wave0[i:j])/(5*10E9)*1E12/atten  #divide by samples/second and multiply by 1E12 to get pico-coulombs
            pedQ[nEvents] = sum(ped0Offset - ped0[i:j])/(5*10E9)*1E12

            
            if (max(wave0Offset - wave0[i:j])> secondThreshold/4096/50): #did the signal go above the secondary threshold?

                
                
                q0Big[nBigEvents] = sum(wave0Offset - wave0[i:j])/(5*10E9)*1E12/atten
                pedQBig[nBigEvents] = sum(ped0Offset - ped0[i:j])/(5*10E9)*1E12

            
                nBigEvents += 1
                
            i += (j - i)
            nEvents +=1
            
    print('Finished processing signals!')
    
    print(nEvents)
    print(nBigEvents)
    #print(q0_0)
    #print(q1)
    
    ##### trim off extra events in arrays #####
    q0 = np.trim_zeros(q0,'b')
    pedQ = np.trim_zeros(pedQ,'b')

    
    q0Big = np.trim_zeros(q0Big,'b')
    pedQBig = np.trim_zeros(pedQBig,'b')

    
    print(len(q0),len(q0Big))
    
    ##### Plot some histograms using pyplot #####
    
    
    #plt.hist(q0,bins=300,histtype='step')
    #plt.hist(q1,bins=300,histtype='step')
    plt.hist(q0,bins=128,histtype='step')
    plt.hist(pedQ,bins=128,histtype='step')
    plt.yscale('log')
    #plt.xlim((-10000,250000))
    plt.show()
    
    plt.hist(q0Big,bins=128,histtype='step')
    plt.hist(pedQBig,bins=128,histtype='step')
    plt.yscale('log')
    #plt.xlim((-10000,250000))
    plt.show()
    
    ##### Make our own histgrams #####
    hist0,bins0 = np.histogram(q0,bins=nBins,range=(-2,histEndpoint))
    hist0Big,bins0Big = np.histogram(q0Big,bins=nBins,range=(-2,histEndpoint))
    
    pedHist,pedHistBins = np.histogram(pedQ,bins=nBins,range=(-2,2))
    pedBigHist,pedBigHistBins = np.histogram(pedQBig,bins=nBins,range=(-2,2))


    
    plt.step(bins0[:-1],hist0)
    plt.step(pedHistBins[:-1],pedHist)
    plt.yscale('log')
    #plt.xlim((-10000,250000))
    plt.show()
    
    ##### compute some statistics #####
    mean0 = np.mean(q0Big)
    pedMean = np.mean(pedQBig)

    
    print(mean0)
    
    rms0 = np.sqrt(np.mean(q0Big**2))
    
    
    ##### fit the histograms #####
    
    #coeff, pcov = gaussFit(bins0Big[:-1],hist0Big,np.max(hist0Big),mean0,np.std(q0Big))
    #coeff1Big, pcov1Big = gaussFit(bins1Big[:-1],hist1Big,np.max(hist1Big),mean0,np.std(q1Big))

    
    #coeff1, pcov1 = langauFit(bins0[:-1],hist0,60000,0.5,45000,15)
    coeff0Big, pcov0Big = langauFit(bins0[:-1],hist0Big,mean0/6,np.std(q0Big)/15,0.1,np.max(hist0Big))
    
    
    
    '''
    plt.step(bins0[:int(nBins/2) ],hist0[:int(nBins/2)])
    plt.step(bins1[:int(nBins/2) ],hist1[:int(nBins/2)])
    '''
    plt.step(bins0[:-1],hist0Big)
    #plt.plot(bins0[:-1],gauss(bins0[:-1],*coeff))
    plt.plot(bins0Big[:-1],pylandau.langau(bins0Big[:-1],*coeff0Big))
    #plt.step(bins1[:-1],hist1)
    #plt.xlim((-10000,250000))
    plt.show()
    
    
    ##### Print #####
    
    #print(coeff)
    #print(coeff1)
    print(coeff0Big)
    print(mean0,rms0, rms0/mean0, np.std(q0Big), np.std(q0Big)/mean0)
    #print('Fit sigma/mean (gaussian): ' + str(coeff[2]/coeff[1]))
    print('Fit sigma/mean (langau): ' + str(coeff0Big[1]/coeff0Big[0]))

    
    ##### Compute residuals #####
    
    #resGauss = gauss(bins0Big[:-1],*coeff) - hist0Big 
    resLangau = pylandau.langau(bins0Big[:-1],*coeff0Big) - hist0Big
    
   #ss_resGauss = np.sum(resGauss**2)
    ss_resLangau = np.sum(resLangau**2)
    
    ss_tot = np.sum((hist0Big - np.mean(hist0Big))**2)
    
    print('R^2 (langau): ' + str(1 - (ss_resLangau/ss_tot)))
    #print('R^2 (gauss): ' + str(1 - (ss_resGauss/ss_tot)))
    
    

    plt.scatter(bins0Big[:-1],resLangau,s=10)
    #plt.scatter(bins0[:-1],resGauss,s=10)
    plt.show()


    ##### Make our final figure #####
    
    figure, axis = plt.subplots(2, 1, height_ratios=[4,1],sharex=True,figsize=(10,7))
    
    axis[0].step(pedBigHistBins[:-1],pedBigHist,linewidth=1,color='Pink',label='pedestal')
    axis[0].step(bins0Big[:-1],hist0Big,linewidth=1,color='Blue',label='ch.0')
    axis[0].plot(bins0Big[:-1],pylandau.langau(bins0Big[:-1],*coeff0Big),color='Black')

    
    #axis[0].set_yscale('log')
    #axis[0].set_ylim((0.1,800))
    axis[0].legend(fontsize='medium',frameon=True)
    axis[0].set_ylabel('# events',fontsize='medium',labelpad=2.0)
    axis[0].set_title(runs)


    
    axis[1].plot(bins0Big[:-1],np.zeros(len(bins0Big[:-1])),linewidth=1,color='Blue')
    #axis[1].scatter(bins0Big[:-1],resGauss,s=3,color='Red')
    axis[1].scatter(bins0Big[:-1],resLangau,s=3,color='Black')
    axis[1].set_ylabel('residuals',fontsize='medium',labelpad=2.0)
    axis[1].set_xlabel('pC',fontsize='medium',labelpad=2.0)
    
    plt.show()

    
    '''
    outputdf0 = pd.DataFrame(columns = ['histogram'])
    outputdf1 = pd.DataFrame(columns = ['histogram'])
    
    for i in range(nBins):
        outputdf0.loc[i] = hist0[i]
        outputdf1.loc[i] = hist1[i]

    
    #write dataframe to .csv
    outputdf0.to_csv(runNum + '/ch_0_hist.csv', index=False)
    outputdf1.to_csv(runNum + '/ch_1_hist.csv', index=False)
    '''
    #print('gain: ' + str(abs(coeff0Big[0]) * 10E-12 / (1.602*10E-19)))
    print('Peak Charge (ch.0): ' + str(abs(coeff0Big[0]) - pedMean) + ' pC')

    return figure

def onlyOnePlot(runs,events,atten,triggerThreshold,secondThreshold,nBins,histEndpoint):
    '''
    Parameters
    ----------
    runs : list of strings
        Defines runs for analysis
    events : int
        Number of events in combined runs. Must be greater than the total 
        number of events
    atten : int
        Attenuation factor. atten=1 for no gain/attenuation.
    triggerThreshold : int
        Threshold for analysis trigger. Typically 500
    secondThreshold : int
        Secondary threshold. # of digitizer bins above offset for secondary trigger
    nBins : int
        # of bins for histograms
    histEndpoint : int/double
        endpoint for histogram in pC

    Returns
    -------
    None.

    '''
    
    #plt.style.use(hep.style.ALICE)
    
    triggerThreshold = triggerThreshold/4096/50 #convert digitizer bins to current
    
    #print('Loading data. This could take a couple minutes. Please wait...')

    df0 = pd.read_csv(runs[0] + "/TR_0_0.txt", header = None)
    df1 = pd.read_csv(runs[0] + "/wave_0.txt", header = None)
    df2 = pd.read_csv(runs[0] + "/wave_1.txt", header = None)
    

    
    if len(runs) > 1:
        for i in range(len(runs)-1):
            
            df0 = pd.concat([df0, pd.read_csv(runs[i+1] + '/TR_0_0.txt', header = None)])
            df1 = pd.concat([df1, pd.read_csv(runs[i+1] + '/wave_0.txt', header = None)])
            df2 = pd.concat([df2, pd.read_csv(runs[i+1] + '/wave_1.txt', header = None)])
           


    
    #print('Data loaded!')

    ###### Convert  digitizer bins to current #####  
    trigger = np.array(df0[0])/4096/50
    wave0 = np.array(df1[0])/4096/50
    ped0 = np.array(df2[0])/4096/50



    
    ##### Plot signals #####
    #plt.plot(trigger,color='Blue',linewidth=1)
    #plt.plot(wave0,color='Black',linewidth=1)
    #plt.plot(ped0,color='Green',linewidth=1)

    #plt.ylabel('current [A]')
    #plt.xlabel('Sample # @ 5GS/s')
    
    #plt.show()
    
    #print('Integrating signals. This could take a moment. Please wait...')
    
    ##### get offsets for each signal #####
    
    triggerOffset = np.mean(trigger[0:50])
    
    wave0Offset = np.mean(wave0[-100:])
    ped0Offset = np.mean(ped0[-100:])

    
    
    #print(wave0Offset)
    length = len(trigger)
    
    ##### set up some variables and arrays to store charge for each event #####
    nEvents = 0
    nBigEvents = 0
    
    q0 = np.zeros(events)
    pedQ = np.zeros(events)

    
    q0Big = np.zeros(events)
    pedQBig = np.zeros(events)

    ##### Integrate signals #####
    i = 0
    
    while i < length: #are we still within the uploaded signal?
        if trigger[i] > triggerOffset - triggerThreshold: #did the trigger go above threshold?
            i += 1
        
        else: 
            j = i + 1
            
            while trigger[j] < triggerOffset - triggerThreshold: 
                j += 1
                
            # a descrete integral is just a sum...   
            q0[nEvents] = sum(wave0Offset - wave0[i:j])/(5*10E9)*1E12/atten  #divide by samples/second and multiply by 1E12 to get pico-coulombs
            pedQ[nEvents] = sum(ped0Offset - ped0[i:j])/(5*10E9)*1E12

            
            if (max(wave0Offset - wave0[i:j])> secondThreshold/4096/50): #did the signal go above the secondary threshold?

                
                
                q0Big[nBigEvents] = sum(wave0Offset - wave0[i:j])/(5*10E9)*1E12/atten
                pedQBig[nBigEvents] = sum(ped0Offset - ped0[i:j])/(5*10E9)*1E12

            
                nBigEvents += 1
                
            i += (j - i)
            nEvents +=1
            
    #print('Finished processing signals!')
    
    print(f'nEvents:{nEvents}')
    print(f'nBigEvents:{nBigEvents}')
    #print(q0_0)
    #print(q1)
    
    ##### trim off extra events in arrays #####
    q0 = np.trim_zeros(q0,'b')
    pedQ = np.trim_zeros(pedQ,'b')

    
    q0Big = np.trim_zeros(q0Big,'b')
    pedQBig = np.trim_zeros(pedQBig,'b')

    
    #print(len(q0),len(q0Big))
    
    ##### Plot some histograms using pyplot #####
    
    
    #plt.hist(q0,bins=300,histtype='step')
    #plt.hist(q1,bins=300,histtype='step')
    #plt.hist(q0,bins=128,histtype='step')
    #plt.hist(pedQ,bins=128,histtype='step')
    #plt.yscale('log')
    #plt.xlim((-10000,250000))
    #plt.show()
    
    #plt.hist(q0Big,bins=128,histtype='step')
    #plt.hist(pedQBig,bins=128,histtype='step')
    #plt.yscale('log')
    #plt.xlim((-10000,250000))
    #plt.show()
    
    ##### Make our own histgrams #####
    hist0,bins0 = np.histogram(q0,bins=nBins,range=(-2,histEndpoint))
    hist0Big,bins0Big = np.histogram(q0Big,bins=nBins,range=(-2,histEndpoint))
    
    pedHist,pedHistBins = np.histogram(pedQ,bins=nBins,range=(-2,2))
    pedBigHist,pedBigHistBins = np.histogram(pedQBig,bins=nBins,range=(-2,2))


    
    #plt.step(bins0[:-1],hist0)
    #plt.step(pedHistBins[:-1],pedHist)
    #plt.yscale('log')
    #plt.xlim((-10000,250000))
    #plt.show()
    
    ##### compute some statistics #####
    mean0 = np.mean(q0Big)
    pedMean = np.mean(pedQBig)

    
    #print(mean0)
    
    rms0 = np.sqrt(np.mean(q0Big**2))
    
    
    ##### fit the histograms #####
    
    #coeff, pcov = gaussFit(bins0Big[:-1],hist0Big,np.max(hist0Big),mean0,np.std(q0Big))
    #coeff1Big, pcov1Big = gaussFit(bins1Big[:-1],hist1Big,np.max(hist1Big),mean0,np.std(q1Big))

    
    #coeff1, pcov1 = langauFit(bins0[:-1],hist0,60000,0.5,45000,15)
    coeff0Big, pcov0Big = langauFit(bins0[:-1],hist0Big,mean0/6,np.std(q0Big)/15,0.1,np.max(hist0Big))
    
    
    
    '''
    plt.step(bins0[:int(nBins/2) ],hist0[:int(nBins/2)])
    plt.step(bins1[:int(nBins/2) ],hist1[:int(nBins/2)])
    '''
    #plt.step(bins0[:-1],hist0Big)
    #plt.plot(bins0[:-1],gauss(bins0[:-1],*coeff))
    #plt.plot(bins0Big[:-1],pylandau.langau(bins0Big[:-1],*coeff0Big))
    #plt.step(bins1[:-1],hist1)
    #plt.xlim((-10000,250000))
    #plt.show()
    
    
    ##### Print #####
    
    #print(coeff)
    #print(coeff1)
    #print(coeff0Big)
    print(mean0,rms0, rms0/mean0, np.std(q0Big), np.std(q0Big)/mean0)
    #print('Fit sigma/mean (gaussian): ' + str(coeff[2]/coeff[1]))
    print('Fit sigma/mean (langau): ' + str(coeff0Big[1]/coeff0Big[0]))

    
    ##### Compute residuals #####
    
    #resGauss = gauss(bins0Big[:-1],*coeff) - hist0Big 
    resLangau = pylandau.langau(bins0Big[:-1],*coeff0Big) - hist0Big
    
   #ss_resGauss = np.sum(resGauss**2)
    ss_resLangau = np.sum(resLangau**2)
    
    ss_tot = np.sum((hist0Big - np.mean(hist0Big))**2)
    
    print('R^2 (langau): ' + str(1 - (ss_resLangau/ss_tot)))
    #print('R^2 (gauss): ' + str(1 - (ss_resGauss/ss_tot)))
    
    

    #plt.scatter(bins0Big[:-1],resLangau,s=10)
    #plt.scatter(bins0[:-1],resGauss,s=10)
    #plt.show()


    ##### Make our final figure #####
    
    figure, axis = plt.subplots(2, 1, height_ratios=[4,1],sharex=True,figsize=(10,7))
    
    axis[0].step(pedBigHistBins[:-1],pedBigHist,linewidth=1,color='Pink',label='pedestal')
    axis[0].step(bins0Big[:-1],hist0Big,linewidth=1,color='Blue',label='ch.0')
    axis[0].plot(bins0Big[:-1],pylandau.langau(bins0Big[:-1],*coeff0Big),color='Black')

    
    #axis[0].set_yscale('log')
    #axis[0].set_ylim((0.1,800))
    axis[0].legend(fontsize='medium',frameon=True)
    axis[0].set_ylabel('# events',fontsize='medium',labelpad=2.0)
    axis[0].set_title(runs)


    
    axis[1].plot(bins0Big[:-1],np.zeros(len(bins0Big[:-1])),linewidth=1,color='Blue')
    #axis[1].scatter(bins0Big[:-1],resGauss,s=3,color='Red')
    axis[1].scatter(bins0Big[:-1],resLangau,s=3,color='Black')
    axis[1].set_ylabel('residuals',fontsize='medium',labelpad=2.0)
    axis[1].set_xlabel('pC',fontsize='medium',labelpad=2.0)
    
    #plt.show()

    
    '''
    outputdf0 = pd.DataFrame(columns = ['histogram'])
    outputdf1 = pd.DataFrame(columns = ['histogram'])
    
    for i in range(nBins):
        outputdf0.loc[i] = hist0[i]
        outputdf1.loc[i] = hist1[i]

    
    #write dataframe to .csv
    outputdf0.to_csv(runNum + '/ch_0_hist.csv', index=False)
    outputdf1.to_csv(runNum + '/ch_1_hist.csv', index=False)
    '''
    #print('gain: ' + str(abs(coeff0Big[0]) * 10E-12 / (1.602*10E-19)))
    print('Peak Charge (ch.0): ' + str(abs(coeff0Big[0]) - pedMean) + ' pC')

    return figure, coeff0Big, pedMean

        
if __name__ == '__main__':
    main(['sam_020824_0'],10000,1.0,500,200,256,50) #'sam_012224_0','sam_012324_0','sam_012524_0','sam_012624_0'
