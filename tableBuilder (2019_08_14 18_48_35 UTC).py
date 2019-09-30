
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np


# In[2]:

def getFirstRows(array,lam,size):
    
    firstRow = [0]*(size)
    firstRow[0]=1+lam; firstRow[1]=-lam
    array = np.append(array,firstRow)

    secondRow = [0]*(size)
    secondRow[1]=-2*lam; secondRow[2]=1+5*lam; secondRow[3]=-4*lam; secondRow[4]=lam
    array = np.append(array,secondRow)
    
    return(array)

def addRows(array,i,lam,size):
    
    newRow = [0]*(size)
    newRow[i-3]=lam; newRow[i-2]=-4*lam; newRow[i-1]=1+6*lam; newRow[i]=-4*lam; newRow[i+1]=lam
    array = np.append(array,newRow)
    return(array)

def getLastRows(array,lam,size):
    
    almostLastRow = [0]*(size)
    almostLastRow[-2]=-2*lam; almostLastRow[-3]=1+5*lam; almostLastRow[-4]=-4*lam; almostLastRow[-5]=lam
    array = np.append(array,almostLastRow)
    
    lastRow = [0]*(size)
    lastRow[-1]=1+lam; lastRow[-2]=-lam
    array = np.append(array,lastRow)
    
    return(array)

def makeMatrix(lam,size):
    
    focs = np.array([])
    
    focs = getFirstRows(focs,lam,size)
    for i in range(3,(size-1)):
        focs = addRows(focs,i,lam,size)
    
    focs = getLastRows(focs,lam,size)
    focs = np.reshape(focs,(size,size))
    
    return(focs)


# call unRateDiffs to get a tuple of vectors (output gap, tau)
def unRateDiffs(y,focs):
    
    y = y.as_matrix()
    tau = np.linalg.solve(focs,y)
    
    return((y-tau,tau))
    #return(y-tau)


# In[3]:

# date incrementer function -- make a function to increase the month on every date by one
# this is to allow target income to be aligned with unemployment and CPI data from the prior month

# takes a date yyyy.mm and outputes a tuple (yyyy,mm)+1 month
def dateInc(date):
    
    year = np.floor(date)
    month = round((date%1)*100,2)
    
    if month != 12:
        month = month + 1
    else:
        month = 1
        year = year + 1
        
    return((int(year), int(month)))

# takes a date of the form yyyy.mm and outputs tuple (yyyy,mm)
def dateRep(date):
    
    year = np.floor(date)
    month = round((date%1)*100,2)

    return((int(year), int(month)))

# takes a dataframe of observed unemployment levels and value for lambda in the HP filer and outputs the dataframe with 
#      the output gap and natural rate of unemployment 
def natUnemployment(unrateData,lam,column,size):
    focs = makeMatrix(lam,size) #14400  
    unrateData['Tau'] = unRateDiffs(unrateData[column],focs)[1]
    unrateData['OutGap'] = unRateDiffs(unrateData[column],focs)[0]  
    return(unrateData)
    
# takes a dataframe with observed cpi and outputs the dataframe with a series for inflation. Inflation is yearly inflation.
def inflation(cpiData,column):    
    cpiData['Inflation'] = (cpiData[column]-cpiData[column].shift(12))/cpiData[column].shift(12)*100
    return(cpiData)

# imports target interest data, outputs a datafram with 3 columns:
#     date: (yyyy,mm)
#     target: target interest. date is incremented such that target from any month corresponds to inflation and unemployment
#           from the previous month
#     change: change in target from previous month to current

# takes parameters bank and trunc, where bank is a string of the desired bank, and trunc is a vector (startIndex,endIndex)
#     if one wishes to truncate the data
def getTarget(bank, trunc=(0,None)):
    
    if bank == 'BoC':
        target = pd.read_csv('rrm\\BoC\\Data\\Target.txt',delimiter = ' ',header = None)
        target.columns = ['Date','Day','Target','Change']
        target = target.drop(['Day'],axis=1)
        
    elif bank == 'BoE':
        target = pd.read_csv('rrm\\BoE\\Data\\UKTARGET.txt',delimiter = ' ',header = None)
        target.columns = ['Date','Target']
        target['Change'] = target['Target']-target['Target'].shift(1); target['Change'][0] = 0
        
    elif bank == 'Fed2':
        target = pd.read_csv('rrm\\Fed2\\Data\\Target2.txt',delimiter = ' ',header = None)
        target.columns = ['Date','Day','Target','Change']
        target = target.drop(['Day'],axis=1)
        
    target['Date'] = target['Date'].apply(dateRep)
    target['PriorTarget'] = target['Target'].shift(1)
    target['ChangeSign'] = target['Change'].apply(np.sign)
    target = target[trunc[0]:trunc[1]]
        
    return(target)

# imports cpi data, outputs dataframe with cpi and inflation information
# cpiType only applies to BoC data -- it selects which cpi is used for calculating inflation
# bank and trunc function as above
def getInflation(bank, cpiType='CoreCPI', trunc=(0,None)):
    
    if bank == 'BoC':
        cpi = pd.read_csv('rrm\\BoC\\Data\\CPI.txt',delimiter = ' ',header = None)
        cpi.columns = ['Date','TotalCPI','TotalCPI_SA','CoreCPI']
        cpi['Date'] = cpi['Date'].apply(dateInc)
        cpi = inflation(cpi,cpiType)
        cpi = cpi[trunc[0]:trunc[1]]
        
    elif bank == 'BoE':
        cpi = pd.read_csv('rrm\\BoE\\Data\\UKRPI.txt',delimiter = ' ',header = None)
        cpi.columns = ['Date','RPI']
        cpi['Date'] = cpi['Date'].apply(dateInc)
        cpi = inflation(cpi,'RPI')
        cpi = cpi[trunc[0]:trunc[1]]
        
    elif bank == 'Fed2':
        cpi = pd.read_csv('rrm\\Fed2\\Data\\CPIAUCNS.txt',delimiter = ' ',header = None)
        cpi.columns = ['Date','CPI']
        cpi['Date'] = cpi['Date'].apply(dateInc)
        cpi = inflation(cpi,'CPI')
        cpi = cpi[trunc[0]:trunc[1]]
        
    return (cpi)
        
# imports unemployment data, outputs dataframe with unemployment, natural rate, and output gap info
# lam is parameter for HP filter
# bank and trunc function as above
def getOutGap(bank, lam=129600, trunc=(0,None)):
    
    if bank == 'BoC':
        un = pd.read_csv('rrm\\BoC\\Data\\UNRATE.txt',delimiter = ' ',header = None)
        un.columns = ['Date','UnRate']
        un['Date'] = un['Date'].apply(dateInc)
        un = un[trunc[0]:trunc[1]]
        
    elif bank == 'BoE':
        un = pd.read_csv('rrm\\BoE\\Data\\UKUN.txt',delimiter = ' ',header = None)
        un.columns = ['Date','UnRate']
        un['Date'] = un['Date'].apply(dateInc)
        un = un[trunc[0]:trunc[1]]
        
    elif bank == 'Fed2':
        un = pd.read_csv('rrm\\Fed2\\Data\\UNRATE.txt',delimiter = ' ',header = None)
        un.columns = ['Date','UnRate']
        un['Date'] = un['Date'].apply(dateInc)
        un = un[trunc[0]:trunc[1]]
        
        
    un = natUnemployment(un,lam,'UnRate',un['Date'].size)
    return(un)

# get all data runs the functions above and returns tuple of dataframes (targetData,inflationData,unemploymentData)
def getAllData(bank, cpiType = 'CoreCPI', lam=129600, trunc=(0,None)):
    
    target = getTarget(bank, trunc)
    inf = getInflation(bank, cpiType, trunc)
    un = getOutGap(bank, lam, trunc)

    return((target,inf,un))

# merge all dataframes into a new dataframe that contains only data where all three imported dataframes share dates
def mergeData(target,inf,un,dropCols=[]):
    mergedTable = inf.merge(un, on = 'Date')
    mergedTable = mergedTable.merge(target, on = 'Date')
    mergedTable = mergedTable.drop(dropCols,axis=1)
    return(mergedTable)

# builds the final dataframe
# bank, cpiType, lam, and trunc function as above
# dropCols takes a list of column names to drop in the outputted table
def constructTable(bank, dropCols=[], cpiType='CoreCPI', lam=129600, trunc=(0,None)):
    
    target,inf,un = getAllData(bank, cpiType, lam, trunc)
    
    mergedTable = mergeData(target, inf, un, dropCols)    
    return(mergedTable)

