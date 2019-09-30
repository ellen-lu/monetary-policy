# %%

# coding: utf-8

# %%

from tableBuilder import *
import math
from scipy.stats import norm
from scipy.stats import logistic
from scipy import special
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from debug import ipsh


# %%
class Bank():
    
    def __init__(self, bank, dropCols=[], cpiType='TotalCPI', lam=129600, trunc=(0,None),F=logistic.cdf,f=logistic.pdf,utility=None):
        
        self.bank = bank
        self.lam = lam
        self.trunc = trunc
        self.table = constructTable(bank,dropCols,cpiType,lam,trunc)
        self.cpi = cpiType
        self.F = F
        self.f = f
        self.T1 = self.table['Date'].loc[self.table['ChangeSign']==-1.0].size
        self.T3 = self.table['Date'].loc[self.table['ChangeSign']==1.0].size
        self.T = self.table['Date'].size
        
    def __repr__(self):
        return(str(self.table))
    
    def __str__(self):
        return('I am the ' + self.bank)  
            
    def dropCols(self, cols, axis=1):
        self.table = self.table.drop(cols,axis=axis)
        
        if axis==0:
            self.T1 = self.table['Date'].loc[self.table['ChangeSign']==-1.0].size
            self.T3 = self.table['Date'].loc[self.table['ChangeSign']==1.0].size
            self.T = self.table['Date'].size
        
    def addCols(self, cols, vectors):
        l = len(cols)
        for i in range(l):
            self.table[cols[i]] = vectors[i]
            
    '''
    
    All code relating to the gradient ascent MLE can be found below.
    
    '''
        
    def calculateScoringMatrix(self,gradient):
        
        score = np.array(self.gradNash(theta[0],theta[1],theta[2],theta[3],theta[4],theta[5],theta[6]))
        return np.outer(score,score)

    def gradAscent(self,alpha,estimate, gradType, llType, conv = 0.0001, ll=False, est=False, worse=0.5, better=1.005):
        
        if est:
            print(estimate)

        gradient = gradType(estimate)
        update = estimate+alpha*gradient

        priorLL = llType(estimate)
        newLL = llType(update)
        
        improve = newLL>priorLL
        
        if improve:    
            if ll:
                print(priorLL)

            if newLL-priorLL<=conv:     #   0.0000001:
                return update

            return(self.gradAscent(better*alpha,update,gradType,llType,conv,ll,est,worse,better))

        return self.gradAscent(worse*alpha,estimate,gradType,llType,conv,ll,est,worse,better)

            
    '''
    
    All fun features can be found below!
    
    '''
    
    def unCorr(self):
        
        table = getOutGap(self.bank,lam=self.lam,trunc=self.trunc)
        
        unShift = table['OutGap'].shift(1).as_matrix()[1:]
        un = table['OutGap'].as_matrix()[1:]
        return(np.corrcoef(unShift,un))
    
    def infCorr(self):
        
        table = getInflation(self.bank, cpiType=self.cpi, trunc=self.trunc)

        infShift = table['Inflation'].shift(1).as_matrix()[1:]
        inf = table['Inflation'].as_matrix()[1:]
        return(np.corrcoef(infShift,inf))
    
    def graphUn(self,lams=[129600],trunc=(0,None),numBwYears=10):
        
        fig, ax = plt.subplots()
        table = getOutGap(self.bank,lams[0],trunc)
        bins = range(0, int(np.floor(len(table['UnRate'])/(numBwYears*12))))
        ax.plot(table['UnRate'],color='M')

        for lam in lams:
            table = getOutGap(self.bank,lam,trunc)
            ax.plot(table['Tau'],label=str(lam))

        ticks = np.array([numBwYears*12*i for i in bins])
        labels = np.array([table['Date'][tick][0] for tick in ticks])
        plt.xticks(ticks, labels)  # Set locations and labels
        plt.legend()
        plt.show()
        
    def plotPred(self,a,b,c,sigma,ampk=None):
        
        if ampk == None:
            intRates = self.table['Target']
            self.getZFric(a,b,c,sigma)
            estimates = a+b*self.table['Inflation']+c*self.table['OutGap']+np.random.normal(0,sigma,1)
            fig, ax = plt.subplots()
            ax.plot(intRates)
            ax.plot(estimates)
            plt.show()
            
        else:
            intRates = self.table['Target']
            self.getZCon(a,theta[1],theta[2],theta[3],sigma)
            estimates = (a+theta[1])/2+b*self.table['Inflation']+c*self.table['OutGap']+np.random.normal(0,sigma,1)
            fig, ax = plt.subplots()
            ax.plot(intRates)
            ax.plot(estimates)
            plt.show() 

class Consensus(Bank):
    
    def getZCon(self,theta,shift=True):
        
        if shift:
            self.table['zmmk']=(self.table['Target'].shift(-1)-theta[0]-theta[2]*self.table['Inflation']                            -theta[3]*self.table['OutGap'])/theta[4]
            self.table['zmpk']=(self.table['Target'].shift(-1)-theta[1]-theta[2]*self.table['Inflation']                            -theta[3]*self.table['OutGap'])/theta[4]
        else:
            self.table['zmmk']=(self.table['Target']-theta[0]-theta[2]*self.table['Inflation']                -theta[3]*self.table['OutGap'])/theta[4]
            self.table['zmpk']=(self.table['Target']-theta[1]-theta[2]*self.table['Inflation']                            -theta[3]*self.table['OutGap'])/theta[4]
        
    def llc(self,theta):
        
        self.getZCon(theta)
        
        termOne = -(self.T1+self.T3)*math.log(theta[4])
        sumOne = ((self.table['Target']-theta[1]-theta[2]*self.table['Inflation']                            -theta[3]*self.table['OutGap'])/theta[4]).where(self.table['ChangeSign']==-1).                            apply(norm.pdf).apply(math.log).sum()
        sumTwo = (self.table['zmmk'].where(self.table['ChangeSign']==0).apply(norm.cdf)-self.table['zmpk']                  .where(self.table['ChangeSign']==0).apply(norm.cdf)).apply(math.log).sum()
        sumThree = ((self.table['Target']-theta[0]-theta[2]*self.table['Inflation']                            -theta[3]*self.table['OutGap'])/theta[4]).where(self.table['ChangeSign']==1).apply(norm.pdf).apply(math.log).sum()

        return(termOne+sumOne+sumTwo+sumThree)

    def gradientConsensusFirstTerm(self,theta):
        
        self.getZCon(theta,shift=False)

        gradA = self.table['zmpk']/theta[4]
        gradB = self.table['Inflation']*self.table['zmpk']/theta[4]
        gradC = self.table['OutGap']*self.table['zmpk']/theta[4]
        gradSigma = self.table['zmpk']**2/theta[4] ## double check the negative here

        likelihoodGradient = (gradA,gradB,gradC,gradSigma)

        return(likelihoodGradient)

    def gradientConsensusSecondTerm(self,theta):
        
        self.getZCon(theta)

        commonDenom = theta[4]*(self.table['zmmk'].apply(norm.cdf)-self.table['zmpk'].apply(norm.cdf))
        commonDiff = self.table['zmmk'].apply(norm.pdf)-self.table['zmpk'].apply(norm.pdf)

        gradAmpk = self.table['zmpk'].apply(norm.pdf)/commonDenom
        gradAmmk = -self.table['zmmk'].apply(norm.pdf)/commonDenom
        gradB = -self.table['Inflation']*commonDiff/commonDenom
        gradC = -self.table['OutGap']*commonDiff/commonDenom
        gradSigma = -(self.table['zmmk']*self.table['zmmk'].apply(norm.pdf)                     -self.table['zmpk']*self.table['zmpk'].apply(norm.pdf))/commonDenom

        likelihoodGradient = (gradAmmk,gradAmpk,gradB,gradC,gradSigma)

        return(likelihoodGradient)

    def gradientConsensusThirdTerm(self,theta):
        
        self.getZCon(theta,shift=False)

        gradA = self.table['zmmk']/theta[4]
        gradB = self.table['Inflation']*self.table['zmmk']/theta[4]
        gradC = self.table['OutGap']*self.table['zmmk']/theta[4]
        gradSigma = self.table['zmmk']**2/theta[4] ## double check the negative here
        
        likelihoodGradient = (gradA,gradB,gradC,gradSigma)

        return(likelihoodGradient)

    def gradCon(self,theta):
        
        self.getZCon(theta)

        gradAmmk = (self.gradientConsensusThirdTerm(theta)[0].where(self.table['ChangeSign']==1).sum()                    +self.gradientConsensusSecondTerm(theta)[0].where(self.table['ChangeSign']==0).sum())

        gradAmpk = (self.gradientConsensusFirstTerm(theta)[0].where(self.table['ChangeSign']==-1).sum()                    +self.gradientConsensusSecondTerm(theta)[1].where(self.table['ChangeSign']==0).sum())

        gradB = (self.gradientConsensusFirstTerm(theta)[1].where(self.table['ChangeSign']==-1).sum()                 +self.gradientConsensusSecondTerm(theta)[2].where(self.table['ChangeSign']==0).sum()                 +self.gradientConsensusThirdTerm(theta)[1].where(self.table['ChangeSign']==1).sum())

        gradC = (self.gradientConsensusFirstTerm(theta)[2].where(self.table['ChangeSign']==-1).sum()                 +self.gradientConsensusSecondTerm(theta)[3].where(self.table['ChangeSign']==0).sum()                 +self.gradientConsensusThirdTerm(theta)[2].where(self.table['ChangeSign']==1).sum())

        gradSigma = -(self.T1+self.T3)/theta[4]                    +(self.gradientConsensusFirstTerm(theta)[3].where(self.table['ChangeSign']==-1).sum()                    +self.gradientConsensusSecondTerm(theta)[4].where(self.table['ChangeSign']==0).sum()                    +self.gradientConsensusThirdTerm(theta)[3].where(self.table['ChangeSign']==1).sum())

        #print(gradientConsensusFirstTerm(theta)[3])

        gradient = (gradAmmk,gradAmpk,gradB,gradC,gradSigma)
        return(np.array(gradient))
    
    def compareGradToLLC(self,theta,epsilon=0.00001):
        
        for i in range(0,5):
            beta = theta[:]
            beta[i] = beta[i]+epsilon
            print(self.gradCon(theta)[i])
            print(((self.llc(beta)-self.llc(theta))/epsilon))
            
    def checkGradImprovingCon(self,theta,epsilon=0.00001):
        
        good = True
        llh = self.llc(theta)
        
        for i in range(0,5):
            blarp = theta[:]
            blarp[i] = blarp[i]+epsilon*self.gradCon(theta)[i]
            if (self.llc(blarp)<llh):
                good = False
                print(i)
                
        if good:
            print("You're all good!")
            
    def gradAscentCon(self,alpha,estimate, conv = 0.0001, ll=False, est=False, worse=0.5, better=1.005):
        
        if est:
            print(estimate)

        gradient = self.gradCon(estimate)
        update = estimate+alpha*gradient

        priorLL = self.llc(estimate)
        newLL = self.llc(update)
        
        improve = newLL>priorLL
        
        if improve:    
            if ll:
                print(priorLL)

            if newLL-priorLL<=conv:     #   0.0000001:
                return update

            return(self.gradAscentCon(better*alpha,update,conv,ll,est,worse,better))

        return self.gradAscentCon(worse*alpha,estimate,conv,ll,est,worse,better) 
            
class Frictionless(Bank):
    
            
    def getZFric(self,theta):
        self.table['fricZ'] = (self.table['Target']-theta[0]-theta[1]*self.table['Inflation']-theta[2]*self.table['OutGap'])/theta[3]
        
    def llf(self,theta):
        self.getZFric(theta)
        return(-self.T*math.log(theta[3])+self.table['fricZ'].apply(norm.pdf).apply(math.log).sum())
    
    def gradFric(self,theta):
    
        self.getZFric(theta)

        gradA = 1/theta[3]*(self.table['fricZ']).sum()
        gradB = 1/theta[3]*(self.table['fricZ']*self.table['Inflation']).sum()
        gradC = 1/theta[3]*(self.table['fricZ']*self.table['OutGap']).sum()
        gradSigma = -self.T/theta[3]+(1/(theta[3])*self.table['fricZ']**2).sum()
    
        return(np.array((gradA,gradB,gradC,gradSigma)))
    
    def OLSFric(self,trunc=(0,None)):
        
        size = self.table['Inflation'][trunc[0]:trunc[1]].size
        
        X = np.array([1]*size)
        X = pd.Series(X).to_frame()
        X.columns = ['Constant']
        
        inf = self.table['Inflation'][trunc[0]:trunc[1]].reset_index()
        un = self.table['OutGap'][trunc[0]:trunc[1]].reset_index()

        X['Inflation'] = inf['Inflation']
        X['OutGap'] = un['OutGap']
        X_frame = X.copy()
        X = X.values

        Y = self.table['Target'][trunc[0]:trunc[1]]
        Y = Y.values
        beta = np.matmul(np.linalg.inv(np.matmul(np.transpose(X),X)),np.matmul(np.transpose(X),Y))
        
        M = np.identity(X_frame['Constant'].size)-np.matmul(X,np.matmul(np.linalg.inv(np.matmul(np.transpose(X),X)),np.transpose(X)))
        sig = 1/(X_frame['Constant'].size)*np.matmul(np.transpose(Y),np.matmul(M,Y))
        sig = math.sqrt(sig)
        
        beta = np.append(beta,[sig])
        
        dev = np.sqrt(sig*np.linalg.inv(np.matmul(np.transpose(X),X)))
        
        #print(dev)
        
        print('The OLS estimate is:')
        return(beta) 
    
    '''def gradAscentFric(self,alpha,estimate, conv = 0.0001, ll=False, est=False, worse=0.5, better=1.005):
        
        if est:
            print(estimate)

        gradient = self.gradFric(estimate)
        update = estimate+alpha*gradient

        priorLL = self.llf(estimate)
        newLL = self.llcf(update)
        
        improve = newLL>priorLL
        
        if improve:    
            if ll:
                print(priorLL)

            if newLL-priorLL<=conv:     #   0.0000001:
                return update

            return(self.gradAscentFric(better*alpha,update,conv,ll,est,worse,better))

        return self.gradAscentFric(worse*alpha,estimate,conv,ll,est,worse,better) '''
    
    def makeGrid(self,theta,binSize=2,pC_binSize=0.4,numDraws=40):
        
        a=theta[0]; b=theta[1]; c=theta[2];sigma=theta[3]
        
        params = np.array([a,b,c,sigma])
        grids = []
        for i in range(len(theta)):
            grids.append([])
            for j in range(numDraws):
                if i==3:
                    grids[i].append(random.uniform(max(params[i]-binSize,0),params[i]+binSize))
                else:
                    grids[i].append(random.uniform(params[i]-binSize,params[i]+binSize))
                    
        ipsh()
            
        self.grid = pd.DataFrame(np.transpose(np.array(grids)))
        self.grid.columns = ['a','b','c','sigma']
        return(self.grid)
    
    def computeGridLL(self,theta,binSize=2,pC_binSize=0.4,numDraws=40):
        
        self.makeGrid(theta,binSize,pC_binSize,numDraws)
        params = {0:'a',1:'b',2:'c',3:'sigma'}
        
        def makeThetaLL(row):
            rowTheta = row.values
            try:
                return self.llf(rowTheta)
            except:
                return None
                
        self.grid['LL'] = self.grid.apply(lambda row: makeThetaLL(row), axis=1)
        return(self.grid)
    
    def checkParams(self,theta):
        
        if theta[3]<0:
            theta[3]=0

        return theta
    
    def gradAscentFric(self,alpha,estimate,conv = 0.0001,ll=False,est=False,n=1,m=0,v=0,beta_1=0.9,beta_2=0.999):
    
        anotherEpsilon = 10e-8
        estimate=self.checkParams(estimate)
        
        if n%10==1 and n<70:
            self.computeGridLL(estimate,numDraws=10,binSize=2*random.uniform(.25,1),pC_binSize=0.4*random.uniform(0.25,1))
            row = self.grid['LL'].argmax()
            newEstimate = self.grid.loc[row].copy().values
            newEstimate = newEstimate[0:-1]
            if self.grid['LL'].max()>self.llf(estimate):
                estimate = newEstimate
        
        if est:
            print(estimate)
        
        # adapted from https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c
        gradient = self.gradFric(estimate)
        m = beta_1 * m + (1 - beta_1) * gradient
        m_tild = beta_1*m + (1-beta_1)*gradient
        v = beta_2 * v + (1 - beta_2) * np.power(gradient, 2)
        m_hat = m_tild / (1 - np.power(beta_1, n))
        v_hat = v / (1 - np.power(beta_2, n))
        w = alpha*m_hat / (np.sqrt(v_hat) + anotherEpsilon)

        update = estimate+w
        update = self.checkParams(update)
        
        priorLL = self.llf(estimate)
        newLL = self.llf(update)

        improve = newLL>priorLL

        n+=1

        if ll:
            print(newLL)

        if improve:    
            if abs(newLL-priorLL)<=conv:
                return update

        return(self.gradAscentFric(alpha,update,conv,ll,est,n,m,v))
    
    def compareGradToLLF(self,theta,epsilon=0.00001):
        
        for i in range(0,4):
            beta = theta[:]
            beta[i] = beta[i]+epsilon
            print(self.gradFric(theta)[i])
            print(((self.llf(beta)-self.llf(theta))/epsilon))
            
    def checkGradImprovingFric(self,theta,epsilon=0.00001):
        
        good = True
        llh = self.llf(theta)
        
        for i in range(0,4):
            blarp = theta[:]
            blarp[i] = blarp[i]+epsilon*self.gradFric(theta)[i]
            if (self.llf(blarp)<llh):
                good = False
                print(i)
                
        if good:
            print("You're all good!") 
    
'''class Nash(Bank):

    def I(self,a,b,c):
        return a+b*self.table['Inflation']+c*self.table['OutGap']

    def addZLZR(self,theta):

        i = self.table['PriorTarget']
        self.table['zL'] = i-self.I(theta[1],theta[2],theta[3])
        self.table['zR'] = i-self.I(theta[0],theta[2],theta[3])
        
    def addILIR(self,theta):
        self.table['IL'] = self.I(theta[1],theta[2],theta[3])+self.table['nashZ']
        self.table['IR'] = self.I(theta[0],theta[2],theta[3])+self.table['nashZ']

    def guessRho(self,theta,F=logistic.cdf):
        
        self.addZLZR(theta)
        rho = pd.Series([1-F(theta[4])]*self.table['Date'].size)
        self.table['rho'] = rho.copy()

        return rho
    
    def giveZ(self,theta,plot=False,new=False):
        
        comm = 1/2*(self.table['Target']-self.table['PriorTarget'])
        i = self.table['Target']
      
        B = 2*((self.I(theta[1],theta[2],theta[3])+self.I(theta[0],theta[2],theta[3]))/2-i+1/2*comm)
        C = (self.I(theta[1],theta[2],theta[3])-i)*(self.I(theta[0],theta[2],theta[3])-i)+comm*(self.table['rho']*\
                    self.I(theta[1],theta[2],theta[3])+(1-self.table['rho'])*self.I(theta[0],theta[2],theta[3])-i)

        if plot:
            self.addZLZR(theta)
            x = np.arange(-2,5,.1)
            y = x**2+B[1]*x+C[1]
            zero = np.zeros(x.size)
            fig, ax = plt.subplots()
            ax.plot(x,y)
            ax.plot(x,zero)
            ax.axvline(self.table['zL'][1],color='r')
            ax.axvline(self.table['zR'][1],color='m')
            plt.show()
        
        if new:
            self.table['newZ'] = pd.Series([0]*self.table['Date'].size)
            self.table['newZ'].loc[self.table['ChangeSign']==1] = (-B+(B**2-4*C)**(1/2))/2
            self.table['newZ'].loc[self.table['ChangeSign']==-1] = (-B-(B**2-4*C)**(1/2))/2
            return self.table['newZ']
                
        else:
            self.table['nashZ'] = pd.Series([0]*self.table['Date'].size)
            self.table['nashZ'].loc[self.table['ChangeSign']==1] = (-B+(B**2-4*C)**(1/2))/2
            self.table['nashZ'].loc[self.table['ChangeSign']==-1] = (-B-(B**2-4*C)**(1/2))/2
            return self.table['nashZ']
        
        
    def giveRho(self,theta,new=False,F=logistic.cdf):
        
        i = self.table['Target']
        self.addZLZR(theta)
        
        if new:
            rho = ((self.I(theta[4],theta[2],theta[3])+self.table['newZ']-i)/theta[5]).apply(F)
            self.table['newRho'] = rho.copy()
            
        else:
            rho = ((self.I(theta[4],theta[2],theta[3])+self.table['nashZ']-i)/theta[5]).apply(F)
            self.table['rho'] = rho.copy()     
        
        return rho
    
    
    def getRhoZHelper(self,theta,conv,printN,et,n=0):
        
        n+=1

        self.giveZ(theta,new=True)
        self.giveRho(theta,new=True)
                
        rhoConv = all((self.table['rho']-self.table['newRho']).loc[self.table['ChangeSign']!=0].apply(abs)<conv)
        zConv = all((self.table['nashZ']-self.table['newZ']).loc[self.table['ChangeSign']!=0].apply(abs)<conv)
    
        self.table['nashZ'] = self.table['newZ'].copy(); self.table = self.table.drop(['newZ'],axis=1)
        self.table['rho'] = self.table['newRho'].copy(); self.table = self.table.drop(['newRho'],axis=1)
        
        if (rhoConv,zConv) == (True,True):
            if printN:
                print(n)
            return self.table
            
        return self.getRhoZHelper(theta,conv,printN,et,n=n)
        
    
    def getRhoZ(self,theta,conv=0.0001,printN=False,et=False):
    
        self.guessRho(theta)
        self.giveZ(theta)
        self.giveRho(theta)
        self.getRhoZHelper(theta,conv,printN,et) 
        self.addZLZR(theta)
        self.addILIR(theta)
        return self.table
    
    def lln(self,theta):

        T = self.table['Date'].where(self.table['ChangeSign']!=0).size

        self.getRhoZ(theta)

        sumOne = -1/(2*theta[6]**2)*(self.table['nashZ'].where(self.table['ChangeSign']!=0)**2).sum()
        sumTwo =  ((self.table['zR']/theta[6]).where(self.table['ChangeSign']==0).apply(norm.cdf)\
                -(self.table['zL']/theta[6]).where(self.table['ChangeSign']==0).apply(norm.cdf)).apply(math.log).sum()

        return(-T*math.log(theta[6])+sumOne+sumTwo)


    def gradNashFirst(self,theta,F=logistic.cdf,f=logistic.pdf): 

        i = self.table['Target']

        comm = 1/2*(self.table['Target']-self.table['PriorTarget'])
        den = 2*(self.table['nashZ']+(self.I(theta[1],theta[2],theta[3],)+self.I(theta[0],theta[2],theta[3]))/2-i)+\
                comm+comm/theta[5]*(theta[1]-theta[0])*f((self.I(theta[4],theta[2],theta[3])+self.table['nashZ']-i)/theta[5])

        dZdaR = -(self.table['nashZ']+(self.I(theta[1],theta[2],theta[3])-i)+comm*(1-self.table['rho']))/den
        dZdaL = -(self.table['nashZ']+(self.I(theta[0],theta[2],theta[3])-i)+comm*self.table['rho'])/den
        dZdMu = -(comm/theta[5]*(theta[1]-theta[0])*f((self.I(theta[4],theta[2],theta[3])+self.table['nashZ']-i)/theta[5]))/den
        dZdS = (comm*(theta[1]-theta[0])*((self.I(theta[4],theta[2],theta[3])+self.table['nashZ']-i)/theta[5]**2)*\
                f((self.I(theta[4],theta[2],theta[3])+self.table['nashZ']-i)/theta[5]))/den

        gradaR = -(self.table['nashZ']*dZdaR)/theta[6]**2
        gradaL = -(self.table['nashZ']*dZdaL)/theta[6]**2
        gradB = (self.table['nashZ']*self.table['Inflation'])/theta[6]**2
        gradC = (self.table['nashZ']*self.table['OutGap'])/theta[6]**2
        gradMu = -(self.table['nashZ']*dZdMu)/theta[6]**2
        gradS = -(self.table['nashZ']*dZdS)/theta[6]**2
        gradSigma = (self.table['nashZ']**2)/theta[6]**3

        return(gradaR,gradaL,gradB,gradC,gradMu,gradS,gradSigma)

    def gradNashSecond(self,theta):

        numr = (self.table['zR']/theta[6]).apply(norm.pdf)-(self.table['zL']/theta[6]).apply(norm.pdf)
        den = (self.table['zR']/theta[6]).apply(norm.cdf)-(self.table['zL']/theta[6]).apply(norm.cdf)
        
        gradaR = -1/theta[6]*((self.table['zR']/theta[6]).apply(norm.pdf)/den)
        gradaL = 1/theta[6]*((self.table['zL']/theta[6]).apply(norm.pdf)/den)
        gradB = -1/theta[6]*(self.table['Inflation']*numr/den)
        gradC = -1/theta[6]*(self.table['OutGap']*numr/den)
        gradMu = pd.Series([0]*self.table['Date'].size)
        gradS = pd.Series([0]*self.table['Date'].size)
        gradSigma = -1/(theta[6]**2)*(self.table['zR']*(self.table['zR']/theta[6]).apply(norm.pdf)-\
                                 self.table['zL']*(self.table['zL']/theta[6]).apply(norm.pdf))/den

        return(gradaR,gradaL,gradB,gradC,gradMu,gradS,gradSigma)

    def gradNash(self,theta):
        
        self.getRhoZ(theta)
        
        gradaR = self.gradNashFirst(theta)[0].where(self.table['ChangeSign']!=0).sum()+\
                     self.gradNashSecond(theta)[0].where(self.table['ChangeSign']==0).sum()
        gradaL = self.gradNashFirst(theta)[1].where(self.table['ChangeSign']!=0).sum()+\
                    self.gradNashSecond(theta)[1].where(self.table['ChangeSign']==0).sum()
        gradB = self.gradNashFirst(theta)[2].where(self.table['ChangeSign']!=0).sum()+\
                    self.gradNashSecond(theta)[2].where(self.table['ChangeSign']==0).sum()
        gradC = self.gradNashFirst(theta)[3].where(self.table['ChangeSign']!=0).sum()+\
                    self.gradNashSecond(theta)[3].where(self.table['ChangeSign']==0).sum()
        gradMu = self.gradNashFirst(theta)[4].where(self.table['ChangeSign']!=0).sum()+\
                    self.gradNashSecond(theta)[4].where(self.table['ChangeSign']==0).sum()
        gradS = self.gradNashFirst(theta)[5].where(self.table['ChangeSign']!=0).sum()+\
                    self.gradNashSecond(theta)[5].where(self.table['ChangeSign']==0).sum()
        gradSigma = -(self.T)/theta[6]+self.gradNashFirst(theta)[6].where(self.table['ChangeSign']!=0).sum()+\
                    self.gradNashSecond(theta)[6].where(self.table['ChangeSign']==0).sum()
        
        return(np.array([gradaR,gradaL,gradB,gradC,gradMu,gradS,gradSigma]))
    
    def compareGradToLLN(self,theta,epsilon=0.00001):
        
        for i in range(0,7):
            beta = theta[:]
            beta[i] = beta[i]+epsilon
            print(self.gradNash(theta)[i])
            print(((self.lln(beta)-self.lln(theta))/epsilon))
            
    def checkGradImprovingNash(self,theta,epsilon=0.00001):
        
        good = True
        llh = self.lln(theta)
        
        for i in range(0,7):
            blarp = theta[:]
            blarp[i] = blarp[i]+epsilon*self.gradNash(theta)[i]
            if (self.lln(blarp)<llh):
                good = False
                print(i)
                
        if good:
            print("You're all good!")'''

