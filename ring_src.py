# -*- coding: utf-8 -*-
from numpy import *
import matplotlib as mpl
from matplotlib import cm,colors
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.optimize import leastsq
import os,time

# For full simulation which include dispersion, use PhaseMatchingBiphotonFWM.py
# This code is intended for design purpose (faster simulation) and assumes linear dispersion between pump, signal and idler.


# -----------------------------------------------------------------------------#    
# Plot functions
# -----------------------------------------------------------------------------#
# Lattice: bidimensional numpy array, example : lattice = ones((size, size), dtype=float )
# extent: axis extent for each axis  [begin_x,end_x,begin_y,end_y] 

def plotcolormap(lattice,extent,fname = None):
    fig = plt.figure()    
    print lattice
    print extent
    map1=colors.LinearSegmentedColormap.from_list('bla',['#000000','#FF0000','#FFFF00'])
    begin_x,end_x,begin_y,end_y = extent
    aspect = (end_x - begin_x )/(end_y - begin_y)
    plt.imshow(lattice, map1, vmin = 0,interpolation='nearest',extent=extent, aspect = 'equal')
    #plt.imshow(lattice, map1,vmin = 0, interpolation='nearest',extent=extent,aspect = aspect)
    plt.gca().xaxis.set_major_locator( MaxNLocator(nbins = 7, prune = 'lower') )
    plt.gca().yaxis.set_major_locator( MaxNLocator(nbins = 6) )
    #cbar = plt.colorbar()
    #cbar.locator = MaxNLocator( nbins = 6)
    # vmin=0,vmax = 1,
    if fname is None:
        plt.show()
    else:
        plt.savefig(fname)
    
    plt.close()

def plot(plots):
    for x,y,style in plots:
        plt.plot(x, y, style) # x, y, 'k--',
    plt.grid(True)
    plt.title('')
    plt.xlabel('')
    plt.ylabel('')    
    plt.show()    

def plotcolormapphase(lattice,extent):
    fig = plt.figure()    
    map1=colors.LinearSegmentedColormap.from_list('bla',['#0000FF','#000000','#FF0000'])
    
    plt.imshow(lattice, map1,vmin = -pi,vmax = pi, interpolation='nearest',extent=extent)
    # vmin=0,vmax = 1,
    plt.show()

# -----------------------------------------------------------------------------#    
# MISC FUNCTIONS (helpers for classes)
# -----------------------------------------------------------------------------#    
def funcpeak(lbda,lbda0):
    T = 1.*10**(-9)
    signu = 0.441/T 
    siglbda = signu/(c*10**6)*(lbda0)**2    
    return sqrt(1./(sqrt(2*pi)*siglbda) * exp(-(lbda-lbda0)**2/(2*siglbda**2))) 

"""
input state as a 2D matrix
!! the input state is not given as a density matrix
it's a pure state given in a matrix
"""
def schmidtnumber(state):
    N,M = state.shape
    ror=zeros((N,N)) # reduced density matrix
    for l in xrange(N):
        for n in xrange(N):
            for p in xrange(N):
                ror[l,n]+=state[p,l]*state[p,n]

    ror2 = dot(ror,ror)

    # compute the trace of ror2
    tmp = 0
    for k in xrange(N):
        tmp+= ror2[k,k]
    schn = 1.0/tmp
    return schn

def parse_extent(line):  
    l1 = line.split(":")[1]
    l2 = l1.split(",")[0]
    
    swlmin,swlmax = l2.split("-")
    wlmin,wlmax = float(swlmin),float(swlmax)
    return wlmin,wlmax
    
def parse_biphoton_data(line):
    l1 = line.replace("\n","")
    ls = l1.split(" ")
    res = []
    for e in ls:
        res.append(float(e))
    return array(res)
     
# -----------------------------------------------------------------------------#    
# CONSTANTS
# -----------------------------------------------------------------------------#
I = 1.0j
HPLANCK = 6.626068*10**(-34) #m2 kg / s
HBAR = HPLANCK/(2*pi)
EPSILON0 = 8.85418782*10**(-12)#m-3 kg-1 s4 A2 or C.V-1.M-1
c = 299792458.0 # CLIGHT = 299792458. # m/s
n2_Si = 6.3* 10**(-18) # m2/W (Semicond. Sci. Technol. 23 (2008) 064007 (9pp))



# -----------------------------------------------------------------------------#
# CLASS FWM_Simu 
# -----------------------------------------------------------------------------#
# This class calculates the joint spectral distribution obtained for a straight 
# waveguide with a given set of parameters
# Init (
#         * Waveguide cross section 
#         * Waveguide length (Meters) 
#         * Pump power (Watts) 
#         * Pump wavelength (um)
#         * Pulse duration (Seconds)
#         * Repetition rate (Hz)
#      ) 
#
# computeJS: Does the simulation
#    
    
class Simu(object):
    def __init__(self,
                pumpwl = 1.55,
                pulseduration=1.*10**(-12),
                 nb_points_pump = 400
                 ): 
        
        self.T = pulseduration # in seconds
        self.setPumpwl(pumpwl)
        self.pumpenvelop(pumpwl) # computes siglbda
        self.gamma = 200. # W^-1 m^-1 ;  non linear coeff IEEE JOURNAL OF SELECTED TOPICS IN QUANTUM ELECTRONICS, VOL. 16, NO. 1, JANUARY/FEBRUARY 2010
        self.nb_points_pump = nb_points_pump
        
    def setPumpwl(self,x):
        self.lbda_p = x
        
    def setPulseDuration(self,x):        
        self.T  = x
        self.pumpenvelop(self.lbda_p)        
    
    # **************
    # Pump envelopes
    # **************
    def pumpenvelop(self,lbda):        
        return self.gaussppumpenvelop(lbda) #self.gaussppumpenvelop(lbda)
        #return self.rectpumpenvelop(lbda) #self.gaussppumpenvelop(lbda)
          
    def gausspulsedpumpenvelop(self,lbda,dlbda = 0.4*10**(-4)):
        return self.gaussppumpenvelop(lbda) *(sin(2*pi*(lbda)/dlbda))**2# From laser textbook              
    
    def rectpumpenvelop(self,lbda):
        signu = 0.441/self.T # self.linewidth #0.441/sigma_t # From laser book, in Hz
        lbda0 = self.lbda_p
        siglbda = signu/(c*10**6)*(lbda0)**2
        w = sqrt(2*pi)*siglbda
        self.siglbda = siglbda
        a = 1/sqrt(w)
        lbda_min = lbda0-w/2
        lbda_max = lbda0+w/2
        step = w / 400
        self.pumprange = arange(lbda_min,lbda_max,step)

        output = (lbda>=lbda_min)*(lbda<=lbda_max)*a 
          
        return output              
    
    def gaussppumpenvelop(self,lbda):
        lbda0 = self.lbda_p
        k0,k = map(lambda x : 2*pi*self.ng/x,[lbda0,lbda])
        signu = 0.441/self.T # self.linewidth #0.441/sigma_t # From laser book, in Hz
        siglbda = signu/(c*10**6)*(lbda0)**2
        ng = self.ng
        sigk = siglbda/(lbda0)**2*2*pi*ng
        self.siglbda = siglbda
        return sqrt(1./(sqrt(2*pi)*siglbda) * exp(-(lbda-lbda0)**2/(2*siglbda**2))) # this gauss envelop 
    
    # Rectangular pulse in the temporal domain
    # lbda in um
    # T : pulse length [S]
    def sincpumpenvelop(self,lbda):
        T = self.T
        om = 2*pi*c/(lbda*10**-6)
        om0 = 2*pi*c/(self.lbda_p*10**(-6))
        dom = om - om0
        #return sinc(dom*T/2) * sqrt(T/(2*pi)) # this normalization works when integrating over omega
        # *** WARNING, in python, sinc(x) = sin(pi*x)/(pi*x) which is already normalized to one ! ***
        return sinc(dom*T/2) * sqrt(T*pi*c*10**6/(lbda**2))  # c in um/s, lbda in um, T in s;  this normalization is for lambda
    
    # **************
    #
    # **************
    
    # This provides the range of lbdas which should be used to accurately span the pump
    def updatepumprange(self):
        print "Get pump range ..."
        lbda_p = self.lbda_p
        lbda_step= 0.00000001 # step for finding the pump range
        P = 0.
        targetfraction = 0.95
        
        deltalbda = 0.5*10**(-6) # initialize deltalbda at 1pm
        
        while (P<targetfraction):
        
            deltalbda = 2*deltalbda
            lbdas = arange(lbda_p-deltalbda,lbda_p+deltalbda,lbda_step)
            #print P
            
            P=(self.pumpenvelop(lbdas)*self.pumpenvelop(lbdas).conjugate()).sum()*lbda_step
            print P
        print P
        N = self.nb_points_pump
        step = (lbdas[-1]-lbdas[0])/N # Step for the returned pump range
        res = arange(lbdas[0],lbdas[-1],step)
        #print "Size of pump lbdas" ,lbdas.size
        #print self.pumpenvelop(lbda_p)
        print "Pump range : (um)",lbdas[0],lbdas[-1]
        self.pumprange = res
        return res
    
    def setRangeJS(self,lbda_s_min,lbda_s_max,lbda_i_min,lbda_i_max):
        self.lbda_s_min = lbda_s_min
        self.lbda_s_max = lbda_s_max
        self.lbda_i_min = lbda_i_min
        self.lbda_i_max = lbda_i_max
        self.extent = [x*1000 for x in [self.lbda_i_min,self.lbda_i_max,self.lbda_s_min,self.lbda_s_max]] # um to nm 
        
        print self.extent
     
    def setRangeScanResonance(self,lbda_s_min,lbda_s_max):
        # Get the range for signal centered on the resonance
        lsm,lsM = lbda_s_min,lbda_s_max        
        # Get the range for idler using rough energy conservation
        lp = self.lbda_p
        lp_min = min(self.pumprange)
        lp_max = max(self.pumprange)        
        lim = 1./(2./lp_min - 1./lsM)
        liM = 1./(2./lp_max - 1./lsm)
        
        print "avg_pumps", (lim+lsm)/2,(liM+lsM)/2
        
        #print "%.2f %.2f ; %.2f %.2f (pm)" % (lsm*10**6,lsM*10**6,lim*10**6,liM*10**6)
        print lsm,lsM,lim,liM
        self.setRangeJS(lsm,lsM,lim,liM)
    
        
    # Override these methods to add custom filters on signal and idler arm
    def filter_idler(self,lbda):
        return ones(lbda.size)
    
    def filter_signal(self,lbda):
        return ones(lbda.size)
    
    def getPurityAfterFilter(self):
        Ni = self.Ni
        Ns = self.Ns
        # Apply custom filters:
        m_filter_signal =zeros((Ni,Ns))
        m_filter_idler =zeros((Ni,Ns))
        
        for i in arange(Ni):
            m_filter_signal[i,:] = self.filter_signal(self.a_lbda_s)
        for j in arange(Ns):
            m_filter_idler[:,j] = self.filter_idler(self.a_lbda_i)
        lattice = self.normlattice*m_filter_signal*m_filter_idler
        
        # Multiply by the appropriate missing constants
        
        Z =  lattice.sum()# sqrt(abs(lattice*conjugate(lattice)).sum())                  
        normlattice = sqrt(abs(lattice/Z))         
        self.normlattice_unfiltered  = self.normlattice[:,:] # Save the previous matrix
        self.normlattice = normlattice # assign the new filtered matrix
        purity = self.computeHeraldedPhotonPurity() # computes the purity after filtering
        self.normlattice = self.normlattice_unfiltered # restore the previous matrix
        return purity
    
    def computeJS(self):
        pass
    
    def plotBiphoton(self,fname = None):
        plotcolormap(self.normlattice,self.extent,fname)  
        
    def __g__(self,i,j):
        #return (self.normlattice[i,:]*conjugate(self.normlattice[j,:])).sum()
        return (self.normlattice[i,:]*exp(I*self.phases[i,:])*conjugate(self.normlattice[j,:]*exp(I*self.phases[j,:]))).sum()
    
    def __g_nophase__(self,i,j):
        return (self.normlattice[i,:]*conjugate(self.normlattice[j,:])).sum()
    
    def __G_nophase__(self,i,j,k,l):
        return self.__g_nophase__(i,j)*self.__g_nophase__(k,l)
    
    vectg = vectorize(__g__)
    
    def __G__(self,i,j,k,l):
        return self.__g__(i,j)*self.__g__(k,l)
    
    vectG = vectorize(__G__)
    vectG_nophase = vectorize(__G_nophase__)
    # Purity = Tr(ro**2)
    def computenaivepurity(self):
        lattice = sqrt(self.normlattice)
        N = self.N
        P = 0
        for n in xrange(self.N):
            for m in xrange(self.N):
                P+= (lattice[:,n]*conjugate(lattice[:,m])).sum()*(lattice[:,m]*conjugate(lattice[:,n])).sum()                
        self.purity = abs(P)
        self.schn = 1./P
        return P
    # Computes the probability of getting coincidences between two heralded photons from different sources
    
    def computePcoincfrom2photons(self):
        lattice = sqrt(self.normlattice)
        #print "State Norm:", abs(lattice*conjugate(lattice)).sum() # equivalent to the trace
        print "Computing proba coincidence"
        N = self.N
        omega1 = zeros((N,N),int)
        omega2 = zeros((N,N),int)
        for i in range(N):
            omega1[:,i]= arange(N)
            omega2[i,:]= arange(N)
        
        Gnnmm = self.vectG(self,omega1,omega1,omega2,omega2)
        Gnmmn = self.vectG(self,omega1,omega2,omega2,omega1)
        print "Gnnmm: ",Gnnmm.sum()
        print "Gnmmn: ",Gnmmn.sum()
        Pcoinc = 0.5*(Gnnmm.sum()-Gnmmn.sum()) # See proof in my labbook from 2012 (27/01/2012)
        print "Pcoinc: ",Pcoinc
        print "Visibility: ", 1.-Pcoinc/0.5
        self.visibility= 1.-Pcoinc/0.5
        return 1.-Pcoinc/0.5
        
    def computeHeraldedPhotonPurity(self):
        lattice = self.normlattice
        N = self.N
        omega1 = zeros((N,N),int)
        omega2 = zeros((N,N),int)
        for i in range(N):
            omega1[:,i]= arange(N)
            omega2[i,:]= arange(N)

        purity = self.vectG(self,omega1,omega2,omega2,omega1).sum()
        self.purity = abs(purity)
        self.schn = 1/purity
        return abs(purity)
              
###
# -----------------------------------------------------------------------------#
# CLASS RingSimu 
# -----------------------------------------------------------------------------#
# This class calculates the joint spectral distribution obtained in a ring  
# resonator for a given set of parameters
# Init (
#         * Cavirt length (Meters) 
#         * Pump wavelength (um)
#         * Pulse duration (Seconds)
#         * Repetition rate (Hz)
#         * N: grid sampling (the JSA is stored in a NxN grid)
#         * r: ring coupling (r = 1 means no coupling, while r = 0 means full coupling)
#         * tau: round trip transmission which accounts for the loss in the ring resonator
#         * ng: group index 
#      ) 
#
# setPumpToClosestRes(lambda) : Sets the pump to the closest resonance to the given wavelength
# setRangeScanResonance(p) : Sets the resonance to be used for collecting the idler photon. p is the resonance number. 
#                            p = 0 is the same as the pump resonance
#                            p = +1 or -1 are the next nearest resonance to the pump
#                            p = +M or -M ....
# 
# plotcavityresponse() : Shows the transmission spectrum of the cavity
# computeJS() : Does the simulation
#
# __applycavity__(lambda) : This is the function which applies the cavity. By default, this function applies a ring resonator. 
#                       Different cavities can however be used.

# save(filename) : Saves the result of the simulation including all the parameters, the full state, and the derived parameters such as the Schmidt number                                      
#            
class RingSimu(Simu):
    def __init__(self,
                    length = 80., # um 
                    pumpwl = 1.55,
                    pulseduration=1.*10**(-12),
                    N = 200,
                    r = 0.98,
                    tau = 1.0,
                    ng = 4.2): # 300*10**3 -> 300 kHz linewidth
        self.ng = ng
        Simu.__init__(self,
                      pumpwl = pumpwl,pulseduration=pulseduration)
        
        
        self.lbda_p = pumpwl # in um # We take the cavity resonance wavelength equal to the pump central wavelength
        self.mpump = -1 # resonance number closest to the pump
        # Ring parameters
        self.L = length # Length of the ring in um
        self.r = r
        self.tau = tau #  tau = 1. -> No loss 
        #self.tau = self.r # critical coupling
        self.N = N
        self.lattice = zeros((N,N))
        # For loading purpose : Params
        self.purity = -1
        self.schn = -1
        self.geneeff = -1
        self.setters = {"Purity" : self.__setPurity__,
                 "Schmidt number" : self.__setSchn__,
                 "r" : self.__setr__,
                 "Nb pairs per pulse" : self.__setgeneeff__,
                 "Pulse duration (ps)" : self.__setT__ ,
                 "N" : self.__setN__,
                 }
       
        self.resonancenumber = 0 # Resonance scanned for signal
        
    # Setters when loading
    def __setPurity__(self,x):
        self.purity = x
        
    def __setSchn__(self,x):
        self.schn = x
        
    def __setr__(self,x):
        self.r = x
        
    def __setgeneeff__(self,x):
        self.geneeff = x
        
    def __setT__(self,x):
        self.T = x
        
    def __setN__(self,x):
        self.N = x
        self.lattice = zeros((x,x))        
        self.phases = zeros((x,x))        
    
    ###
    def setTau(self,x):
        self.tau = x
    
    def setr(self,x):
        self.r = x
    
    def setL(self,L):
        self.L = L
    
    ###
    def ring(self,lbda):                
        k = 2*pi*self.ng/(lbda)
        t = sqrt(1-self.r**2)
        tau = self.tau
        r = self.r
        return I*t/(1-tau*r*exp(I*k*self.L))
    
    def cavityTransmission(self,lbda):
        t = sqrt(1-self.r**2)
        return self.r+I*t*self.ring(lbda)
    
    # Override these methods to add custom filters on signal and idler arm
    def filter_idler(self,lbda):
        return ones(lbda.size)
    
    def filter_signal(self,lbda):
        return ones(lbda.size) 
    
    def __applycavity__(self,lbda):                
        return self.ring(lbda)
    
    # Returns the closest cavity resonance for a given lambda and the resonance number
    def getClosestCavityRes(self,lbda):
        m = round(self.L*self.ng/lbda)
        kp0 = m*2*pi/self.L # target pump propagation constant
        lbda0 = 2*pi*self.ng/kp0 
        return lbda0,m
        
    # Centers the pump on the closest cavity resonance
    def setPumpToClosestRes(self,lbda):
        self.lbda_p,self.mpump = self.getClosestCavityRes(lbda)        
        print "Pump is set at %.7f um" % self.lbda_p
    
    # Get the range to scan for signal for the nth resonance with respect to the pump
    # Rq : The pump should have been set such that mpump has a meaningful value
    def getSignalRange(self,n):
        FWHM = (1-self.r*self.tau)*self.lbda_p**2/(self.ng*sqrt(2)*pi*self.L)
        print "FWHM (um) : ",FWHM
        fullrange = 5*FWHM # 
        wlFSR = self.lbda_p**2/(self.L*self.ng) # FSR in lambda
        print "FSR (um) : ",wlFSR
        lbda_s,m = self.getClosestCavityRes(self.lbda_p+n*wlFSR)
        print "Resonance (um) : ",lbda_s
        return lbda_s-fullrange,lbda_s+fullrange
    
    def plotCavityResponse(self,albda = None):
        lbda_i, mi = self.getClosestCavityRes(0.5 * (self.lbda_i_min+self.lbda_i_max))
        lbda_s = 1./(2./self.lbda_p-1./lbda_i)
        if albda == None:
            albda = arange(min(self.lbda_s_min,self.lbda_i_min),
                           max(self.lbda_s_max,self.lbda_i_max),0.0000001)
        cavity = self.__applycavity__(albda)*self.__applycavity__(albda).conjugate()
        pump = self.pumpenvelop(albda)**2
        
        
        signal_wl = funcpeak(albda,lbda_s)
        idler_wl = funcpeak(albda,lbda_i)
        plot([(albda,cavity,"-"),
              (albda,pump/pump.max()*cavity.max(),"-"),
              (albda,signal_wl/signal_wl.max()*cavity.max(),"r-"),
              (albda,idler_wl/idler_wl.max()*cavity.max(),"r-")
              ]) # Plot the pump normalised wrt the biggest field enhancement
    
    def setRangeJS(self,lbda_s_min,lbda_s_max,lbda_i_min,lbda_i_max):
        self.lbda_s_min = lbda_s_min
        self.lbda_s_max = lbda_s_max
        self.lbda_i_min = lbda_i_min
        self.lbda_i_max = lbda_i_max
     
    def setRangeScanResonance(self,m):
        # Get the range for signal centered on the resonance
        lsm,lsM = self.getSignalRange(m)
        self.resonancenumber = m
        # Get the range for idler using rough energy conservation
        lp = self.lbda_p
        lim = 1./(2./lp - 1./lsM)
        liM = 1./(2./lp - 1./lsm)
        #print "%.2f %.2f ; %.2f %.2f (pm)" % (lsm*10**6,lsM*10**6,lim*10**6,liM*10**6)
        print lsm,lsM,lim,liM
        self.setRangeJS(lsm,lsM,lim,liM)
        
    def updatePumpRange(self):
        print "Get pump range ..."
        lbda_p = self.lbda_p
        print lbda_p
        lbda_step= 0.00000001 # step for finding the pump range
        P = 0.
        targetfraction = 0.95
        
        deltalbda = 0.5*10**(-6) # initialize deltalbda at 1pm
        
        while (P<targetfraction):
        
            deltalbda = 2*deltalbda
            lbdas = arange(lbda_p-deltalbda,lbda_p+deltalbda,lbda_step)
            #print P
            
            P=(self.pumpenvelop(lbdas)*self.pumpenvelop(lbdas).conjugate()).sum()*lbda_step
            print P
        print P
        N = self.nb_points_pump
        # get cavity range
        # If the pump is broader than the cavity, then we should chop the pump to the cavity region such that the grid is fine enough in the cavity
        # If the pump is narrower than the cavity, then keep pump range
        lsm,lsM = self.getSignalRange(0)
        rl = lsM-lsm
        lsm = lsm-rl/2
        lsM = lsM+rl/2
        lbdamax = min(lbdas[-1],lsM)
        lbdamin = max(lbdas[0],lsm)
        
        step = (lbdamax-lbdamin)/N # Step for the returned pump range
        
        res = arange(lbdamin,lbdamax,step)
        #print "Size of pump lbdas" ,lbdas.size
        #print self.pumpenvelop(lbda_p)
        self.pumprange = res
        print "Pump range : (um)",lbdas[0],lbdas[-1]
        return res    
        
    def getjointproba(self):
        return self.normlattice

    def getjointprobascaled(self):
        return self.normlattice/self.normlattice.max()
    
    def computeJS(self, target_proba = 0.1): # begin=1.545,end=1.555,step=0.0001
        xi = (1. - sqrt(1-4*target_proba)) / 2 
        self.target_proba = target_proba
        L = self.L # Cavity length 
        N = self.N   
        lbda_s_min = self.lbda_s_min 
        lbda_s_max = self.lbda_s_max
        lbda_i_min = self.lbda_i_min 
        lbda_i_max = self.lbda_i_max         
        step_i = (lbda_i_max-lbda_i_min)/N
        step_s = (lbda_s_max-lbda_s_min)/N
        
        a_lbda_i = arange(lbda_i_min,lbda_i_max,step_i)[0:N]
        a_lbda_s = arange(lbda_s_min,lbda_s_max,step_s)[0:N]
        Ni = a_lbda_i.size
        Ns = a_lbda_s.size
        print Ni, Ns
        Ni = N
        Ns = N
        self.step_i = step_i
        self.step_s = step_s
        
        rangepump = self.pumprange
        M = rangepump.size
        dlbda_pump = (rangepump.max()-rangepump.min())/M
        lattice = zeros((Ni,Ns))
        a_lbda_p1 = rangepump
        cav_resp_p1 = self.__applycavity__(a_lbda_p1)
        a_p1 = self.pumpenvelop(a_lbda_p1) # pump amplitude 1
        ng = self.ng
        print "Steps" ,step_i,step_s
        #dbgpm = 0.
        pumpmax = self.pumpenvelop(self.lbda_p)
        phases = zeros((Ni,Ns))
        for j in xrange(Ns):
            print j
            lbda_s = a_lbda_s[j] # lbda_s_min+j*step_s
            cav_resp_s = self.__applycavity__(lbda_s)
            for i in xrange(Ni):
                lbda_i = a_lbda_i[i] # lbda_i_min+i*step_i
                a_lbda_p2 = 1./(1./lbda_s+1./lbda_i-1./a_lbda_p1)                                
                a_p2 = self.pumpenvelop(a_lbda_p2) # pump amplitude 2
                a_res = a_p1*a_p2*cav_resp_p1*self.__applycavity__(a_lbda_p2)* self.__applycavity__(lbda_i)*cav_resp_s # 
                a_res = a_res * a_lbda_p2/a_lbda_p1
                
                # Multiply by the dlambda; 
                # The pump function is i um^(-1/2), dlbda_pump is in um 
                a_res = a_res*dlbda_pump
                res = a_res.sum() # unitless
                # Multiply by the dlambda
                # Since the formula was derived for domega, we have to remember that domega = -2*pi*c/lbda**2 * dlbda
                lattice[i,Ns-1-j]= abs(res.real**2+res.imag**2)* (step_i/(lbda_i**2)) * (step_s/(lbda_s**2)) 
                phases[i,Ns-1-j] = angle(res)
                
        
        # Multiply by the appropriate missing constants
        lattice = lattice*(c*self.gamma*(self.L))**2/(2*pi**2)
        Z =  lattice.sum()# sqrt(abs(lattice*conjugate(lattice)).sum())                  
        self.normlattice = sqrt(abs(lattice/Z))
        
        self.lattice = lattice 
        self.phases = phases 
        beta =  2*lattice.sum() 
        
        # Compute the energy required per pulse
        Epulse = arctanh(sqrt(xi))/sqrt(beta) # Approximation valid in the case of two-mode squeezer
        self.Epulse = Epulse
        # Theory calculation for CW regime for comparison
        
        vg = c/self.ng
        print "Epulse (nJ) ", self.Epulse*10**9
        print "gamma W-1,m-1", self.gamma
        print "L (um)", L
        print "T (ps)", self.T*10**12
        print "vg %e" % vg
        print "r : %.4f" % self.r
        print "tau : %.4f" % self.tau
        print "Siglbda : %.5f" % (self.siglbda)
        print "Nb pairs per pulse:",target_proba        

        lbda_i0 = (lbda_i_max+lbda_i_min)/2
        lbda_s0 = (lbda_s_max+lbda_s_min)/2
        self.extent = list(array([lbda_i_min-lbda_i0,lbda_i_max-lbda_i0,lbda_s_min-lbda_s0,lbda_s_max-lbda_s0])*1000) # Check where should go i and s

    
    def getPhases(self):
        return self.phases
    
    def getAverageSpectra(self):
        return self.normlattice.sum(axis = 0),self.normlattice.sum(axis = 1)
    
    def save(self,directory="resonances_toshiba"):
        timestamp = time.strftime("%m%d_%H%M",time.localtime(time.time()))
        
        # Create repository if it does not exist
        if not os.path.exists("data\\%s" % directory):
            os.makedirs("data\\%s" % directory)

        fname = "data\\%s\\simu_%s_r=%.3f_tau=%.3f_%.2fps_res=%d.txt" % (directory,timestamp,self.r,self.tau,self.T * 10**12,self.resonancenumber)
        # Header
        fw = open(fname,"w")
        fw.write("#Laser parameters\n")
        fw.write("%s : %.3f\n" % ("Pulse duration (ps)",self.T*10**12))
        fw.write("%s : %.18e\n" % ("Energy per pulse (uJ)",self.Epulse*1000000))
        fw.write("%s : %.6f\n" % ("Pump wavelength (um)",self.lbda_p))
        fw.write("\n#Waveguide parameters\n")
        fw.write("%s : %.3f\n" % ("gamma (W-1 m-1)",self.gamma))
        fw.write("\n#Ring parameters\n")
        fw.write("%s : %.3f\n" % ("Cavity length (um)",self.L))
        fw.write("%s : %.5f\n" % ("Tau",self.tau))
        fw.write("%s : %.5f\n" % ("r",self.r))
        fw.write("\n#BiPhoton state properties\n")
        fw.write("%s : %.5f\n" % ("Nb pairs per pulse",self.target_proba))
        self.computeHeraldedPhotonPurity()
        #self.computePcoincfrom2photons()
        #fw.write("%s : %.6f\n" % ("Visibility from two heralded sources",self.visibility))
        fw.write("%s : %.6f\n" % ("Schmidt number",abs(self.schn)))
        fw.write("%s : %.6f\n" % ("Purity",abs(1/self.schn)))
        
        # Theory calculation for CW regime for comparison
        vg = c/self.ng
        beta2 = self.gamma**2*(self.Epulse/self.T)**2*(self.L*10**(-6))/8 * vg*self.T*(1-self.r**2)**4/(1-self.r)**7
        fw.write("%s : %.5f\n" % ("Nb pairs(analytical CW)",beta2))
        fw.write("\n")
        fw.write("N=%d\n" % self.N)
        fw.write("Resonance number : %d\n" % self.resonancenumber)
        fw.write("\n#Scan range\n")
        fw.write("%s : %.6e - %.6e, %.6e\n" % ("idl min, idl max, step (um)",self.lbda_i_min,self.lbda_i_max,self.step_i))
        fw.write("%s : %.6e - %.6e, %.6e\n" % ("sig min, sig max, step (um)",self.lbda_s_min,self.lbda_s_max,self.step_s))
        fw.write("\n#Raw data Biphoton distribution\n")
        # Saves the joint spectrum
        for j in xrange(self.N):
            line = " ".join(("%.18e" % x) for x in self.lattice[:,self.N-1-j])
            fw.write(line+"\n")       
        fw.write("\n#Raw data Biphoton phase distribution\n")
        # Saves the joint spectrum
        for j in xrange(self.N):
            line = " ".join(("%.18e" % x) for x in self.phases[:,self.N-1-j])
            fw.write(line+"\n")       
        fw.close()
        return fname
    
    def load(self,fname):
        print "Loading %s ..." % fname
        f = open(fname,"r")
        line = f.readline()
        
        while (len(line)>0):
            
            if line.startswith("#Scan range"):
                # Load the extent of the wavelength for signal and idler
                line = f.readline() # Readline for the idler
                self.lbda_i_min,self.lbda_i_max = parse_extent(line)
                line = f.readline() # Readline for the signal
                self.lbda_s_min,self.lbda_s_max = parse_extent(line)
                self.extent = [self.lbda_i_min,self.lbda_i_max,self.lbda_s_min,self.lbda_s_max] # Check where should go i and s
            if line.startswith("#Raw data Biphoton distribution"):
                # Load the biphoton distribution
                for j in xrange(self.N):
                    line = f.readline() 
                    self.lattice[:,self.N-1-j] = parse_biphoton_data(line)
            
            if line.startswith("#Raw data Biphoton phase distribution"):
                # Load the biphoton phase distribution
                for j in xrange(self.N):
                    line = f.readline() 
                    self.phases[:,self.N-1-j] = parse_biphoton_data(line)
                
            if line.find("#")>=0:
                l1 = line.split("#")[0]
            if line.find(":")>=0:
                line = line.replace("\n","")
                name,value = line.split(" : ")
                if name in self.setters.keys():
                    self.setters[name](float(value))
            elif line.startswith("N="):
                name,value = line.split("=")
                self.setters[name](int(value))
            line = f.readline()  
        Z =  self.lattice.sum()# sqrt(abs(lattice*conjugate(lattice)).sum())                  
        self.normlattice = sqrt(abs(self.lattice/Z))         
        f.close()
 
def main():
    
    T = 5.
    N = 100 
    r = 0.93
    tau = 1.-0.0198
    radius = 10.
    coupling_length = 5.
    lbda0= 1.55
    res_number = 1 # resonance number (pump resonance is 0).
    for res_number in [1]: 
        for r in [0.9]: # [0.95,0.96,0.97,0.98,0.99]: # 0.85,0.86,0.87,0.88,0.89,0.90,0.91,0.92,0.93,0.94,0.95,0.96
            for tau in [0.997]: # 0.76,0.96,0.98
                #for r2 in [0.9998,0.9997,0.9996,0.9995,0.9994]: #[1.0,0.9999,0.999,0.99]:
                mySim =RingSimu(length = 2*(radius*pi+coupling_length),
                                pulseduration = T*10**(-12),
                                N = N,
                                r = r,
                                tau = tau, 
                                pumpwl = lbda0) # 500 

                mySim.setPumpToClosestRes(lbda0)                    
                mySim.setRangeScanResonance(+res_number)                    
                #mySim.plotCavityResponse()                    
                mySim.updatepumprange()
                mySim.computeJS()                
                fname = mySim.save("Ring_pumpscan")
                mySim.plotBiphoton()
                #mySim.plotBiphoton(fname[:-3]+"png")

    
if __name__ == "__main__": 
    main()

