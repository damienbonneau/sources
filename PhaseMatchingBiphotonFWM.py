# -*- coding: utf-8 -*-
from numpy import *
import matplotlib as mpl
from matplotlib import cm,colors
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.optimize import leastsq
import os,time



# -----------------------------------------------------------------------------#    
# Plot functions
# -----------------------------------------------------------------------------#
# Lattice: bidimensional numpy array, example : lattice = ones((size, size), dtype=float )
# extent: axis extent for each axis  [begin_x,end_x,begin_y,end_y] 

def plotcolormap(lattice,extent,fname = None):
    fig = plt.figure()    
    
    map1=colors.LinearSegmentedColormap.from_list('bla',['#000000','#FF0000','#FFFF00'])
    begin_x,end_x,begin_y,end_y = extent
    aspect = (end_x - begin_x )/(end_y - begin_y)
    plt.imshow(lattice, map1,vmin = 0, interpolation='nearest',extent=extent,aspect = aspect)
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
# CLASS Waveguide 
# -----------------------------------------------------------------------------#
# Init (width, height):
#    * Take the width and height of the waveguide cross section as parameters
#    * Loads a file containing lbda vs neff
#    * fits a dispersion curve to the data loaded
# This class has methods to obtain the effective index, the group index, and wave number when given a wavelength
#
class Waveguide(object):
    def __init__(self,width,height):
        self.rootname = "waveguide_data_noslab"
        self.width = width
        self.height = height
        s = "%dx%d" % (width,height)
        files = os.listdir(self.rootname)
        for fname in files:
            if fname.find(s) >=0:
                self.__load__(fname)
                self.__fit__()                                     
        
        # We fix the FWM effective area that we calculate using the overlap between the four fields
        self.Aeff = 0.03 # um^2
        
                
    def __load__(self,fname):
        path = self.rootname+"\\"+fname
        f = open(path)
        line = f.readline()
        lbdas = []
        neffs = []
        while(len(line))>0:
            splitted = line.split("\t")                  
            lbda,neff = splitted[0:2] 
            
            line = f.readline()
            if lbda>0:
                lbdas.append(float(lbda))
                neffs.append(float(neff))
        
        self.lbdas = array(lbdas)
        self.neffs = array(neffs)
        return
        
    def __fit__(self):
        p0 = [1,0,0,0]        
        plsqwl2n = leastsq(self.__residuals__, p0, args=(self.neffs, self.lbdas)) 
        self.pwl2n = plsqwl2n[0] # wavelength to neff
        #print self.p
        
    def __func__(self,p,x):
        d,c,b,a = p 
        return a*x**3+b*x**2+c*x+d
        
    def __residuals__(self,p,y, x):
        err = y-self.__func__(p,x)
        return err                
    
    def getneff(self,lbda):    
        return self.__func__(self.pwl2n,lbda)
    
    # lbda in um
    def wl2kv(self,a_lbda):
        return 2*pi*self.getneff(a_lbda)/(a_lbda) # the kvector z component is returned in um-1
    
    def kv2wl(self,a_kv):
        pass # not as easy ... 
        
    def plotneff(self):
        x = arange(min(self.lbdas),max(self.lbdas),0.1)
        plots = [(self.lbdas,self.neffs,"-"),(x,self.getneff(x),"-")]
        plot(plots)    
    
    def getng(self,lbda):
        lbda_step = 0.00001
        lbda1 = lbda - lbda_step
        lbda2 = lbda + lbda_step
        neff1 = self.getneff(lbda1)
        neff2 = self.getneff(lbda2)
        neff = self.getneff(lbda)
        
        ng = neff -lbda*(neff2-neff1)/(2*lbda_step)
        return ng

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
    
class FWM_Simu(object):
    def __init__(self,wg = Waveguide(550,220),
                      length = 0.03, # 0.03 ->3cm
                      pumppower = 0.1*10**-3,pumpwl = 1.55,pulseduration=1.*10**(-12),reprate = 40*10**6, N= 200
                 ): 
        
        self.T = pulseduration # in seconds
        self.wg = wg # waveguide crosssection (Waveguide object)
        self.length = length # Propagation length in the waveguide
        self.L = length
        self.pumppower = pumppower  # in W
        #self.gamma = 3*10**2 # W^-1 m^-1 ;  non linear coeff IEEE JOURNAL OF SELECTED TOPICS IN QUANTUM ELECTRONICS, VOL. 16, NO. 1, JANUARY/FEBRUARY 2010
        self.lbda_p = pumpwl
        #self.pumpenvelop(self.lbda_p)
        self.pumpenvelop(pumpwl) # computes siglbda
        self.gamma = 460. # 2*pi/(self.lbda_p*10**(-6))*n2_Si/(self.wg.Aeff*10**(-12)) #W-1 M-1
        #print "Gamma", self.gamma
        self.reprate = reprate # Hz
        self.Epulse = self.pumppower/self.reprate #Energy per pulse in J
        self.N = N
    def setPumpwl(self,x):
        self.lbda_p = x
        
        
    def setPulseDuration(self,x):        
        self.T  = x
        self.pumpenvelop(self.lbda_p)        
    # Define phase matching function

    def getdk(self,p1,p2,lbda_p1,lbda_p2,lbda_s,lbda_i):
        kp1,kp2,ki,ks = map(self.wg.wl2kv,[lbda_p1,lbda_p2,lbda_i,lbda_s])
        ga = self.gamma*10**(-6) # to put gamma in um
        dk = kp1+kp2-ks-ki-ga*(p1+p2) # When putting gamma, the phase matching bandwidth changes dramatically
        return dk
    
    # **************
    # Pump envelopes
    # **************
    def pumpenvelop(self,lbda):        
        return self.gaussppumpenvelop(lbda) #self.gaussppumpenvelop(lbda)
        #return self.rectpumpenvelop(lbda) #self.gaussppumpenvelop(lbda)
    
    def toplotCWGain(self,lbda_s = arange(1.5,1.6,0.0001)):
        lbda_i = 1./(2/self.lbda_p-1/lbda_s)
        a_dk = self.getdk(0,0,self.lbda_p,self.lbda_p,lbda_s,lbda_i) # um-1
        a_phasematching = sinc(self.length*10**6/2*a_dk) 
        return a_phasematching**2
                
    def gausspulsedpumpenvelop(self,lbda,dlbda = 0.4*10**(-4)):
        return self.gaussppumpenvelop(lbda) *(sin(2*pi*(lbda)/dlbda))**2# From laser textbook              
    
    def rectpumpenvelop(self,lbda):
        signu = 0.441/self.T # self.linewidth #0.441/sigma_t # From laser book, in Hz
        sigomega = 2*pi*signu
        lbda0 = self.lbda_p
        siglbda = signu/(c*10**6)*(lbda0)**2
        w = sqrt(2*pi)*siglbda
        self.siglbda = siglbda
        a = 1/sqrt(w)
        lbda_min = lbda0-w/2
        lbda_max = lbda0+w/2
        #print "lbdas", lbda_min,lbda_max
        step = w / 400
        self.pumprange = arange(lbda_min,lbda_max,step)
        #print "min ", lbda_min,lbda[0]
        #print "max ", lbda_max,lbda[-1]
        output = (lbda>=lbda_min)*(lbda<=lbda_max)*a 
        #if type(lbda) == type(zeros(5)):
        #    print min(lbda),lbda_min,lbda_max,max(lbda)," ---> ", output.sum()                
            
        return output              
    
    def gaussppumpenvelop(self,lbda):
        lbda0 = self.lbda_p
        k0,k = map(self.wg.wl2kv,[lbda0,lbda])
        signu = 0.441/self.T # self.linewidth #0.441/sigma_t # From laser book, in Hz
        sigomega = 2*pi*signu
        siglbda = signu/(c*10**6)*(lbda0)**2
        ng = self.wg.getng(lbda0)
        sigk = siglbda/(lbda0)**2*2*pi*ng
        self.siglbda = siglbda
        omega0 = 2*pi*c/lbda0
        omega = 2*pi*c/lbda 

        #return exp(-2*log(2)*((lbda0-lbda)*10**-6)**2/(siglbda**2)) # From laser textbook
        return sqrt(1./(sqrt(2*pi)*siglbda) * exp(-(lbda-lbda0)**2/(2*siglbda**2))) # this gauss envelop is on lambda which is probably not very physical ...
        #return sqrt(1./(sqrt(2*pi)*sigomega) * exp(-(omega-omega0)**2/(2*sigomega**2)))*sqrt(2*pi*c)/lbda
    
    
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
        N = 400
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
    
    
    def computeJS_old(self,begin=1.545,end=1.555): # begin=1.545,end=1.555,step=0.0001
        #size = int((end-begin)/step)
        size = self.N
        step = (end-begin) / self.N
        P = self.pumppower
        L = self.length
        lattice = ones((size, size), dtype=float )        
        phases = ones((size, size), dtype=float )        
        for i in xrange(size):
            print i
            lbda_i = i*step+begin
            for j in xrange(size):
                lbda_s = j*step+begin
                a_lbda_p1 = self.pumprange
                a_lbda_p2 = 1./(1/lbda_s+1/lbda_i-1/a_lbda_p1)
              
                a_p1 = P*self.pumpenvelop(a_lbda_p1) # pump amplitude 1
                a_p2 = P*self.pumpenvelop(a_lbda_p2) # pump amplitude 2
                a_dk = self.getdk(a_p1,a_p2,a_lbda_p1,a_lbda_p2,lbda_s,lbda_i)
                a_phasematching = 1
                a_expi = 1
                #a_phasematching = sinc(L/2*a_dk)                
                a_expi = exp(I*L/2*a_dk)
                a_res = a_phasematching*a_expi*a_p1*a_p2
                res = a_res.sum()*a_res.size*step
 
                lattice[i,size-1-j]= sqrt(abs(res.real**2+res.imag**2)) #res res #
                phases[i,size-1-j] = angle(res)
        #N = sqrt((lattice*conjugate(lattice)).max())    
        #lattice = lattice/N
        self.lattice = lattice
        self.phases = phases
        self.extent = [begin*1000,end*1000,begin*1000,end*1000]   
        Z =  lattice.sum()# sqrt(abs(lattice*conjugate(lattice)).sum())                  
        self.normlattice = sqrt(abs(lattice/Z))     
        
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
        P = self.pumppower
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
        self.a_lbda_i = a_lbda_i
        self.a_lbda_s = a_lbda_s
        Ni = a_lbda_i.size
        Ns = a_lbda_s.size
        print Ni, Ns
        self.Ni = Ni
        self.Ns = Ns
        self.step_i = step_i
        self.step_s = step_s
        
        rangepump = self.pumprange
        M = rangepump.size
        dlbda_pump = (rangepump.max()-rangepump.min())/M
        lattice = zeros((Ni,Ns))
        a_lbda_p1 = rangepump        
        a_p1 = self.pumpenvelop(a_lbda_p1) # pump amplitude 1        
        ng = self.wg.getng(self.lbda_p)
        print "Steps" ,step_i,step_s
        #dbgpm = 0.
        pumpmax = self.pumpenvelop(self.lbda_p)
        phases = zeros((Ni,Ns))
        print max(a_p1)
        for j in xrange(Ns):
            #rint j
            lbda_s = a_lbda_s[j] # lbda_s_min+j*step_s            
            for i in xrange(Ni):
                lbda_i = a_lbda_i[i] # lbda_i_min+i*step_i
                a_lbda_p2 = 1./(1./lbda_s+1./lbda_i-1./a_lbda_p1)                                                 
                
                a_p2 = self.pumpenvelop(a_lbda_p2) # pump amplitude 2
                #print a_lbda_p2[0],a_lbda_p2[-1]," ---> ", a_p2.sum()                
               
                #print max(a_p2)
                # In order to save computation time we can take a_pm = 1. for small cavities
                a_dk = 1.
                a_pm = 1.
                #a_dk = self.getdk(P*a_p1*conjugate(a_p1),P*a_p2*conjugate(a_p2),a_lbda_p1,a_lbda_p2,lbda_s,lbda_i)
                #a_pm =  sinc(L/2*a_dk/pi)  # the L will be added later in the global constant               
                
                
                a_res = a_p1*a_p2*a_pm
                a_res = a_res * a_lbda_p2/a_lbda_p1
                # Multiply by the dlambda; 
                # The pump function is i um^(-1/2), dlbda_pump is in um 
                a_res = a_res*dlbda_pump
                res = a_res.sum() # unitless
                
                #res = res 
                # Multiply by the dlambda
                # Since the formula was derived for domega, we have to remember that domega = -2*pi*c/lbda**2 * dlbda
                lattice[i,Ns-1-j]= abs(res.real**2+res.imag**2)* (step_i/(lbda_i**2)) * (step_s/(lbda_s**2)) 
                #print angle(res)
                phases[i,Ns-1-j] = angle(res)
                # Check what should be the proper formula which keeps the joint spectral amplitude instead of joint spectral probability distribution
                
        # Apply custom filters:
        # m_filter_signal =zeros((Ni,Ns))
        # m_filter_idler =zeros((Ni,Ns))
        
        # for i in arange(Ni):
            # m_filter_signal[i,:] = self.filter_signal(a_lbda_s)
        # for j in arange(Ns):
            # m_filter_idler[:,j] = self.filter_idler(a_lbda_i)
        # lattice = lattice*m_filter_signal*m_filter_idler
        
        # Multiply by the appropriate missing constants
        lattice = lattice*(c*self.Epulse*self.gamma*(self.L))**2/(2*pi**2) #/ (2*pi*ng)
        Z =  lattice.sum()# sqrt(abs(lattice*conjugate(lattice)).sum())                  
        self.normlattice = sqrt(abs(lattice/Z))         
        self.lattice = lattice 
        self.phases = phases 
    
    
    
    def plotBiphoton(self,fname = None):
        plotcolormap(self.lattice,self.extent,fname)  
        
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
        omega1 = zeros((N,N))
        omega2 = zeros((N,N))
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
        #self.computePcoincfrom2photons()
        lattice = self.normlattice
        N = self.N
        omega1 = zeros((N,N))
        omega2 = zeros((N,N))
        for i in range(N):
            omega1[:,i]= arange(N)
            omega2[i,:]= arange(N)
        #print "State Norm:", abs(lattice*conjugate(lattice)).sum() # equivalent to the trace                
        
        purity = self.vectG(self,omega1,omega2,omega2,omega1).sum()
        #purity2 = self.vectG_nophase(self,omega1,omega2,omega2,omega1).sum()
        # print "Purity: ", purity,purity2
        self.purity = abs(purity)
        self.schn = 1/purity
        """
        print "Computing heralded photon purity"
        N = self.N
        omega1 = zeros((N,N))
        omega2 = zeros((N,N))
        for i in range(N):
            omega1[:,i]= arange(N)
            omega2[i,:]= arange(N)
        
        x = self.vectg(self,arange(N),arange(N))
        print "Tr_ro1: ",x.sum()
        g12 = self.vectg(self,omega1,omega2)
        
        purity = (g12*g12).sum() # no dot product here, the formula (g12*g12).sum() provides exactly the trace over
                                 # the reduced density matrix squared.
                                 
        

        #print schn, schmidtnumber(lattice)
        """
        return abs(purity)
              
###
# -----------------------------------------------------------------------------#
# CLASS FWM_RingSimu 
# -----------------------------------------------------------------------------#
# This class calculates the joint spectral distribution obtained in a ring  
# resonator for a given set of parameters
# Init (
#         * Waveguide cross section 
#         * Waveguide length (Meters) 
#         * Pump power (Watts) 
#         * Pump wavelength (um)
#         * Pulse duration (Seconds)
#         * Repetition rate (Hz)
#         * N: grid sampling (the JSA is stored in a NxN grid)
#         * r: ring coupling (r = 1 means no coupling, while r = 0 means full coupling)
#         * tau: round trip transmission which accounts for the loss in the ring resonator
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
# applycavity(lambda) : This is the function which applies the cavity. By default, this function applies a ring resonator. 
#                       Different cavities can however be used.

# save(filename) : Saves the result of the simulation including all the parameters, the full state, and the derived parameters such as the Schmidt number                                      
#            
class FWM_RingSimu(FWM_Simu):
    def __init__(self,wg = Waveguide(550,220),
                      length = 80., # um 
                      pumppower = 45.*10**-3,pumpwl = 1.55,pulseduration=1.*10**(-12),N = 200,r = 0.98,tau = 1.0): # 300*10**3 -> 300 kHz linewidth
        FWM_Simu.__init__(self,wg = wg,
                      length = length, # 0.03 ->3cm
                      pumppower = pumppower,pumpwl = pumpwl,pulseduration=pulseduration)
        
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
    
    def setTau(self,x):
        self.tau = x
    
    def setr(self,x):
        self.r = x
    
    def setL(self,L):
        self.L = L
    
    def ring(self,lbda):                
        k = self.wg.wl2kv(lbda)
        t = sqrt(1-self.r**2)
        tau = self.tau
        r = self.r
        return I*t/(1-tau*r*exp(I*k*self.L))
    
    def cavity_transmission(self,lbda):
        t = sqrt(1-self.r**2)
        return self.r+I*t*self.ring(lbda)
    
    # Override these methods to add custom filters on signal and idler arm
    def filter_idler(self,lbda):
        return ones(lbda.size)
    
    def filter_signal(self,lbda):
        return ones(lbda.size)
    
    # If using two coupled rings
    def set_r2(self,r2 = 0.999):
        self.r2 = r2
        
    def CROW2(self,lbda):
        k = self.wg.wl2kv(lbda)
        r2 = self.r2
        t2 = sqrt(1-r2**2)
        r1 = self.r
        t1 = sqrt(1-r1**2)
        tau = self.tau
        L1 = self.L
        L2 = L1
        g1 = tau*exp(I*L1*k)
        g2 = tau*exp(I*L2*k)
        return I*t1*(r2-g2)/(1-r2*g2+r1*g1*(g2-r2)) 
    
    def applycavity(self,lbda):                
        return self.ring(lbda)
    
    # Returns the closest cavity resonance for a given lambda and the resonance number
    def getClosestCavityRes(self,lbda):
        m = round(self.wg.wl2kv(lbda)*self.L/(2*pi))
        
        kp0 = m*2*pi/self.L # target pump propagation constant
        # The problem is now to get lbda0 from kp0
        # We start approximating the neff of lbda0 using the one of lambda
        neff = self.wg.getneff(lbda)
        
        # Using a scipy optimize method could be more robust and faster than the following code
        lbda0 = 2*pi*neff/kp0
        print lbda0
        lbdastep = 1*10**(-7) * sign(lbda0-lbda) 
        kp = self.wg.wl2kv(lbda0) 
        err = (kp-kp0)/kp0
        while(abs(err)>0.0000001):
            lbda0 += lbdastep
            kp = self.wg.wl2kv(lbda0) 
            newerr = (kp-kp0)/kp0
            if newerr**2>err**2:
                lbdastep = lbdastep*(-1)
            err = newerr
        return lbda0,m
    # Centers the pump on the closest cavity resonance
    def setPumpToClosestRes(self,lbda):
        self.lbda_p,self.mpump = self.getClosestCavityRes(lbda)        
        print "Pump is set at %.7f um" % self.lbda_p
    
    # Get the range to scan for signal for the nth resonance with respect to the pump
    # Rq : The pump should have been set such that mpump has a meaningful value
    def getSignalRange(self,n):
        FWHM = (1-self.r*self.tau)*self.lbda_p**2/(self.wg.getng(self.lbda_p)*sqrt(2)*pi*self.L)
        print "FWHM (um) : ",FWHM
        fullrange = 5*FWHM # 
        wlFSR = self.lbda_p**2/(self.L*self.wg.getng(self.lbda_p)) # FSR in lambda
        print "FSR (um) : ",wlFSR
        lbda_s,m = self.getClosestCavityRes(self.lbda_p+n*wlFSR)
        print "Resonance (um) : ",lbda_s
        return lbda_s-fullrange/2,lbda_s+fullrange/2
    
    def plotcavityresponse(self,albda = arange(1.5477-0.01,1.5477+0.01,0.0000001)):
        cavity = self.applycavity(albda)*self.applycavity(albda).conjugate()
        pump = self.pumpenvelop(albda)**2
        lbda_i,m_i = self.getClosestCavityRes(1.548)
        lbda_s = 1./(2./self.lbda_p-1./lbda_i)
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
        
    def updatepumprange(self):
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
        N = 400
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
    
    def computeJS(self): # begin=1.545,end=1.555,step=0.0001
        print self.wg.getng(self.lbda_p)
        P = self.pumppower
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
        cav_resp_p1 = self.applycavity(a_lbda_p1)
        a_p1 = self.pumpenvelop(a_lbda_p1) # pump amplitude 1
        ng = self.wg.getng(self.lbda_p)
        print "Steps" ,step_i,step_s
        #dbgpm = 0.
        pumpmax = self.pumpenvelop(self.lbda_p)
        phases = zeros((Ni,Ns))
        for j in xrange(Ns):
            print j
            lbda_s = a_lbda_s[j] # lbda_s_min+j*step_s
            cav_resp_s = self.applycavity(lbda_s)
            for i in xrange(Ni):
                lbda_i = a_lbda_i[i] # lbda_i_min+i*step_i
                a_lbda_p2 = 1./(1./lbda_s+1./lbda_i-1./a_lbda_p1)                                
                a_p2 = self.pumpenvelop(a_lbda_p2) # pump amplitude 2
                
                # In order to save computation time we can take a_pm = 1. for small cavities
                a_dk = self.getdk(P*a_p1*conjugate(a_p1),P*a_p2*conjugate(a_p2),a_lbda_p1,a_lbda_p2,lbda_s,lbda_i)
                a_pm =  sinc(L/2*a_dk/pi)  # the L will be added later in the global constant               
                
                #a_pm = 1.
                a_res = a_p1*a_p2*a_pm*cav_resp_p1*self.applycavity(a_lbda_p2)* self.applycavity(lbda_i)*cav_resp_s # 
                a_res = a_res * a_lbda_p2/a_lbda_p1
                # Multiply by the dlambda; 
                # The pump function is i um^(-1/2), dlbda_pump is in um 
                a_res = a_res*dlbda_pump
                res = a_res.sum() # unitless
                #res = res 
                # Multiply by the dlambda
                # Since the formula was derived for domega, we have to remember that domega = -2*pi*c/lbda**2 * dlbda
                lattice[i,Ns-1-j]= abs(res.real**2+res.imag**2)* (step_i/(lbda_i**2)) * (step_s/(lbda_s**2)) 
                #print angle(res)
                phases[i,Ns-1-j] = angle(res)
                # Check what should be the proper formula which keeps the joint spectral amplitude instead of joint spectral probability distribution
                
        # Apply custom filters:
        # m_filter_signal =zeros((Ni,Ns))
        # m_filter_idler =zeros((Ni,Ns))
        
        # for i in arange(Ni):
            # m_filter_signal[i,:] = self.filter_signal(a_lbda_s)
        # for j in arange(Ns):
            # m_filter_idler[:,j] = self.filter_idler(a_lbda_i)
        # lattice = lattice*m_filter_signal*m_filter_idler
        
        # Multiply by the appropriate missing constants
        lattice = lattice*(c*self.Epulse*self.gamma*(self.L))**2/(2*pi**2) #/ (2*pi*ng)
        Z =  lattice.sum()# sqrt(abs(lattice*conjugate(lattice)).sum())                  
        self.normlattice = sqrt(abs(lattice/Z))         
        self.lattice = lattice 
        self.phases = phases 
        xi =  2*lattice.sum() 
        xi = tanh(sqrt(xi))**2 # Approximation valid in the case of two-mode squeezer
        self.probapair = xi * (1-xi)
        # Theory calculation for CW regime for comparison
        
        vg = c/self.wg.getng(self.lbda_p)
        print "Epulse (nJ) ", self.Epulse*10**9
        print "gamma W-1,m-1", self.gamma
        print "L (um)", L
        print "T (ps)", self.T*10**12
        print "vg %e" % vg
        print "r : %.4f" % self.r
        print "tau : %.4f" % self.tau
        
        print "Siglbda : %.5f" % (self.siglbda)
        #deltalbda = self.siglbda*sqrt(2*pi) # Such that the approx rectangular pulse results matches the gaussian def
        #beta2_pulsed = (self.Epulse*self.gamma*c)**2/(32*ng**4*pi**6)*self.lbda_p**4/(L**2*deltalbda**2)*(1-self.r**2)**4/(1-self.tau*self.r)**4
        xi = (self.Epulse*self.gamma*c)**2/(32*ng**4*pi**2)*self.lbda_p**4*pumpmax**4/(L**2)*(1-self.r**2)**4/(1-self.tau*self.r)**4
        
        xi = tanh(sqrt(xi))**2
        beta2_pulsed = xi * (1-xi)
        #beta2_pulsed = (self.Epulse*self.T*self.gamma/(L*10**(-6)))**2*vg**4/16.*(1-self.r**2)**4/(1-self.tau*self.r)**4
        xi = self.gamma**2*self.pumppower**2*(L*10**(-6))/8 * vg*self.T*(1-self.r**2)**4/(1-self.r*self.tau)**7        
        xi = tanh(sqrt(xi))**2
        beta2_CW = xi * (1-xi)
        
        # We multiply the lattice by a factor of two since we only integrate over half of Phi(k1,k2) and we should account for the other symmetrical half
        print "Nb pairs per pulse:",self.probapair
        print "Flat pulse model:", beta2_pulsed
        print "CW model:", beta2_CW
        lbda_i0 = (lbda_i_max+lbda_i_min)/2
        lbda_s0 = (lbda_s_max+lbda_s_min)/2
        self.extent = list(array([lbda_i_min-lbda_i0,lbda_i_max-lbda_i0,lbda_s_min-lbda_s0,lbda_s_max-lbda_s0])*1000) # Check where should go i and s
        self.beta2_pulsed = beta2_pulsed
        self.beta2_CW = beta2_CW
    
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
        fw.write("%s : %.4f\n" % ("Pump power avg (mW)",self.pumppower*1000))
        fw.write("%s : %.3f\n" % ("Repetition rate(MHz)",self.reprate/(10**6)))
        fw.write("%s : %.18e\n" % ("Energy per pulse (uJ)",self.Epulse*1000000))
        fw.write("%s : %.6f\n" % ("Pump wavelength (um)",self.lbda_p))
        fw.write("\n#Waveguide parameters\n")
        fw.write("%s : %.3f\n" % ("Width (nm)",self.wg.width))
        fw.write("%s : %.3f\n" % ("Height (nm)",self.wg.height))
        fw.write("%s : %.3f\n" % ("Aeff (um^2)",self.wg.Aeff))
        fw.write("%s : %.3f\n" % ("gamma (W-1 m-1)",self.gamma))
        fw.write("\n#Ring parameters\n")
        fw.write("%s : %.3f\n" % ("Cavity length (um)",self.L))
        fw.write("%s : %.5f\n" % ("Tau",self.tau))
        fw.write("%s : %.5f\n" % ("r",self.r))
        fw.write("\n#BiPhoton state properties\n")
        fw.write("%s : %.5f\n" % ("Nb pairs per pulse",self.probapair))
        fw.write("%s : %.5f\n" % ("Flat pulse model",self.beta2_pulsed))
        fw.write("%s : %.5f\n" % ("CW model",self.beta2_CW))
        self.computeHeraldedPhotonPurity()
        #self.computePcoincfrom2photons()
        #fw.write("%s : %.6f\n" % ("Visibility from two heralded sources",self.visibility))
        fw.write("%s : %.6f\n" % ("Schmidt number",abs(self.schn)))
        fw.write("%s : %.6f\n" % ("Purity",abs(1/self.schn)))
        
        # Theory calculation for CW regime for comparison
        vg = c/self.wg.getng(self.lbda_p)
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
       
class CustomPump():
    def __init__(self,fname="G2 Straight Transmission.csv"):                                
        self.rootname = "."
        self.__load__(fname)        
        self.__fit__() 
        
    def __load__(self,fname):
        path = os.path.join(self.rootname,fname)
        f = open(path)
        line = f.readline()
        lbdas = []
        amplitudes = []
        for i in arange(30):
            line = f.readline()
            
        while(len(line))>0:
            splitted = line.split(",")                  
            lbda,amplitude = splitted[0:2] 
            
            line = f.readline()
            if lbda>0:
                lbdas.append(float(lbda)/1000) # nm -> um
                amplitudes.append(float(amplitude))
        
        self.lbdas = array(lbdas)
        self.amplitudes = array(amplitudes)
        self.amplitudes = self.amplitudes/self.amplitudes.sum() # Normalise
        self.lbda_p = self.lbdas[self.amplitudes.argmax()]                        
        
    def __fit__(self):
        # Gaussian multiplied by rational fraction to account for distorsion
        
        a = (10**3)
        b = (10**3)
        c = (10**3)**1.5
        d = 10
        e = 1
        f = 1
        sig = 1.0*10**(-3) # um
        p0 = [self.lbda_p,sig,a,b,c,d,e,f]      
        plsq = leastsq(self.__residuals__, p0, args=(self.amplitudes, self.lbdas)) 
        self.p = plsq[0]        
        print self.p
    # p : parameters
    # lbdas : wavelengths
    def __func__(self,p,lbdas):         
        lbda0,sig,a,b,c,d,e,f = p
        dlbdas = lbdas-lbda0
        res = exp(-dlbdas**2/(2*sig**2))*(a*dlbdas+f/(b*dlbdas**3+c*dlbdas**2+d*dlbdas+e))
        return res
        
    def __residuals__(self,p,y, x):
        err = y-self.__func__(p,x)
        return err     
    
    def getPulse(self,lbda):    
        return self.__func__(self.p,lbda)
    
    def plotres(self):
        lbda1,lbda2 = min(self.lbdas),max(self.lbdas)
        x = arange(lbda1,lbda2,0.000001)        
                
        #self.p = (A,r,tau)
        plots = [(self.lbdas,self.amplitudes,"ro"),(x,self.getPulse(x),"k-")] # (neff0 self.lbdas,self.Iouts,"ro"),
        #plot(plots)            
        print self.lbda_p        
        return plots
        
# Fit ring when seeded by a pulse laser from which we know the shape        
class RingPulsed():
    def __init__(self,R,Lc,fname,pumpfunc):
        self.R = R # radius (um)
        self.Lc = Lc # coupling length (um)
        self.L = 2*(pi*R + Lc) # Total length
        #FSR = 1.5556-1.5477 # um
        self.neff0 = 4.14330  #4.143277 # Starting effective group index  4.1434        
        self.pumpfunc = pumpfunc
        self.rootname = "."
        self.__load__(fname)
        
        self.__fit__() 
        
    def __load__(self,fname):        
        path = os.path.join(self.rootname,fname)
        f = open(path)
        line = f.readline()
        lbdas = []
        amplitudes = []
        for i in arange(30):
            line = f.readline()
            
        while(len(line))>0:
            splitted = line.split(",")                  
            lbda,amplitude = splitted[0:2] 
            
            line = f.readline()
            if lbda>0:
                lbdas.append(float(lbda)/1000) # nm -> um
                amplitudes.append(float(amplitude))
        
        self.lbdas = array(lbdas)
        self.amplitudes = array(amplitudes)
        self.amplitudes = self.amplitudes/self.amplitudes.sum() # Normalise
        self.lbda_p = self.lbdas[self.amplitudes.argmin()] 
        # adjust the neff0 guess
        m = int(self.neff0*self.L/self.lbda_p)
        self.neff0 = m*self.lbda_p/self.L
        
    def __fit__(self):
        
        a = b = c = d=e=f=0.000000000000001
        p0 = [max(self.amplitudes),0.9,0.9,self.neff0,a,b,c,d,e,f]      
        plsq = leastsq(self.__residuals__, p0, args=(self.amplitudes, self.lbdas)) 
        self.p = plsq[0]        
        print self.p
    # p : parameters
    # lbdas : wavelengths
    def __func__(self,p,lbdas):         
        A,r,tau,neff,a,b,c,d,e,f = p         
        dlbdas = lbdas-self.lbda_p
        #neff = self.neff0
        L = self.L 
        phi = 2*pi*L*neff/lbdas
        r2 = r**2
        tau2 = tau**2        
        K = 2*r*tau*cos(phi)
        res = A*(r2+tau2-K)/(1+r2*tau2-K) * self.pumpfunc(lbdas) * (a+b*dlbdas+c*dlbdas**3)/(d+e*dlbdas+f*dlbdas**3)
        return res
    
    def ringResponse(self,p,lbdas):         
        A,r,tau,neff,a,b,c,d,e,f = p         
        dlbdas = lbdas-self.lbda_p
        #neff = self.neff0
        L = self.L 
        phi = 2*pi*L*neff/lbdas
        r2 = r**2
        tau2 = tau**2        
        K = 2*r*tau*cos(phi)
        res = A*(r2+tau2-K)/(1+r2*tau2-K) * (a+b*dlbdas+c*dlbdas**3)/(d+e*dlbdas+f*dlbdas**3)*max(self.pumpfunc(lbdas))
        return res
        
    def __residuals__(self,p,y, x):
        err = y-self.__func__(p,x)
        return err     
    
    def getIout(self,lbda):    
        return self.__func__(self.p,lbda)
    
    def plotres(self):
        lbda1,lbda2 = min(self.lbdas),max(self.lbdas)
        x = arange(lbda1,lbda2,0.000001)                                     
        plots = [(self.lbdas,self.amplitudes,"bo"),(x,self.getIout(x),"k-"),(x,self.ringResponse(self.p,x),"b--")] # (self.lbdas,self.Iouts,"ro"),
        #plot(plots)    
        self.lbda_p = self.lbdas[self.amplitudes.argmin()]
        print self.lbda_p
        return plots
        
    # December 15, 2004 / Vol. 29, No. 24 / OPTICS LETTERS p 2861
    # Ultrahigh-quality-factor silicon-on-insulator microring resonator
    def computeQ(self):
        A,r,tau,neff=self.p[0:4]
        return (2*pi*neff/self.lbda_p)*self.L/(-2*log(r*tau))
        
def main():
    
    # Load the pulse
    #pump = CustomPump("G2 Straight Transmission.csv")
    #pump.plotres()
    #pumpfunc = pump.getPulse
    
    wg = Waveguide(450,220)  
    T = 100.*10**(-12)
    #for T in [100.,50.,25.,10.,5.]: 
    N = 100 # 200# N = 50 Provides accurate number for r = 0.98 rings with 100ps pulses
    #for T in [1000.,500.,200.,100.,50.,25.,10.]:
    r = 0.93
    tau = 1.-0.0198
    radius = 10.
    coupling_length = 5.
    lbda0= 1.55
    res_number = 1 # resonance number (pump resonance is 0).
    for res_number in [1]: #arange(0,1):# [1,2,3,4]:
        for T in [5.0] : # ,0.75,1.,1.5,2.0,0.5,1.,5.,,50.,100.,500.,1000.,2000. #arange(10.,1000,10.): # [60.,70.,80.,90.,110.,120.,130.,140.,150.,160.,170.,180.,190.,210.,220.,230.,240.,250.,260.,270.,280.,290.]: #arange(10.,100.,10.): # arange(5,55,5): #[25.,50.,100.,200.,500.]: [1.0,2.0,5.0,10.0,20.0,50.0,100.0,200.0,500.0,1000.0,]
            for r in [0.9]: # [0.95,0.96,0.97,0.98,0.99]: # 0.85,0.86,0.87,0.88,0.89,0.90,0.91,0.92,0.93,0.94,0.95,0.96
                for tau in [0.997]: # 0.76,0.96,0.98
                    #for r2 in [0.9998,0.9997,0.9996,0.9995,0.9994]: #[1.0,0.9999,0.999,0.99]:
                    mySim =FWM_RingSimu(wg,length = 2*(radius*pi+coupling_length),pulseduration = T*10**(-12),N = N,r = r,tau = tau,pumppower = 3.*10**-3,pumpwl = lbda0) # 500 
                    #mySim.pumpenvelop = pumpfunc
                    
                    mySim.setRangeScanResonance(+res_number)                    
                    mySim.plotcavityresponse()                    
                    mySim.updatepumprange()
                    mySim.computeJS()                
                    fname = mySim.save("Ring_pumpscan")
                    mySim.plotBiphoton(fname[:-3]+"png")

# -----------------------------------------------------------------------------#    
# MISC FUNCTIONS II: Specific FWM applications
# -----------------------------------------------------------------------------#        
def plot1Dgain():
    wgs = [
            #Waveguide(450,220),
            Waveguide(470,220)
            #Waveguide(500,220),
            #Waveguide(550,220),
          ]
    plots = []
    colors = ["r-","b-","g-"]
    i = 0
    lbda_s = arange(1.40,1.70,0.0001)    
    for wg in wgs:
        simu = FWM_Simu(wg = wg,length = 0.0058,pumpwl = 1.5479)
        res = simu.toplotCWGain(lbda_s)
        plots.append((lbda_s,res,colors[i]))
        i += 1    
    
    fw = open("fwm_bandwidth_cw.csv","w")
    fw.write("Wavelength (um), FWM gain (a.u)")
    for i in arange(lbda_s.size):
        line = "%.5f,%.5f\n" % (lbda_s[i],res[i])
        fw.write(line)
    fw.close()
    plot(plots)

def plotnbpairsScaling():
    lbda_min =  1.542
    lbda_max = 1.544
   
    wg = Waveguide(550,220)
    lbda_s = arange(1.5,1.6,0.0001)
    tointegrate = (lbda_s>lbda_min) * (lbda_s<lbda_max)
    lengths = arange(0,0.01,0.0001)
    #lengths = arange(0,100.,0.1)
    res = []
    for L in lengths:
        simu = FWM_Simu(wg = wg,length = L )
        gainperbandwidth = (L/2)**2*simu.toplotCWGain(lbda_s = lbda_s) # 
        #res.append(gainperbandwidth[tointegrate].sum())
        res.append(gainperbandwidth.sum())
    
    plot([(lengths,res,"r-")])   
    
if __name__ == "__main__": 
    #pump = CustomPump("G2 Straight Transmission.csv")
    #pump.plotres()
    #ring = RingPulsed(20,5,"G2 Ring Transmission.csv",pump.getPulse)
    #plot(ring.plotres()+pump.plotres())
    main()
    #plotnbpairsScaling()
    #plot1Dgain()
