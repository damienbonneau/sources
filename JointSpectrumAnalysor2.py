import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np

from PhaseMatchingBiphotonFWM import *
app = QtGui.QApplication([])

class DoubleBusRing():
    def __init__(self,r1,r2,tau,L): 
        Component.__init__(self)
        self.r1 = r1
        self.r2 = r2
        self.t1 = sqrt(1-r1**2)
        self.t2 = sqrt(1-r2**2)
        self.tau = tau
        self.L = L
        self.ng = 4.2
     
    def set_r1(self,x):
        self.r1 = x
    
    def set_r2(self,x):
        self.r2 = x
    
    def set_tau(self,x):
        self.tau = x
    
    def set_L(self,x):
        self.L = x
    
    def theta(self,lbda):
        return 2*pi*self.L*self.ng/lbda
    
    def getThrough(self,lbda):
        r1 = self.r1
        r2 = self.r2
        tau = self.tau
        eth = exp(I*self.theta(lbda))
        return (r1-r2*tau*eth)/(1-r1*r2*tau*eth)
        
    def getDrop(self,lbda):
        r1 = self.r1
        r2 = self.r2
        t1 = self.t1
        t2 = self.t2
        
        tau = self.tau
        th = self.theta(lbda)
        eth = exp(I*th)
        return -t1*t2*sqrt(tau)*exp(I*th/2)/(1-r1*r2*tau*eth)    


def dbg_img(N):
    x0 = N/2
    y0 = N/2
    a = np.ones((N,N))
    b = np.ones((N,N))
    for i in xrange(N):
        b[i,:] = 10*exp(-(arange(N)-x0)**2/0.01)
    
    for i in xrange(N):
        b[:,i] = 10*exp(-(arange(N)-x0)**2/0.01)
    print "MAX"
    c = 255*b/b.max()
    
    return c
    
class Main():
    def __init__(self):                                                
        #mySim.plotcavityresponse()                    
        #mySim.computeJS()                
        #fname = mySim.save("Ring_pumpscan")
        
        self.wg = Waveguide(450,220)         
        self.N = 200 # 50
        self.r = 0.98
        self.tau = 0.98
        self.pumppower = 3.*10**-3
        self.pumpwl = 1.55
        self.sim = None
        self.lbda_min0 = 1.53 # um
        self.lbda_max0 = 1.57 # um
        lbda_mid = (self.lbda_min0+self.lbda_max0)/2
        self.dlbda = self.lbda_max0-self.lbda_min0
        self.lbda_step = 0.000002 # um
        self.albda = arange(self.lbda_min0,self.lbda_max0+self.lbda_step,self.lbda_step)
        self.length = 2*(10.*pi+5.)
        self.pulseduration = 1.*10**(-12)
        
        # Widget add-on to get values from the plot
        self.signal_region = pg.LinearRegionItem([self.lbda_max0-self.dlbda/10,self.lbda_max0])
        self.signal_region.setZValue(-10)
        
        self.idler_region = pg.LinearRegionItem([self.lbda_min0,self.lbda_min0+self.dlbda/10])
        self.idler_region.setZValue(-10)
        
        self.sim = FWM_RingSimu(self.wg,
                                length = self.length,
                                pulseduration = self.pulseduration,
                                N = self.N,
                                r = self.r,
                                tau = self.tau,
                                pumppower = self.pumppower,
                                pumpwl = self.pumpwl
                                ) 
        
        self.update()
        self.sim.updatepumprange()        
        self.updateAfterChanged()
        self.sim.computeHeraldedPhotonPurity()
        
        ###
        # Plots
        ###
        
        self.plot_pumpcavity = pg.PlotWidget(title="Cavity response and pump") 
        self.c_plot_cavity = self.plot_pumpcavity.plot(self.albda,self.cavity_response, pen=(255,0,0),fillLevel=0. )                               
        self.c_plot_pump = self.plot_pumpcavity.plot(self.albda,self.pump, pen=(0,0,255),fillLevel=0.,fillBrush=(200,200,255,100) )                               
        
        self.plot_pumpcavity.addItem(self.signal_region)
        self.plot_pumpcavity.addItem(self.idler_region)
                
        self.zoom_signal = pg.PlotWidget(title="Zoom on signal")
        self.z_plot_cavity = self.zoom_signal.plot(self.albda,self.cavity_response, pen=(255,0,0),fillLevel=0.)
        self.z_plot_pump =self.zoom_signal.plot(self.albda,self.pump,fillLevel=0.,fillBrush=(200,200,255,100))
                           
        self.signal_region.sigRegionChanged.connect(self.updateZoomSignal)
        self.zoom_signal.sigXRangeChanged.connect(self.updateRegionSignal)
        self.updateZoomSignal()
        
        self.zoom_idler = pg.PlotWidget(title="Zoom on idler")
        self.z_idler_plot_cavity = self.zoom_idler.plot(self.albda,self.cavity_response, pen=(255,0,0),fillLevel=0.)
        self.z_idler_plot_pump =self.zoom_idler.plot(self.albda,self.pump,fillLevel=0.,fillBrush=(200,200,255,100))
                           
        self.idler_region.sigRegionChanged.connect(self.updateZoomIdler)
        self.zoom_idler.sigXRangeChanged.connect(self.updateRegionIdler)
        self.updateZoomIdler()
        
        # Additional filter plots
        # Green : relevant filter  
        # Red : transmission of the source ring; help lining up the filter
        # Grey : filter response of the other channel (to check that the filter idler filter does not affect the signal response (and vice versa))
        
        self.plot_idler_filter = pg.PlotWidget(title="Idler filter")
        self.c_if_source = self.plot_idler_filter.plot(self.albda,log10(self.cavity_transmission), pen=(255,0,0),fillLevel=0.)
        self.c_if_idler_filter = self.plot_idler_filter.plot(self.albda,log10(self.filter_idler), pen=(0,255,0),fillLevel=0.)
        self.c_if_signal_filter = self.plot_idler_filter.plot(self.albda,log10(self.filter_signal), pen=(100,100,100),fillLevel=0.)
        #self.plot_idler_filter.sigXRangeChanged.connect(self.updateRegionIdler)
        
        self.idler_region.sigRegionChanged.connect(self.updateIdlerFilter)
        self.updateIdlerFilter()
        
        self.plot_signal_filter = pg.PlotWidget(title="Signal filter")
        self.c_sf_source = self.plot_signal_filter.plot(self.albda,self.cavity_transmission, pen=(255,0,0),fillLevel=0.)
        self.c_sf_signal_filter = self.plot_signal_filter.plot(self.albda,self.filter_signal, pen=(0,255,0),fillLevel=0.)
        self.c_sf_idler_filter = self.plot_signal_filter.plot(self.albda,self.filter_idler, pen=(100,100,100),fillLevel=0.)
        #self.plot_signal_filter.sigXRangeChanged.connect(self.updateRegionSignal)
        
        self.signal_region.sigRegionChanged.connect(self.updateSignalFilter)                
        self.updateSignalFilter()
        
        # 2D plot joint spectral probability density
        
        data = self.sim.getjointprobascaled()
        imv = pg.ImageView()  
        #img1a = pg.ImageItem(arr)
        imv.setImage(data)
        imv.setLevels(data.min(), data.max())
        
        imv.show()
        
        self.JSplot = imv                                 
            
        self.JScontours = []
        
        #data = self.sim.getjointprobascaled()
        levels = np.linspace(0, 1, 20)
        for i in range(len(levels)):
            v = levels[i]
            ## generate isocurve with automatic color selection
            c = pg.IsocurveItem(level=v, pen=(i, len(levels)*1.5))
            c.setParentItem(self.JSplot.getImageItem())  ## make sure isocurve is always correctly displayed over image
            c.setZValue(10)
            self.JScontours.append(c)               
        
        # 2D plot joint spectral phase
        data = self.sim.getPhases()
        imv = pg.ImageView()        
        imv.setImage(data)
        imv.setLevels(data.min(), data.max())
        imv.show()
        self.JSPhasePlot = imv                                 
            
        self.JSPhaseContours = []
        #data = self.sim.getjointprobascaled()
        levels = np.linspace(-3.14, 3.14, 10)
        for i in range(len(levels)):
            v = levels[i]
            ## generate isocurve with automatic color selection
            c = pg.IsocurveItem(level=v, pen=(i, len(levels)*1.5))
            c.setParentItem(self.JSPhasePlot.getImageItem())  ## make sure isocurve is always correctly displayed over image
            c.setZValue(10)
            self.JSPhaseContours.append(c)               
        
        # results 
        
        results_layout=QtGui.QGridLayout() 
        results_widget=QtGui.QWidget()  
        results_widget.setLayout(results_layout)                
                
        self.results = [
                        ("Purity","%.4f" , self.sim.purity),
                        ("Schmidt number","%.2f" , self.sim.schn),
                        ("Nb pairs per pulse","%.3e" , self.sim.probapair),
                        ("Flat pulse model","%.3e" , self.sim.beta2_pulsed),
                        ("CW model","%.3e" , self.sim.beta2_CW)
                        ]
        
        i = 0   
        j = 0
        results_value_labels = []
        for txt,fmt,value in self.results:
            value_lbl = QtGui.QLabel(fmt % value)
            txt_lbl = QtGui.QLabel(txt)
            results_layout.addWidget(txt_lbl,i,0+2*j)    
            results_layout.addWidget(value_lbl,i,1+2*j)    
            results_value_labels.append(value_lbl)
            i+=1
            if i == 4:
                i = 0
                j+=1                        
        self.results_widget = results_widget
        self.results_value_labels = results_value_labels
        
        
    def save(self):
        self.sim.save(directory = "simulations")
    
    def updateResults(self):
        self.sim.computeHeraldedPhotonPurity()
        
        new_res = [self.sim.purity,self.sim.schn, self.sim.probapair,self.sim.beta2_pulsed,self.sim.beta2_CW]
        i = 0
        for _,fmt,_ in self.results:
            self.results_value_labels[i].setText(fmt % new_res[i])        
            i += 1
            
    def updateJSplot(self):
        data = self.sim.getjointprobascaled()
        #levels = np.linspace(data.min(), data.max(), 10)
        self.JSplot.setImage(data)
        self.JSplot.setLevels(data.min(), data.max())
        for c in self.JScontours:
            c.setData(data)
            
    def updateJSPhasePlot(self):
        data = self.sim.getPhases()
        #levels = np.linspace(data.min(), data.max(), 10)
        self.JSPhasePlot.setImage(data)
        self.JSPhasePlot.setLevels(data.min(), data.max())
        for c in self.JSPhaseContours:
            c.setData(data)
    
    #def updateIdlerRegion(self):
    #    m,M = self.idler_region.getRegion()
    
    def updateZoomSignal(self):
        m,M = self.signal_region.getRegion()
        step = self.lbda_step*0.5
        albda2 = arange(m,M+step,step)
        self.zoom_signal.setXRange(*self.signal_region.getRegion(), padding=0)
        
        cav = abs(self.sim.applycavity(albda2)*self.sim.applycavity(albda2).conjugate())
        pump = self.sim.pumpenvelop(albda2)**2
        
        self.z_plot_pump.setData(albda2,pump)
        self.z_plot_cavity.setData(albda2,cav)        
        
    def updateZoomIdler(self):
        m,M = self.idler_region.getRegion()
        step = self.lbda_step*0.5
        albda2 = arange(m,M+step,step)
        self.zoom_idler.setXRange(*self.idler_region.getRegion(), padding=0)
        
        cav = abs(self.sim.applycavity(albda2)*self.sim.applycavity(albda2).conjugate())
        pump = self.sim.pumpenvelop(albda2)**2        
        self.z_idler_plot_pump.setData(albda2,pump)
        self.z_idler_plot_cavity.setData(albda2,cav)
        
    def updateIdlerFilter(self):
        m,M = self.idler_region.getRegion()        
        step = self.lbda_step*0.5
        albda2 = arange(m,M+step,step)
        self.plot_idler_filter.setXRange(*self.idler_region.getRegion(), padding=0)
        
        cavity_transmission = abs(self.sim.cavity_transmission(albda2)*self.sim.cavity_transmission(albda2).conjugate())
        filter_idler = abs(self.sim.filter_idler(albda2)*self.sim.filter_idler(albda2).conjugate())
        filter_signal = abs(self.sim.filter_signal(albda2)*self.sim.filter_signal(albda2).conjugate())
        
        self.c_if_source.setData(albda2,10*log10(cavity_transmission))
        self.c_if_idler_filter.setData(albda2,10*log10(filter_idler))
        self.c_if_signal_filter.setData(albda2,10*log10(filter_signal))                
    
    def updateSignalFilter(self):
        m,M = self.signal_region.getRegion()
        step = self.lbda_step*0.5
        albda2 = arange(m,M+step,step)
        self.plot_signal_filter.setXRange(*self.signal_region.getRegion(), padding=0)
        
        cavity_transmission = abs(self.sim.cavity_transmission(albda2)*self.sim.cavity_transmission(albda2).conjugate())
        filter_signal = abs(self.sim.filter_signal(albda2)*self.sim.filter_signal(albda2).conjugate())
        filter_idler = abs(self.sim.filter_idler(albda2)*self.sim.filter_idler(albda2).conjugate())
                
        self.c_sf_source.setData(albda2,10*log10(cavity_transmission))
        self.c_sf_signal_filter.setData(albda2,10*log10(filter_signal))
        self.c_sf_idler_filter.setData(albda2,10*log10(filter_idler))        
        
                
    def updateRegionSignal(self):        
        self.signal_region.setRegion(self.zoom_signal.getViewBox().viewRange()[0])
    def updateRegionIdler(self):        
        self.idler_region.setRegion(self.zoom_idler.getViewBox().viewRange()[0])
    
    
    def setLbdaMin0(self,x):    
        self.lbda_min0 = x
    
    def setLbdaMax0(self,x):    
        self.lbda_max0 = x        
     
    def setLength(self,x):
        self.length = x
        self.sim.setL(self.length)
    
    def setPumpWl(self,x):
        self.pumpwl = x
        self.sim.setPumpwl(x)
    
    def setPulseDuration(self,x):
        self.pulseduration = x
        self.sim.setPulseDuration(x)
    
    def setGridResolution(self,x):
        self.N = x
    
    def setr(self,x):
        self.r = x
        self.sim.setr(x)
    
    def settau(self,x):
        self.tau = x
        self.sim.setTau(x)
    
    def setPumpPower(self,x):
        self.pumppower = x
    
    def setPumpWavelength(self,x):
        self.pumpwl = x
    
    def update(self):                                        
        self.albda = arange(self.lbda_min0,self.lbda_max0+self.lbda_step,self.lbda_step)
        self.cavity_response = abs(self.sim.applycavity(self.albda)*self.sim.applycavity(self.albda).conjugate())
        self.cavity_transmission = abs(self.sim.cavity_transmission(self.albda)*self.sim.cavity_transmission(self.albda).conjugate())
        self.filter_idler = abs(self.sim.filter_idler(self.albda)*self.sim.filter_idler(self.albda).conjugate())
        self.filter_signal = abs(self.sim.filter_signal(self.albda)*self.sim.filter_signal(self.albda).conjugate())
        
        self.pump = self.sim.pumpenvelop(self.albda)**2
        self.pump = self.pump /self.pump.max()*self.cavity_response.max()
        
        lsm,lsM = self.signal_region.getRegion()
        lim,liM = self.idler_region.getRegion()
        lp = self.sim.lbda_p
        #lim = 1./(2./lp - 1./lsM)
        #liM = 1./(2./lp - 1./lsm)        
        self.sim.setRangeJS(lsm,lsM,lim,liM)
               
    def updateAfterChanged(self): # 
        print "Updating JS..."
        self.sim.computeJS()        
        print "Done"
        
    def updatePlots(self):                            
        #self.plot_ptrig.plot(self.xi2s,self.ptrig, pen=(255,0,0))
        for curve,data in [(self.c_plot_cavity,self.cavity_response),
                           (self.c_plot_pump,self.pump),                           
            ]:
            #curve = plot0.plot(pen='y') 
            curve.setData(self.albda,data)
            
        self.plot_pumpcavity.setXRange(self.lbda_min0, self.lbda_max0, padding=0, update=True)
        self.updateZoomIdler()
        self.updateZoomSignal()
        self.updateIdlerFilter()
        self.updateSignalFilter()
        self.updateJSplot()
        self.updateJSPhasePlot()
        #self.JSplot.setImage(self.sim.lattice)
        #self.JSplot.setLevels(self.sim.lattice.min(), self.sim.lattice.max())
        #print self.sim.data

src = Main()

def main():
    print "Starting main"
    global src 
    win = QtGui.QMainWindow()    
    cw = QtGui.QWidget()
    layout = QtGui.QGridLayout()
    cw.setLayout(layout)
    win.setCentralWidget(cw)
    win.show()    
    
    win.setCentralWidget(cw)
    
    
    pg.setConfigOptions(antialias=True)
    win.resize(1200,600)
    win.setWindowTitle('JSAnalysor')   
    
    layout.addWidget(src.plot_pumpcavity,0,0)
    layout.addWidget(src.JSplot,0,1)
    layout.addWidget(src.JSPhasePlot,0,2)
    layout.addWidget(src.zoom_idler,1,0)
    layout.addWidget(src.zoom_signal,1,1)    
    #layout.addWidget(src.plot_idler_filter,1,3)
    #layout.addWidget(src.plot_signal_filter,1,2)

    # Widgets 
    # Spin widgets
    
    def valueChanged_default(sb):
        pass
        #src.updateAfterChanged()
        #src.updatePlots()
        #src.updateResults()
        
    # lbda min
    spin_lbda_min = pg.SpinBox(value=1.530, suffix='',step=0.001, minStep=0.0001)    
    spin_lbda_min.setMinimum(0.0)
    
    def valueChanging_lbdamin0(sb, value):
        src.setLbdaMin0(value) 
        src.update() 
        src.updatePlots() 
    
    spin_lbda_min.sigValueChanging.connect(valueChanging_lbdamin0)    
    spin_lbda_min.sigValueChanged.connect(valueChanged_default)    
    
    # lbda max
    spin_lbda_max = pg.SpinBox(value=1.570, suffix='',step=0.001, minStep=0.0001)    
    spin_lbda_max.setMinimum(0.0)
    
    def valueChanging_lbdamax0(sb, value):
        src.setLbdaMax0(value) 
        src.update() 
        src.updatePlots() 
  
    spin_lbda_max.sigValueChanging.connect(valueChanging_lbdamax0)    
    spin_lbda_max.sigValueChanged.connect(valueChanged_default)            
    
    # Pump width    
    spin_pumpwidth = pg.SpinBox(value=1., suffix='ps',step=0.1, minStep=0.01)    
    spin_pumpwidth.setMinimum(0.0)   
    
    def valueChanged_pumpduration(sb):                
        src.sim.updatepumprange()        
        #src.updateAfterChanged()
        src.updatePlots()
        #src.updateResults()
    
    def valueChanging_pumpduration(sb, value):
        src.setPulseDuration(value*10**(-12))         
        src.update() 
        src.updatePlots()         
        
    
    spin_pumpwidth.sigValueChanged.connect(valueChanged_pumpduration)    
    spin_pumpwidth.sigValueChanging.connect(valueChanging_pumpduration)        
    
    # Pump wavelength    
    spin_pumpwl = pg.SpinBox(value=1.55, suffix='um',step=0.0001, minStep=0.000001)    
    spin_pumpwl.setMinimum(0.0)    
    
    def valueChanged_pumpwl(sb):                
        src.sim.updatepumprange()        
        #src.updateAfterChanged()
        src.updatePlots()
        #src.updateResults()
    
    def valueChanging_pumpwl(sb, value):
        src.setPumpWl(value)         
        src.update() 
        src.updatePlots() 
        
    
    spin_pumpwl.sigValueChanging.connect(valueChanging_pumpwl)    
    spin_pumpwl.sigValueChanged.connect(valueChanged_pumpwl)    
    
    
    # r    
    spin_r = pg.SpinBox(value=0.98, suffix='',step=0.005, minStep=0.001)        
    spin_r.setMinimum(0.0)    
    spin_r.setMaximum(1.0)    
    def valueChanging_r(sb, value):
        src.setr(value) 
        src.update() 
        src.updatePlots() 

    spin_r.sigValueChanging.connect(valueChanging_r)    
    spin_r.sigValueChanged.connect(valueChanged_default)  
    
    # tau
    spin_tau = pg.SpinBox(value=0.98, suffix='',step=0.005, minStep=0.001)        
    spin_tau.setMinimum(0.0)    
    spin_tau.setMaximum(1.0)    
    def valueChanging_tau(sb, value):
        src.setr(value) 
        src.update() 
        src.updatePlots() 

    spin_tau.sigValueChanging.connect(valueChanging_tau)    
    spin_tau.sigValueChanged.connect(valueChanged_default)  
    
    # Length
    spin_length = pg.SpinBox(value=70.0, suffix='',step=0.01, minStep=0.001)        
    spin_length.setMinimum(1.0)    
        
    def valueChanging_L(sb, value):
        src.setLength(value) 
        src.update() 
        src.updatePlots() 

    spin_length.sigValueChanging.connect(valueChanging_L)    
    spin_length.sigValueChanged.connect(valueChanged_default)  
        
    ##
    
    spins = [ 
    ("Lambda min", spin_lbda_min),
    ("Lambda max", spin_lbda_max),       
    ("Pump pulse duration", spin_pumpwidth),    
    ("Pump wavelength", spin_pumpwl),    
    ("r", spin_r),    
    ("tau", spin_tau),    
    ("L", spin_length),    
    ]    
    
    spinlayout=QtGui.QGridLayout() 
    spins_widget=QtGui.QWidget()  
    spins_widget.setLayout(spinlayout)
    i = 0
    j = 0
    for text,spin in spins:     
        label = QtGui.QLabel(text)
        spinlayout.addWidget(label,i,0+2*j)
        spinlayout.addWidget(spin,i,1+2*j)
        i += 1
        if i == 4:
            i = 0
            j +=1
    
    def applyButtonCallBack():
        src.update() 
        src.updateAfterChanged()
        src.updatePlots()
        src.updateResults()
    
    
    apply_button = QtGui.QPushButton("Compute JS")
    apply_button.pressed.connect(applyButtonCallBack)
    spinlayout.addWidget(apply_button,i,1+2*j)
    
    def saveButtonCallBack():
        
        src.save()
    
    apply_button = QtGui.QPushButton("Save")
    apply_button.pressed.connect(saveButtonCallBack)
    spinlayout.addWidget(apply_button,i,2+2*j)
    
    #
    # Additional filter        
    ######
                
    """
    filter = DoubleBusRing(r1,r2,tau,L)
    spins2 = [ 
    ("r1", spin_f_s_r1),
    ("r2", spin_f_s_r2),       
    ("tau", spin_f_s_tau),    
    ("L", spin_f_s_L)
    ]
    """  
    
    
    layout.addWidget(spins_widget,2,0)        
    ##
    
    layout.addWidget(src.results_widget,2,1)        
    
    
   
    import sys
    pg.setConfigOptions(useWeave=False)
    print "Hello"
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        print "Running"
        QtGui.QApplication.instance().exec_()
        print "Done"
if __name__ == '__main__':
    main()

    

