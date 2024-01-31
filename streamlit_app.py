import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy import stats

#############################################################################
Nobs = 20
alpha_true = 0.5
beta_x_true = 1.0
beta_y_true = 10.0
B_true = 0.5
#############################################################################
data_13 = np.loadtxt('glg_tte_n7_bn201221963_16ms_ascii.dat') 
rate_13 = np.sum(data_13[:, 26:510:4], axis=1)
time_13 = data_13[:, 0] 
data13 = np.loadtxt('01_glg_tte_n7_bn201221963_16ms_ascii.dat')
time13 = data13[:, 0] 
time13_more=np.linspace(time13[0],time13[-1],500)    #CHANGE       #####################         #######################
rate13 = np.sum(data13[:, 26:510:4], axis=1)
rate13_error = np.sqrt(data13[:,27]**2+data13[:,31]**2+data13[:,35]**2+data13[:,39]**2+data13[:,43]**2+data13[:,47]**2+data13[:,51]**2+data13[:,55]**2+data13[:,59]**2+data13[:,63]**2+data13[:,67]**2+data13[:,71]**2+data13[:,75]**2+data13[:,79]**2+data13[:,83]**2+data13[:,87]**2+data13[:,91]**2+data13[:,95]**2+data13[:,99]**2+data13[:,103]**2+data13[:,107]**2+data13[:,111]**2+data13[:,115]**2+data13[:,119]**2+data13[:,123]**2+data13[:,127]**2+data13[:,131]**2+data13[:,135]**2+data13[:,139]**2+data13[:,143]**2+data13[:,147]**2+data13[:,151]**2+data13[:,155]**2+data13[:,159]**2+data13[:,163]**2+data13[:,167]**2+data13[:,171]**2+data13[:,175]**2+data13[:,179]**2+data13[:,183]**2+data13[:,187]**2+data13[:,191]**2+data13[:,195]**2+data13[:,199]**2+data13[:,203]**2+data13[:,207]**2+data13[:,211]**2+data13[:,215]**2+data13[:,219]**2+data13[:,223]**2+data13[:,227]**2+data13[:,231]**2+data13[:,235]**2+data13[:,239]**2+data13[:,243]**2+data13[:,247]**2+data13[:,251]**2+data13[:,255]**2+data13[:,259]**2+data13[:,263]**2+data13[:,267]**2+data13[:,271]**2+data13[:,275]**2+data13[:,279]**2+data13[:,283]**2+data13[:,287]**2+data13[:,291]**2+data13[:,295]**2+data13[:,299]**2+data13[:,303]**2+data13[:,307]**2+data13[:,311]**2+data13[:,315]**2+data13[:,319]**2+data13[:,323]**2+data13[:,327]**2+data13[:,331]**2+data13[:,335]**2+data13[:,339]**2+data13[:,343]**2+data13[:,347]**2+data13[:,351]**2+data13[:,355]**2+data13[:,359]**2+data13[:,363]**2+data13[:,367]**2+data13[:,371]**2+data13[:,375]**2+data13[:,379]**2+data13[:,383]**2+data13[:,387]**2+data13[:,391]**2+data13[:,395]**2+data13[:,399]**2+data13[:,403]**2+data13[:,407]**2+data13[:,411]**2+data13[:,415]**2+data13[:,419]**2+data13[:,423]**2+data13[:,427]**2+data13[:,431]**2+data13[:,435]**2+data13[:,439]**2+data13[:,443]**2+data13[:,447]**2+data13[:,451]**2+data13[:,455]**2+data13[:,459]**2+data13[:,463]**2+data13[:,467]**2+data13[:,471]**2+data13[:,475]**2+data13[:,479]**2+data13[:,483]**2+data13[:,487]**2+data13[:,491]**2+data13[:,495]**2+data13[:,499]**2+data13[:,503]**2+data13[:,507]**2+data13[:,511]**2)

#############################################################################
def Norris_vec(time, t1,t2):
    A = max(rate13)
    tmax = time13[rate13.argmax()]
    nu1 = 1.6
    nu2 = 1
    model1 = A * np.exp(-(abs(time-tmax)/t1)**nu1)
    model2 = A * np.exp(-(abs(time-tmax)/t2)**nu2)
    if (time < tmax):
       return model1 
    else: 
       return model2
Norris=np.vectorize(Norris_vec)
#############################################################################
popt13, pcov13 = opt.curve_fit(Norris, xdata=time13, ydata=rate13, p0=(2., 0.1))#, maxfev = 1000000,sigma=rate_error13, absolute_sigma=True)
perr13 = np.sqrt(np.diag(pcov13))
residuals13 = rate13 - Norris(time13,*popt13)
chi_squared13 = np.sum( (residuals13)**2/(rate13_error)**2)
reduced_chi_squared13 = chi_squared13/(len(rate13)-2)
#############################################################################
Rise_time_1 = popt13[0]
Fall_time_1 = popt13[1]
z = 1.046
print('A = ', max(rate13))
print('Peak_time = ', time13[rate13.argmax()])
print('Rise time1 = ', Rise_time_1/(1+z), "+/-", pcov13[0,0]**0.5/(1+z))
print('Fall time1 = ', Fall_time_1/(1+z), "+/-", pcov13[1,1]**0.5/(1+z))
print('chi_squared13 = ', chi_squared13)
print('dof = ', (len(rate13)-2))
print('reduced_chi_squared13 = ', reduced_chi_squared13)
#############################################################################
with open("Norris_parameters_n7.txt", 'a') as sys.stdout:
    print('####################################################################')    
    print('channel all')
    print('A = ', max(rate13))
    print('Peak_time = ', time13[rate13.argmax()])
    print('Rise time1 = ', Rise_time_1/(1+z), "+/-", pcov13[0,0]**0.5/(1+z))
    print('Fall time1 = ', Fall_time_1/(1+z), "+/-", pcov13[1,1]**0.5/(1+z))
    print('chi_squared13 = ', chi_squared13)
    print('dof = ', (len(rate13)-2))
    print('reduced_chi_squared13 = ', reduced_chi_squared13)
#############################################################################
fig = plt.figure(figsize=(6, 6))
plt.subplots_adjust(hspace = 0.25)
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax1.axes.get_xaxis().set_visible(True) ## FALSE THUS DOESNT SHOW X-VALUES, TRUE SHOWS X-VALUES
ax1.set_xlim(-1,1)   
#ax2.set_xlim(-1,4)
ax1.axvline(-0.112, color = 'm',linestyle='--',linewidth = 1)
ax1.axvline(0.208, color = 'm',linestyle='--',linewidth = 1)
#ax2.axvline(np.sqrt(t1*t2), color = 'm',linestyle='--',linewidth = 1)
ax1.title.set_text('GRB201221D (10 keV - 1 MeV)')
#ax1.title.set_text('GRB201221D in the energy range 10 keV - 1 MeV')
ax1.set_xlabel("Seconds after trigger time")
ax1.set_ylabel("Rate")
ax2.set_xlabel("Seconds after trigger time")
ax2.set_ylabel("Rate")
y_more = Norris(time13_more, *popt13)            #CHANGE       #####################         #######################
ax1.plot(time_13, rate_13, drawstyle = 'steps', color = 'b')
ax1.legend(loc = "upper left", title = 'NaI n7')
ax2.plot(time13, rate13, drawstyle = 'steps', color = 'b')
ax2.plot(time13_more-0.008, y_more, color = 'm') #CHANGE       #####################         #######################
plt.savefig('GRB201221D n7 channel all NaI_2.png')
plt.savefig('GRB201221D n7 channel all NaI.pdf')
plt.show()

'''

NU1 = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0]



for nu1 in NU1:
    def Norris_vec(time, t1,t2):
       A = max(rate13)
       tmax = time13[rate13.argmax()]
    #nu1 = 2
       nu2 = 2
       model1 = A * np.exp(-(abs(time-tmax)/t1)**nu1)
       model2 = A * np.exp(-(abs(time-tmax)/t2)**nu2)
       #print('NU = ', nu1,nu2)
       if (time < tmax):
          return model1 
       else: 
          return model2
       return nu1
    Norris=np.vectorize(Norris_vec)
#############################################################################
    popt13, pcov13 = opt.curve_fit(Norris, xdata=time13, ydata=rate13, p0=(2., 0.1))#, maxfev = 1000000,sigma=rate_error13, absolute_sigma=True)
    perr13 = np.sqrt(np.diag(pcov13))
    residuals13 = rate13 - Norris(time13,*popt13)
    chi_squared13 = np.sum( (residuals13)**2/(rate13_error)**2)
    reduced_chi_squared13 = chi_squared13/(len(rate13)-2)
#############################################################################
    print('NU = ', nu1)
    Rise_time_1 = popt13[0]
    Fall_time_1 = popt13[1]
    z = 1.046
    print('Rise time1 = ', Rise_time_1/(1+z), "+/-", pcov13[0,0]**0.5/(1+z))
    print('Fall time1 = ', Fall_time_1/(1+z), "+/-", pcov13[1,1]**0.5/(1+z))
    print('chi_squared13 = ', chi_squared13)
    print('dof = ', (len(rate13)-2))
    print('reduced_chi_squared13 = ', reduced_chi_squared13)

#############################################################################
    fig = plt.figure(figsize=(6, 6))
    plt.subplots_adjust(hspace = 0.25)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.axes.get_xaxis().set_visible(True) ## FALSE THUS DOESNT SHOW X-VALUES, TRUE SHOWS X-VALUES
    ax1.set_xlim(-2,2)   
#ax2.set_xlim(-1,4)
    ax1.axvline(-0.400, color = 'm',linestyle='--',linewidth = 1)
    ax1.axvline(0.992, color = 'm',linestyle='--',linewidth = 1)
#ax2.axvline(np.sqrt(t1*t2), color = 'm',linestyle='--',linewidth = 1)
    ax1.title.set_text('GRB131004A channel 2 (25 - 50 keV)')
    ax1.set_xlabel("Seconds after trigger time")
    ax1.set_ylabel("Rate (counts/s)")
    ax2.set_xlabel("Seconds after trigger time")
    ax2.set_ylabel("Rate (counts/s)")
    y_more = Norris(time13_more, *popt13)            #CHANGE       #####################         #######################
    ax1.plot(time_13, rate_13, drawstyle = 'steps', color = 'b')
    ax1.legend(loc = "upper left", title = 'NaI na')
    ax2.plot(time13, rate13, drawstyle = 'steps', color = 'b')
    ax2.plot(time13_more-0.001, y_more, color = 'm') #CHANGE       #####################         #######################
    #plt.savefig('GRB131004A na channel 2')
    #plt.show()

'''
