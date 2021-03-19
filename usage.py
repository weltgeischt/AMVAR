import amvar
import numpy as np
import matplotlib.pyplot as plt

########################################################
#1 .create surrogate data
sr = 500 #sampling rate
ntrials = 100 #number of trials
simdur = 1 #simulation duration in seconds

datamat = amvar.generate_surrogate(ntrials=100, simdur=1, sr=500)
# plot one example trial
time = np.arange(0., simdur, 1 / sr)
f,ax = plt.subplots()
ax.plot(time,datamat[0,5,:],'k')
ax.plot(time,datamat[1,5,:],'grey')
ax.legend(['ch1','ch2'])
ax.set_ylabel('Amplitude')
ax.set_xlabel('Times [s]')
plt.show()

########################################################
#2. to decide on window width and order, look at the AIC of an exemplary window --> try to minimize the AIC
ww_test = [10,20] #window width you want to test
orders_test = np.arange(1,8) #orders you want to test
aic_dict = amvar.get_AICdict(datamat,ww_test,orders_test, winstart=0, method='lsq')#calculates aic in one window starting as indicated by pts winstart
f,ax = plt.subplots()
ax.plot(orders_test,aic_dict['aic'][10],'b')
ax.plot(orders_test,aic_dict['aic'][20],'g')
ax.legend(['ww10','ww20'])
ax.set_ylabel('AIC')
ax.set_xlabel('order')
plt.show()


########################################################
#3. run the amvar algorithm
ww = 20
order = 5
upperfreq =100 # maximal frequency desired
fresol = 0.1 #frequency resolution
jump_step = np.int(ww/2.)
[ar,ea],sm = amvar.run_ar(datamat, order=order, ww=ww, algo='lsq', sr=sr,upperfreq =upperfreq,fresol=0.1, jump_step=jump_step)
#--> ar: autoregressive coefficient, ea: noise cov., sm: spectral matrix


########################################################
#4. plot spectral quantities like power and coherence
#look at spectral flavors available
print (amvar.spectral_flavor_fns.keys())
#select eg power
flavor = 'power'
chans = [0,0] # for e.g. coherence it would be chans = [0,1]
spgrm = amvar.get_spectrogram(sm, chs=chans, flavour=flavor) #eg coherence between channels 0 and 1

# plot the spectrogram
plotmat = np.log10(spgrm.T) if flavor=='power' else spgrm.T
cmap = 'hsv' if flavor.count('phase') else 'magma' #hsv for cyclic phase representation
freqVec = np.arange(0., upperfreq + fresol, fresol)
nwins,nfreqs = spgrm.shape
x = np.linspace(0, time.max(), nwins)
y = np.r_[freqVec[1:], freqVec.max() + np.diff(freqVec)[0]]
X, Y = np.meshgrid(x, y)
f,ax = plt.subplots()
ax.set_title(flavor)
spim = ax.pcolormesh(X, Y, plotmat , cmap=cmap)
ax.set_ylabel('Freq [Hz]')
ax.set_xlabel('Times [s]')
cbar = plt.colorbar(spim, orientation='vertical')
plt.show()


########################################################
#5.plot residual distribution to check whether they appear to be normally distributed
residmat = amvar.get_residmat(0,datamat,ar,ww,jump_step,order)
f = amvar.plot_residual_mat(residmat, time, nbins=50)