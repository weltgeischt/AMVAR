import numpy as np
from scipy import optimize
from time import ctime
from functools import reduce


##########CALCULATING AR COEFFICIENTS, NOISE COVARIANCE AND SPECTRAL MATRICES

def run_ar(datachunks, order=5, ww=20, algo='lsq', sr=500, upperfreq=100, fresol=0.1,**kwargs):
	'''datachunks is in (nchans,ntrials,dpoints)'''

	jump_step = kwargs['step'] if 'step' in kwargs else int(ww / 2.)
	freqVec = np.arange(0., upperfreq + fresol, fresol)

	n_chans = datachunks.shape[0]

	print	('###############################################')
	print	('INPUT DATA    {0} channels, {1} trials, {2} points'.format(*[x for x in datachunks.shape]))

    # preprocessing: zscoring
	data_z = z_score(datachunks)

	# setting up sliding window mechanism
	dpoints = datachunks.shape[2]
	winstarts = np.arange(0, dpoints - ww + 1, jump_step)
	nwins = len(winstarts)

	# empty matrices for the output
	allARs = np.zeros((nwins, order + 1, n_chans, n_chans), dtype='complex')
	allEAs = np.zeros((nwins, n_chans, n_chans), dtype='complex')
	allSMs = np.zeros((nwins, len(freqVec), n_chans, n_chans), dtype='complex')
	print ('AR-MODEL    order{0}    ww{1}    step{4}    sr{2}    chunkshape{3}    wins{5}'. \
		format(order, ww, sr, data_z.shape, jump_step, nwins))

	for ss, winstart in enumerate(winstarts):
		if np.mod(ss, 50) == 0:
			print ('window number', ss, ctime())

		snipInput = data_z[:, :, winstart:winstart + ww].transpose((1, 0, 2))[:]

		ARs, EA = getArEa(snipInput, order, method=algo)
		allARs[ss], allEAs[ss] = ARs, EA
		allSMs[ss] = calcSpMat(ARs, EA, freqVec, sr=sr)
	# print (ARs.shape,EA.shape,spvals.shape)
	# allSMs[ss] = spvals

	return [allARs, allEAs], allSMs

def z_score(allchunks):
	meanchunks, stdchunks = np.mean(allchunks, 1), np.std(allchunks, 1)
	data_z = (allchunks - meanchunks[:, None, :]) / stdchunks[:, None, :]
	return data_z


def getArEa(data, order, method='lsq'):
	'''data is (trials,channels,points)'''

	if method == 'yw':
		print ('obtainable upon request')

	elif method == 'lsq':

		ARs, resids, estimated = calcAR_lsq(data, order)
		EA, __ = getNoiseCovarianceMatrix(resids)

		zeroAR = np.eye(data.shape[-2])[None, :, :]
		ARs = np.vstack([zeroAR, ARs])

	return ARs, EA

def calcSpMat(ARs, EA, freqs, sr=500):
	spList = []
	for freq in freqs:
		temp = np.sum([ARs[jj] * np.e ** (2j * np.pi * freq / sr * jj) for jj in range(ARs.shape[0])], axis=0)
		tf = np.linalg.inv(temp)
		sf = reduce(np.dot, [tf, EA, tf.conj().T])
		spList.append(sf)
	return np.array(spList)


def calcAR_lsq(dataWin, order):
	'''leastsquare based MVAR procedure
	input: datawindow to be analised (trials x channels x window Width), order of AR process
	output: ar coefficients (order x chans x chans), residuals (trials, ww-order, channels)
			fit of leastsquare method (), such that residuals+ fit[:,order:]=dataWin[:,:,order:]
	'''

	trials = len(dataWin)
	chans = len(dataWin[0])
	ww = len(dataWin[0, 0])
	resids = ww - order

	arMat = np.zeros((order, chans, chans))
	residMat = np.zeros((trials, resids, chans))
	estimatedMat = np.zeros((trials, chans, ww))
	# print order,chans,trials,ww
	guessMat = np.ones((order, chans))  # input only what regards value of one channel
	try:
		guessMat[0, 1:] = 0.5
		guessMat[1:, :] = 0.25
	except:
		pass

	guess = guessMat.flatten()
	for chan in range(chans):
		# the idea for a multivariate process is to iterate over all possible channel-considered, channels-contributing combinations
		# print shape(delete(dataWin,chan,1)),shape(guess)

		coeffs, success = optimize.leastsq(mvarErr, guess.copy(),
										   args=(dataWin[:, chan], np.delete(dataWin, chan, 1)))  # ,maxfev=10000
		# print success
		# the Coeff of chans influence on the chan is the first one, so it has to be moved to its proper place...

		if type(coeffs) == float: coeffs = np.array(([coeffs]))

		if chan == 0:
			coeffsSorted = coeffs.reshape(order, chans)
		else:
			a = np.ones((len(coeffs)))
			a[::chans] = 0
			nakedCoeffs0 = np.take(coeffs, np.where(a)[0])
			nakedCoeffs = nakedCoeffs0.reshape(order, chans - 1)
			coeffsSorted = np.insert(nakedCoeffs, chan, coeffs[::chans], axis=1)

		arMat[:, chan, :] = coeffsSorted
		fit = -mvarFit(coeffs, dataWin[:, chan], np.delete(dataWin, chan, 1))
		estimatedMat[:, chan, :] = fit
		noise = dataWin[:, chan, order:] - fit[:,
										   order:]  # first order points lack nb of predecessors the model requires
		residMat[:, :, chan] = noise

	return arMat, residMat, estimatedMat


mvarErr = lambda p00,data1,dataArray:(mvarFit(p00,data1,dataArray)+data1).flatten()

def mvarFit(p00,data1,dataArray):
    '''fitfunction for calcAR_lsq
    input: parameters(arCoeffs with respect to data to be predicted, flattened),
            data of channel whose value is to be predicted with parameters,
            other channels of the process that contribute '''
    trials=len(dataArray)

    chans=len(dataArray[0])+1
    p00=np.array(p00)

    try: order= np.int(len(p00)/chans)
    except: order=1

    ww=len(data1[0])
    startingPoints=data1[:,:order]

    data=np.zeros((trials,chans,ww))
    data[:,0]=data1
    data[:,1:]=dataArray
    #print ('hey')
    p=p00.reshape(order,chans)

    d1 = np.c_[startingPoints,np.zeros((trials,ww-order))]

    #print(shape(d1))

    for trial in np.arange(trials):
        for ii in np.arange(ww-order):
            temp=0
            for chan in np.arange(chans):
                for ord in np.arange(order):
                    temp+=p[ord,chan]*data[trial,chan,ii+order-ord-1]

            d1[trial,ii+order]=temp
    return d1


def getNoiseCovarianceMatrix(residWin):
	'''meanCov should effectively be the same as Dings SIGMA, although differently calculated
	input:residWin(trialsxresidsxchans)
	output:mean covariance matrix(chan x chan) and array of noise Covariance matrices for
	individual trials (trials x chan x chan)'''

	trials = len(residWin)
	chans = len(residWin[0, 0])

	noiseCovList = ([])
	for trial in range(trials):
		trialNoise = residWin[trial, :, :]
		noiseCov = np.cov(trialNoise.T)  # welll actually not quite..., do it the Ding Way
		noiseCovList.append(noiseCov)
	meanCov = np.mean(noiseCovList, 0)

	covTrials = np.array(noiseCovList)

	if chans == 1:
		covTrials = covTrials.reshape(trials, 1, 1)
		meanCov = meanCov.reshape(1, 1)

	return meanCov, covTrials


############## DERIVING SPECTRAL QUANTITIES
def Minor(myMat, indexA, indexB):
	mat0 = np.delete(myMat, indexA, axis=0)
	mat1 = np.delete(mat0, indexB, axis=1)
	return np.linalg.det(mat1)


def powF(sf, chs=[0, 0]):
	chA, chB = chs
	return np.abs(sf[chA, chA]) ** 2

def cohF(sf, chs=[0, 1]):
	chA, chB = chs
	return np.abs(sf[chA, chB] / np.sqrt(sf[chA, chA] * sf[chB, chB]))

def phaseCohF(sf, chs=[0, 1]):
	chA, chB = chs
	return -np.arctan(np.imag(sf[chA, chB]) / np.real(sf[chA, chB]))

def multCohF(sf, chs=[0, 0]):
	chA, chB = chs
	top = np.linalg.det(sf)
	bottom = sf[chA, chA] * Minor(sf, chA, chA)
	return np.abs(np.sqrt(1 - top / bottom))

def partCohF(sf, chs=[0, 1], abs=True):
	chA, chB = chs
	top = Minor(sf, chA, chB)
	bottom = np.sqrt(Minor(sf, chA, chA) * Minor(sf, chB, chB))
	if abs == True:
		return np.abs(top / bottom)
	else:
		return top / bottom

def phasePartCohF(sf, chs=[0, 1]):
	chA, chB = chs
	K_ab = partCohF(sf, chs=[chA, chB], abs=False)
	return np.arctan(np.imag(K_ab) / np.real(K_ab))

spectral_flavor_fns = {'coherence': cohF, 'coherence_phase': phaseCohF,
					   'mult_coherence': multCohF, 'part_coherence': partCohF,
					   'part_coherence_phase': phasePartCohF, 'power': powF}


def get_spectrogram(SMs, chs=[0, 0], flavour='power'):
	nwins, nfreqs, nchans, nchans = SMs.shape
	return np.vstack([get_spectrum(SMs[mywin], chs=chs, flavour=flavour) \
					  for mywin in np.arange(nwins)])

def get_spectrum(sm, chs=[0, 0], flavour='power'):
	nfreqs = sm.shape[0]
	if flavour in ['power', 'mult_coherence']:
		assert chs[0] == chs[1], "quantity %s requires identical channels!"%flavour
	else:
		assert chs[0] != chs[1], "quantity %s requires different channels!"%flavour
	if flavour == 'power':
		return np.abs(sm[:, chs[0], chs[0]]) ** 2

	else:
		myfn = spectral_flavor_fns[flavour]
		return np.array([myfn(sm[ff], chs) for ff in np.arange(nfreqs)])


#Akaike information criterion
def get_AICdict(data, wws_test, orders_test, winstart=0, method='lsq'):

	AIC_dict = {}
	# test yulewalker and lsq for one window
	for ww in wws_test:
		testwin = data[:, :, winstart:ww + winstart].transpose((1, 0, 2))[:]
		aiclist = []
		for order in orders_test:

			#print(testwin.shape, order)

			if order >= ww:
				aicval = None
			# print ww,order,'no val'
			else:
				# print ww,order,time.ctime()

				ARs, EA = getArEa(testwin, order, method=method)
				aicval = getAIC(order, EA, ww, testwin.shape[0])
			# print aicval
			aiclist.append(aicval)
		AIC_dict[ww] = np.array(aiclist)
	return {'aic': AIC_dict, 'orders': orders_test}

def getAIC(order, EA, ww, ntrials):
	nchans = EA.shape[0]
	aicval = 2 * np.log(np.abs(np.linalg.det(EA))) + 2 * (nchans ** 2) * (order + 1) / float(
		((ww - order - 1) * ntrials))
	return aicval




###get and plot residuals

def get_residmat(chan,datamat,ar,ww,jump_step,order):
	nchans, ntrials, npts = datamat.shape
	winstarts = np.arange(0, npts - ww + 1, jump_step)
	nwins = len(winstarts)
	residmat = np.zeros((nwins, ntrials, ww - order))
	for ii, winstart in enumerate(winstarts):
		coeffs = ar[ii, 1:, chan, chan]
		for tt in np.arange(ntrials):
			datawin = z_data(datamat)[chan, tt, winstart:winstart + ww]
			#print (datawin.shape)
			residmat[ii, tt, :] = get_residual_win(datawin, coeffs, order)

	return residmat


def z_data(datamat):
	#datachunks = single_to_multi(datamat)
	return z_score(datamat)

def single_to_multi(data):
	return data.reshape((1,) + data.shape)

def get_residual_win(datawin,coeffs,order):
    '''
    coeffs is the ar coefficents for that window excluding the first (shape: orderxchanxchan)
    '''
    pts_pred = len(datawin)-order
    expected = datawin[order:]
    predicted = -np.array([np.dot(datawin[jj:order+jj],coeffs[::-1]) for jj in np.arange(pts_pred)])
    return np.real(expected-predicted)

def plot_residual_mat(residmat, tvec, nbins=50, cmap='gray', ylabel='Residual', xlabel='Time [s]', meancol='pink',
					  edgeperc=99):
	from scipy.stats import scoreatpercentile
	from matplotlib.pyplot import subplots
	nwins, ntrials, npts = residmat.shape
	gaussedge = scoreatpercentile(residmat, edgeperc)
	resibins = np.linspace(-gaussedge, gaussedge, nbins)
	bw = np.diff(resibins)[0]

	histmat = np.zeros((nwins, nbins - 1))
	for ii in np.arange(nwins):
		histmat[ii], _ = np.histogram(residmat[ii], resibins)

	meanresi = np.mean(residmat.reshape(nwins, -1), axis=1)
	stdresi = np.std(residmat.reshape(nwins, -1), axis=1)

	binvec = resibins[1:] - 0.5 * bw

	f, ax = subplots(figsize=(6, 3), facecolor='w')
	f.subplots_adjust(bottom=0.17, left=0.16, right=0.97, top=0.99)
	x = np.linspace(tvec.min(), tvec.max(), nwins)
	y = binvec[:]
	X, Y = np.meshgrid(x, y)
	# print (X.shape,Y.shape)
	# print (powspect.shape)
	rimg = ax.pcolormesh(X, Y, histmat.T, cmap=cmap)
	ax.plot(x, meanresi, meancol)
	ax.plot(x, meanresi - stdresi, meancol, ls='--')
	ax.plot(x, meanresi + stdresi, meancol, ls='--')
	ax.set_ylim([binvec.min(), binvec.max()])
	ax.set_xlim([tvec.min(), tvec.max()])
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	return f


###############
#surrogate_data
sinewave = lambda time, freq, offset, amp: np.sin(2.*np.pi*freq*time + np.deg2rad(offset)) * amp

def generate_surrogate(ntrials=100, simdur=1, sr=500):
	'''two channels (A and B), one contained a phase-lagged version of the oscillations (freq1 and2) of the other
	output: nchannels x ntrials x timepoints'''


	freq1 = 6
	freq2 = 40
	amp1A = 2
	amp2A = 1
	amp1B = 1.5
	amp2B = 0.8
	off1 = 30
	off2 = 90
	noiseA = 0.1
	noiseB = 0.02

	time = np.arange(0., simdur, 1 / sr)

	datamat = np.zeros((2,ntrials,len(time)))
	for tt in np.arange(ntrials):
		off_gen =  np.random.rand(1)[0]*360
		off_gen2 =  np.random.rand(1)[0]*360
		datamat[0,tt,:] = sinewave(time,freq1,off_gen,amp1A) + np.random.normal(0,noiseA,len(time)) + sinewave(time,freq2,off_gen2,amp2A) #chA
		datamat[1,tt,:] = sinewave(time,freq1,off_gen+off1,amp1B) + np.random.normal(0,noiseB,len(time)) + sinewave(time,freq2,off_gen2+off2,amp2B)#chB
	return datamat

