# -------------------------------------------------------------------------------------------------
# FeatureExtractor.py 
# -------------------------------------------------------------------------------------------------
# Code for preprocessing and spike counting neural data 
# Written by Sean Yoon [sean777@stanford.edu]. 
# -------------------------------------------------------------------------------------------------
# Created     : 2024-06-01
# Last update : 2024-06-01
# -------------------------------------------------------------------------------------------------
from Globals import *
# -------------------------------------------------------------------------------------------------

def butterworth_filter(dat: np.ndarray, filt_type: str, Wn: Union[Tuple, float, int], fs: int, ord: int=4):
    """
    Filters neural data for input frequency band. Uses a non-causal digital butterworth IIR filter 
    and its second-order sections representation (better numerical stability) for computation. 
    
    Args: 
        dat       : [samples x channels] shape array of neural data
        filt_type : type of filter. One of {'bandpass', 'lowpass', 'highpass'}
        Wn        : critical frequency or frequencies. Wn is a length-2 sequence for bandpass and bandstop. 
        fs        : sampling frequency of neural data. 
        ord       : order of the butterworth filter (defaults to 4)
    Returns: 
        filtered  : [samples x channels] shape array of filtered neural data
    """ 
    # quick sanity check 
    if filt_type.lower not in ['bandpass', 'lowpass', 'highpass', 'bandstop']: 
        raise ValueError("Unsupported filtering method")
    
    # initialize filter and apply to data
    sos = signal.butter(N=ord, Wn=Wn, btype=filt_type, output='sos', fs=fs)
    filtered = signal.sosfiltfilt(sos, dat, axis=0)
    return filtered

    
def chebyshev_filter(dat: np.ndarray, filt_type: str,  Wn: Union[Tuple, float, int], fs: int, 
                     ord: int=8, rp: float=0.05): 
    """
    Filters neural data for input frequency band. Uses a Chebyshev type I 
    Preferred when steeper rolloffs are prioritized over greater passband ripples. 
    and its second-order sections representation (better numerical stability) for computation. 

    Args: 
        dat       : [samples x channels] shape array of neural data
        filt_type : type of filter. One of {'bandpass', 'lowpass', 'highpass'}
        Wn        : critical frequency or frequencies. Wn is a length-2 sequence for bandpass and bandstop. 
        fs        : sampling frequency of neural data. 
        ord       : order of the chebyshev filter (defaults to 8)
        rp        : maximum ripple allowed below unity gain, in dB (defaults to 0.05)
    Returns:
        filtered  : [samples x channels] shape array of filtered data
    """ 
    # quick sanity check 
    if filt_type.lower not in ['bandpass', 'lowpass', 'highpass', 'bandstop']: 
        raise ValueError("Unsupported filtering method")
    
    # initialize filter and apply to data
    sos = signal.cheby1(N=ord, Wn=Wn, rp=rp, btype=filt_type, output='sos', fs=fs)
    filtered = signal.sosfiltfilt(sos, dat, axis=0)
    return filtered


def downsample(dat: np.ndarray, decimate_fw: int, fs: int): 
    """
    Downsamples data by given factor of decimate_fw. Assumes that data is already filtered below 
    the Nyquist frequency. 
    
    Args: 
        dat         : [samples x channels] shape array of neural data
        decimate_fw : downsampling factor
        fs          : original sampling frequency 
    """
    return dat[::decimate_fw], int(fs/decimate_fw)

@nb.jit(nopython=True)
def get_car_weights(dat: np.ndarray): 
    """Gets CAR weights for single array recording"""
    dat = np.ascontiguousarray(dat.astype('float32'))
    n_samples = dat.shape[0]
    weights   =  np.zeros(shape=(n_samples, 1), dtype='float32')
    
    # iterate over each timestamp and calculate reference (i.e. channelwise average)
    for t in range(n_samples): 
        weights[t] = np.mean(dat[t,:])
    return weights 
        

def car(dat: np.ndarray, n_arrays: int, n_electrodes: int): 
    """
    Applies common average referencing (CAR) to [samples x channels] shape array of neural data
    
    Args: 
        dat          : [samples x channels] shape array of neural data
        n_arrays     : number of arrays
        n_electrodes : number of electrodes per array
    """
    dat      = np.ascontiguousarray(dat.astype('float32'))
    filtered = np.zeros(shape=dat.shape, dtype='float32')
    
    # iterate over each array and denoise using CAR 
    for arr in range(n_arrays): 
        # extract current array 
        start   = n_electrodes*arr
        end     = n_electrodes*(arr + 1)
        dat_arr = dat[:,start:end]
        
        # get weights for current array and denoise 
        weights = np.ascontiguousarray(get_car_weights(dat_arr))
        filtered[:,start:end] = dat_arr - weights 
    
    return filtered


@nb.jit(nopython=True) 
def lstsq_pseudoinverse(X: np.ndarray, y: np.ndarray):
    """Optimized helper function for calculating the Moore-Penrose pseudoinverse, 
    given by A+ = (AT A)-1 AT. Used for LRR calculation"""
    W = np.linalg.solve(X.T.dot(X), X.T.dot(y)) 
    return W
    
    
def get_lrr_weights(dat: np.ndarray, fs: int, max_seconds: int=0): 
    """Gets LRR weights for single array recording"""
    n_samples, n_channels = dat.shape
    dat     = np.ascontiguousarray(dat.astype('float32'))
    weights = np.zeros(shape=(n_channels, n_channels), dtype='float32')
    
    # subsample data 
    if max_seconds == 0:  # use entire block if max_seconds not specified
        use_idx = n_samples
    else: 
        sample_len = max_seconds*fs
        max_idx    = min(sample_len, n_samples)
        rand_idx   = np.random.permutation(np.arange(n_samples))
        use_idx    = rand_idx[0:max_idx]
        
    lrr_dat = dat[use_idx]
    
    # iterate over each channel and calculate weights
    for ch in range(n_channels): 
        # get a list of all channel indices excluding the current one 
        X = np.zeros(shape=(max_idx, n_channels - 1), dtype='float32')
        X[:,ch:] = lrr_dat[:,ch+1:]
        X[:,:ch] = lrr_dat[:,:ch]
        X = np.ascontiguousarray(X)
        y = np.ascontiguousarray(lrr_dat[:,ch])
        
        # solve the optimized least squares to get weights for this channel
        weights_ch = lstsq_pseudoinverse(X, y)
        
        # Add the weights to the larger weight matrix of all channels in appropriate positions, 
        # leaving space for the current channel where the weight is zero
        weights[ch,:ch]   = weights_ch[:ch] 
        weights[ch,ch+1:] = weights_ch[ch:] 
    
    return weights


def lrr_denoise(dat: np.ndarray, weights: np.ndarray): 
    """Denoises single array using LRR"""
    dat_array = np.ascontiguousarray(dat)
    weights   = np.ascontiguousarray(weights)
    return dat_array - np.dot(dat_array, weights)
    
    
def lrr(dat: np.ndarray, n_arrays: int, n_electrodes: int, fs: int, max_seconds: int, weights_dat=None): 
    """
    Applies common average referencing (CAR) to [samples x channels] shape array of neural data
    
    Args: 
        dat          : [samples x channels] shape array of neural data 
        n_arrays     : number of arrays 
        n_electrodes : number of electrodes per array 
        max_seconds  : length of data in seconds to use for lrr weight calculation. 
        weights_dat  : optional [samples x channels] shape array of neural data to use for calculating weights
    """
    dat = dat.astype('float32')
    filtered = np.zeros(shape=dat.shape, dtype='float32') 
    
    for arr in range(n_arrays): 
        start   = n_electrodes*arr 
        end     = n_electrodes*(arr + 1)
        if weights_dat is None: 
            weights = get_lrr_weights(dat[:,start:end], fs, max_seconds)
        else: 
            weights = get_lrr_weights(weights_dat[:,start:end], fs, max_seconds)
        filtered[:,start:end] = lrr_denoise(dat[:,start:end], weights)

    return filtered 


@nb.jit(parallel=True)
def get_thresholds(dat: np.ndarray, multiplier: float=-4.5): 
    """
    dat        : [samples x channels] shape array of neural data
    multiplier : threshold multiplier. defaults to -4.5
    """
    dat = dat.astype('float32')
    
    n_channels = dat.shape[1]
    thresholds = np.zeros(n_channels, dtype='float32')
    
    # iterate over each channel and calculate rms 
    for ch in range(n_channels): 
        thresholds[ch] = np.sqrt(np.mean(np.square(dat[:,ch])))
    
    return thresholds*multiplier
    
    
@nb.jit(nopython=True)
def count_spikes(dat: np.ndarray, fs: int, bin_size: int, shift_size: int, multiplier: float=-4.5,
                 from_above: bool=True): 
    """
    Optimized function to count number of instances where neural data crosses threshold. Threshold 
    is calculated using channelwise RMS voltage. 

    Args:
        dat        : [samples x channels] shape array of neural data
        fs         : sampling frequency (in Hz)
        bin_size   : bin size in ms
        shift_size : bin shift size in ms 
        multiplier : threshold multiplier. defaults to -4.5
        from_above : whether to count crossings from above threshold to below, or vice versa
        
    Returns: 
        crossings  : [bins x channels] shape array of threshold crossings 
    """
    # convert bin size (ms) to datapoint 
    bin_t   = int(bin_size * fs / 1000)
    shift_t = int(shift_size * fs / 1000) 

    # convert variables to jit-compatible format 
    dat = dat.astype('float32')
    n_samples, n_channels = dat.shape
    n_bins = int(np.ceil(n_samples / shift_t))
    crossings = np.zeros(shape=(n_bins, n_channels), dtype='float32')

    # compare with threshold and use int8 to save energy 
    threshold    = get_thresholds(dat, multiplier).astype('float32')
    below_thresh = (dat <= threshold).astype('int8')
    
    # get datapoints where signal crosses the threshold - faster than np.diff
    if from_above: 
        cross_bool = (below_thresh[1:] - below_thresh[:-1]) == 1  # X[i+1] - X[i]
    else: 
        cross_bool = (below_thresh[:-1] - below_thresh[1:]) == 1  # X[i] - X[i+1]
    
    # iterate over each bin and count number of crossings 
    for bin_idx, t in enumerate(range(0, n_samples, shift_t)): 
        cross_bin = cross_bool[t:t+bin_t]
        for ch in range(n_channels): 
            crossings[bin_idx, ch] = np.sum(cross_bin[:, ch])
    
    return crossings 
    
    
