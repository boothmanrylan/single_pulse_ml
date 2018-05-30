import random
import numpy as np
from baseband import vdif
import astropy.units as u
from scipy import signal
import matplotlib.pyplot as plt

k_dm = 4.148808e3

class Event(object):
    """ Class to generate a realistic fast radio burst and
    add the event to data, including scintillation, temporal
    scattering, spectral index variation, and DM smearing.

    This class was expanded from real-time FRB injection
    in Kiyoshi Masui's
    https://github.com/kiyo-masui/burst\_search
    """
    def __init__(self, t_ref, f_ref, dm=(0.,2000.), fluence=(0.03,0.3),
                 width=(2*0.0016, 1.), spec_ind=(-4.,4), disp_ind=2.,
                 scat_factor=(0, 0.5), freq=(800., 400.)):
        self.t_ref = t_ref
        self.f_ref = f_ref
        self.width = width
        self.freq_low = freq[0]
        self.freq_up = freq[1]

        if hasattr(dm, '__iter__') and len(dm) == 2:
            _dm = tuple(dm)
        else:
            _dm = (float(dm), float(dm))
        self.dm = random.uniform(*_dm)

        if hasattr(fluence, '__iter__') and len(fluence) == 2:
            fluence = (fluence[1]**-1, fluence[0]**-1)
            _fluence = tuple(fluence)
        else:
            _fluence = (float(fluence)**-1, float(fluence)**-1)
        self.fluence = random.uniform(*_fluence)**(-2/3.)
        self.fluence *= 1e3*_fluence[0]**(-2/3.)

        if hasattr(width, '__iter__') and len(width) == 2:
            _width = tuple(width)
        else:
             _width = (float(width), float(width))
        self.width = np.random.lognormal(np.log(_width[0]), _width[1])
        self.width = max(min(self.width, 100*_width[0]), 0.5*_width[0])

        if hasattr(spec_ind, '__iter__') and len(spec_ind) == 2:
            _spec_ind = tuple(spec_ind)
        else:
            _spec_ind = (float(spec_ind), float(spec_ind))
        self.spec_ind = random.uniform(*_spec_ind)

        if hasattr(disp_ind, '__iter__') and len(disp_ind) == 2:
            _disp_ind = tuple(disp_ind)
        else:
            _disp_ind = (float(disp_ind), float(disp_ind))
        self.disp_ind = random.uniform(*_disp_ind)

        if hasattr(scat_factor, '__iter__') and len(scat_factor) == 2:
            _scat_factor = tuple(scat_factor)
        else:
            _scat_factor = (float(scat_factor), float(scat_factor))
        self.scat_factor = np.exp(np.random.uniform(*_scat_factor))
        self.scat_factor = min(1, self.scat_factor + 1e-18) # quick bug fix hack

        self.freq = np.linspace(self.freq_low, self.freq_up, 256) # tel parameter 

    def disp_delay(self, f, _dm, _disp_ind=-2.):
        """ Calculate dispersion delay in seconds for
        frequency,f, in MHz, _dm in pc cm**-3, and
        a dispersion index, _disp_ind.
        """
        return k_dm * _dm * (f**(-_disp_ind))

    def arrival_time(self, f):
        t = self.disp_delay(f, self.dm, self.disp_ind)
        t = t - self.disp_delay(self.f_ref, self.dm, self.disp_ind)
        return self.t_ref + t

    def calc_width(self, dm, freq_c, bw=400.0, NFREQ=1024,
                   ti=0.001, tsamp=0.001, tau=0):
        """ Calculated effective width of pulse
        including DM smearing, sample time, etc.
        Input/output times are in seconds.
        """

        ti *= 1e3
        tsamp *= 1e3
        delta_freq = bw/NFREQ

        # taudm in milliseconds
        tdm = (2*k_dm)*(10**-6) * dm * delta_freq / freq_c**3
        tI = np.sqrt(ti**2 + tsamp**2 + tdm**2 + tau**2)

        return 1e-3*tI

    def scintillation(self, freq):
        """ Include spectral scintillation across
        the band. Approximate effect as a sinusoid,
        with a random phase and a random decorrelation
        bandwidth.
        """
        # Make location of peaks / troughs random
        scint_phi = np.random.rand()
        f = np.linspace(0, 1, len(freq))

        # Make number of scintils between 0 and 10 (ish)
        nscint = np.exp(np.random.uniform(np.log(1e-3), np.log(7)))
        #nscint=5
        envelope = np.cos(2*np.pi*nscint*f + scint_phi)
        envelope[envelope<0] = 0

        return envelope

    def gaussian_profile(self, nt, width, t0=0.):
        """ Use a normalized Gaussian window for the pulse,
        rather than a boxcar.
        """
        t = np.linspace(-nt//2, nt//2, nt)
        g = np.exp(-(t-t0)**2 / width**2)

        if not np.all(g > 0):
            g += 1e-18

        g /= g.max()

        return g

    def scat_profile(self, nt, f, tau=1.):
        """ Include exponential scattering profile.
        """
        tau_nu = tau * (f / self.f_ref)**-4.
        t = np.linspace(0., nt//2, nt)

        prof = 1 / tau_nu * np.exp(-t / tau_nu)
        return prof / prof.max()

    def pulse_profile(self, nt, width, f, tau=100., t0=0.):
        """ Convolve the gaussian and scattering profiles
        for final pulse shape at each frequency channel.
        """
        gaus_prof = self.gaussian_profile(nt, width, t0=t0)
        scat_prof = self.scat_profile(nt, f, tau)
        pulse_prof = signal.fftconvolve(gaus_prof, scat_prof)[:nt]

        return pulse_prof

    def add_to_data(self, delta_t, freq, data, bw, scintillate=True):
        """ Method to add already-dedispersed pulse
        to background noise data. Includes frequency-dependent
        width (smearing, scattering, etc.) and amplitude
        (scintillation, spectral index).
        """
        data = np.copy(data)

        NFREQ = data.shape[0]
        NTIME = data.shape[1]
        tmid = NTIME//2

        scint_amp = self.scintillation(freq)
        self.fluence /= np.sqrt(NFREQ)
        stds = np.std(data)

        width_ = self.calc_width(self.dm, self.f_ref*1e-3,
                                        bw=bw, NFREQ=NFREQ,
                                        ti=self.width, tsamp=delta_t, tau=0)
        index_width = max(1, (np.round((width_/ delta_t))).astype(int))

        for ii, f in enumerate(freq):
            tpix = int(self.arrival_time(f) / delta_t)

            if abs(tpix) >= tmid:
                # ensure that edges of data are not crossed
                continue

            pp = self.pulse_profile(NTIME, index_width, f,
                                    tau=self.scat_factor, t0=tpix)
            pp /= (pp.max()*stds)
            pp *= self.fluence
            pp /= (width_ / delta_t)
            pp = pp * (f / self.f_ref) ** self.spec_ind

            if scintillate is True:
                pp = (0.1 + scint_amp[ii]) * pp

            data[ii] +=pp
        return data


def gen_simulated_frb(NFREQ=16, NTIME=250, sim=True, fluence=(0.03,0.3),
                spec_ind=(-4, 4), width=(2*0.0016, 1), dm=(-0.01, 0.01),
                scat_factor=(-3, -0.5), background_noise=None, delta_t=0.0016,
                freq=(800, 400), FREQ_REF=600., scintillate=True,
                ):
    """ Simulate fast radio bursts using the EventSimulator class.

    Parameters
    ----------
    NFREQ       : np.int
        number of frequencies for simulated array
    NTIME       : np.int
        number of times for simulated array
    sim         : bool
        whether or not to simulate FRB or just create noise array
    spec_ind    : tuple
        range of spectral index
    width       : tuple
        range of widths in seconds (atm assumed dt=0.0016)
    scat_factor : tuple
        range of scattering measure (atm arbitrary units)
    background_noise :
        if None, simulates white noise. Otherwise should be an array (NFREQ, NTIME)

    Returns
    -------
    data : np.array
        data array (NFREQ, NTIME)
    parameters : tuple
        [dm, fluence, width, spec_ind, disp_ind, scat_factor]

    """
    # Hard code incoherent Pathfinder data time resolution
    # Maybe instead this should take a telescope class, which 
    # has all of these things already.
    t_ref = 0. # hack

    if len(freq) < 3:
        freq=np.linspace(freq[0], freq[1], NFREQ)

    if background_noise is None:
        # Generate background noise with unit variance
        data = np.random.normal(0, 1, NTIME*NFREQ).reshape(NFREQ, NTIME)
    else:
        data = background_noise

    # What about reading in noisy background?
    if sim is False:
        return data, []

    # Create event class with those parameters 
    #TODO multiple fluence by 10e-4
    E = Event(t_ref, FREQ_REF, dm=dm, fluence=fluence,
              width=width, spec_ind=spec_ind)
    # Add FRB to data array 
    new_data = E.add_to_data(delta_t, freq, data, bw=max(freq)-min(freq), scintillate=scintillate)

    return new_data, [E.dm, E.fluence, E.width, E.spec_ind, E.disp_ind, E.scat_factor]


def inject_into_vdif(vdif_in, vdif_out, NFREQ=1024, NTIME=2**15,
                     rate=800*u.MHz, fluence=(10**4, 10**4), spec_ind=(2, 2),
                     dm=(10**4, 10**5), scat_factor=(-4, -0.5), freq=(800, 400),
                     FREQ_REF=600.):
    """
    Generates a simulated FRB and injects it (at a random time) into the data
    contained in vdif_in, stores the result in vdif_out.

    Args:
        vdif_in (str):  Path to vdif file to inject the FRB into. vdif_in will
                        not be altered by this method.
        vdif_out (str): Path to a vdif file to store the result of this method
                        in.
        rate (atropy units 1/time): # of samples per second in vdif_in

        ***For all other arguments see gen_simulated_frb***

    Returns: None
    """
    delta_t = NTIME / 156250
    # get the data and the header from vdif_in
    with vdif.open(vdif_in, 'rs', sample_rate=rate) as fh:
        data = fh.read()
        header0 = fh.header0

    # Get the power from data
    data = (np.abs(data)**2).mean(1)
    data = data - np.nanmean(data, axis=1, keepdims=True)
    data = data.T

    background = data[:NFREQ, :NTIME]

    # use the data from vdif_in as background noise in gen_simulated_frb
    new_data, params = gen_simulated_frb(NFREQ=NFREQ, NTIME=NTIME, sim=True,
                                         fluence=fluence, spec_ind=spec_ind,
                                         width=(delta_t*2, 1), dm=dm,
                                         scat_factor=scat_factor,
                                         background_noise=background,
                                         delta_t=delta_t, freq=freq,
                                         FREQ_REF=FREQ_REF)
    plt.figure(figsize=(45,70))
    plt.subplot(121)
    plt.imshow(data, vmin=-1.0, vmax=1.0, interpolation="nearest",
               aspect="auto")
    data[:NFREQ, :NTIME] = new_data
    plt.subplot(122)
    plt.imshow(data, vmin=-1.0, vmax=1.0, interpolation="nearest",
               aspect="auto")

    plt.show()
    # write the event returned by gen_simulated_frb to vdif_out
    # with vdif.open(vdif_out, 'ws', header0=header0, sample_rate=rate) as fh:
    #    fh.write(data.T)


