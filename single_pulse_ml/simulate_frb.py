import random
import numpy as np
from baseband import vdif
import astropy.units as u
from scipy import signal
import matplotlib.pyplot as plt

k_dm = 4.148808e3

class Event(object):
    """
    Class to generate a realistic fast radio burst and add the event to data,
    including scintillation, temporal scattering, spectral index variation, and
    DM smearing .This class was expanded from real-time FRB injection in
    Kiyoshi Masui's https://github.com/kiyo-masui/burst\_search
    """
    def __init__(self, t_ref, f_ref, NFREQ=16, NTIME=250, delta_t=0.0016,
                 background_noise=None, dm=(0.,2000.), fluence=(0.03,0.3),
                 width=(2*0.0016, 1.), spec_ind=(-4.,4), disp_ind=2.,
                 scat_factor=(0, 0.5), freq=(800., 400.), scintillate=True,
                 rate=800*u.MHz):
        self.t_ref = t_ref
        self.f_ref = f_ref
        self.freq_low = freq[0]
        self.freq_up = freq[1]
        self.scintillate = scintillate
        self.bandwidth = max(freq) - min(freq)

        if rate is None:
            self.delta_t = delta_t
        else:
            self.delta_t = (1/rate).value

        self.make_background(background_noise, NFREQ, NTIME, rate)

        self.stds = np.std(self.background_noise)

        try:
            self.dm = random.uniform(*dm)
        except:
            self.dm = dm

        try:
            # There is a weird effect when the bracets around random are
            # removed changing the order of operations
            self.fluence = 1e3*(random.uniform(*fluence)**4/3)
        except:
            self.fluence = 1e3*(fluence**4/3)

        try:
            self.width = np.random.lognormal(np.log(width[0]), width[1])
            width = width[0]
        except:
             self.width = np.random.lognormal(np.log(width), width)
        self.width = max(min(self.width, 100*width), 0.5*width)

        self.time_width = self.calc_width(tau=0)
        self.index_width = max(1, (np.round((self.time_width/ self.delta_t))).astype(int))

        try:
            self.spec_ind = random.uniform(*spec_ind)
        except:
            self.spec_ind = spec_ind

        try:
            self.disp_ind = random.uniform(*disp_ind)
        except:
            self.disp_ind = disp_ind

        try:
            self.scat_factor = np.exp(random.uniform(*scat_factor))
        except:
            self.scat_factor = np.exp(scat_factor)
        self.scat_factor = min(1, self.scat_factor + 1e-18) # quick bug fix hack

        self.freq = np.linspace(self.freq_low, self.freq_up, self.NFREQ) # tel parameter 
        self.simulated_frb = self.simulate_frb()

    def make_background(self, background, NFREQ, NTIME, rate):
        try:
            self.background_noise = background_noise
            self.NFREQ = background_noise.shape[0]
            self.NTIME = background_noise.shape[1]
            self.header0 = None
        except:
            try:
                with vdif.open(background, 'rs', sample_rate=rate) as fh:
                    data = fh.read()
                    self.header0 = fh.header0
                # Get the power from data
                data = (np.abs(data)**2).mean(1)
                data = data - np.nanmean(data, axis=1, keepdims=True)
                self.background_noise = data.T
                self.NFREQ = self.background_noise.shape[0]
                self.NTIME = self.background_noise.shape[1]
            except:
                self.background_noise = np.random.normal(0, 1, size=(NFREQ, NTIME))
                self.NFREQ = NFREQ
                self.NTIME = NTIME
                self.header0 = None

    def disp_delay(self, f):
        """
        Calculate dispersion delay in seconds for frequency,f, in MHz, _dm in
        pc cm**-3, and a dispersion index, _disp_ind.
        """
        return k_dm * self.dm * (f**(-self.disp_ind))

    def arrival_time(self, f):
        t = self.disp_delay(f)
        t = t - self.disp_delay(self.f_ref)
        return self.t_ref + t

    def calc_width(self, tau=0):
        """
        Calculated effective width of pulse including DM smearing, sample time,
        etc.  Input/output times are in seconds.
        """

        ti = self.width * 1e3
        tsamp = self.delta_t * 1e3
        delta_freq = self.bandwidth/self.NFREQ

        # taudm in milliseconds
        tdm = (2*k_dm)*(10**-6) * self.dm * delta_freq / (self.f_ref*1e-3)**3
        tI = np.sqrt(ti**2 + tsamp**2 + tdm**2 + tau**2)

        return 1e-3*tI

    def scintillation(self, freq):
        """
        Include spectral scintillation across the band. Approximate effect as
        a sinusoid, with a random phase and a random decorrelation bandwidth.
        """
        # Make location of peaks / troughs random
        scint_phi = np.random.rand()
        f = np.linspace(0, 1, len(freq))

        # Make number of scintils between 0 and 10 (ish)
        nscint = np.exp(np.random.uniform(np.log(1e-3), np.log(7)))
        #nscint=5
        envelope = np.cos(2*np.pi*nscint*f + scint_phi)
        envelope[envelope<0] = 0
        envelope += 0.1

        return envelope

    def gaussian_profile(self, width, t0=0.):
        """
        Use a normalized Gaussian window for the pulse, rather than a boxcar.
        """
        t = np.linspace(-self.NTIME//2, self.NTIME//2, self.NTIME)
        g = np.exp(-(t-t0)**2 / width**2)

        if not np.all(g > 0):
            g += 1e-18

        g /= g.max()

        return g

    def scat_profile(self, f, tau=1.):
        """
        Include exponential scattering profile.
        """
        tau_nu = tau * (f / self.f_ref)**-4.
        t = np.linspace(0., self.NTIME//2, self.NTIME)

        prof = 1 / tau_nu * np.exp(-t / tau_nu)
        return prof / prof.max()

    def pulse_profile(self, f, t0=0.):
        """
        Convolve the gaussian and scattering profiles for final pulse shape at
        frequency channel f.
        """
        gaus_prof = self.gaussian_profile(self.index_width, t0=t0)
        scat_prof = self.scat_profile(f, self.scat_factor)
        pp = signal.fftconvolve(gaus_prof, scat_prof)[:self.NTIME]
        pp /= (pp.max()*self.stds)
        pp *= self.fluence
        pp /= (self.time_width / self.delta_t)
        pp = pp * (f / self.f_ref) ** self.spec_ind

        return pp

    def simulate_frb(self):
        """
        Method to add already-dedispersed pulse to background noise data.
        Includes frequency-dependent width (smearing, scattering, etc.) and
        amplitude (scintillation, spectral index).
        """
        data = np.copy(self.background_noise)
        tmid = self.NTIME//2

        if self.scintillate:
            scint_amp = self.scintillation(self.freq)
        self.fluence /= np.sqrt(self.NFREQ)

        for ii, f in enumerate(self.freq):
            tpix = int(self.arrival_time(f) / self.delta_t)

            if abs(tpix) >= tmid:
                # ensure that edges of data are not crossed
                continue

            pp = self.pulse_profile(f, t0=tpix)

            if self.scintillate is True:
                pp *= scint_amp[ii]

            data[ii] += pp
        return data

    def plot(self, save=None):
        plt.figure(figsize=(45,70))
        plt.subplot(121)
        plt.imshow(self.background_noise, vmin=-1.0, vmax=1.0, interpolation="nearest",
                   aspect="auto")
        plt.subplot(122)
        plt.imshow(self.simulated_frb, vmin=-1.0, vmax=1.0, interpolation="nearest",
                   aspect="auto")

        if save is None:
            plt.show()
        else:
            plt.savefig(save)


def inject_into_vdif(vdif_in, vdif_out, NFREQ=1024, NTIME=2**15,
                     rate=800*u.MHz, fluence=(10**4, 10**4), spec_ind=(2, 2),
                     dm=(10**2, 10**3), scat_factor=(-4, -0.5), freq=(800, 400),
                     FREQ_REF=600., delta_t=0.0016):
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
    # get the data and the header from vdif_in

    # use the data from vdif_in as background noise in gen_simulated_frb
    event = Event(t_ref=0, fluence=fluence, rate=rate,
                  spec_ind=spec_ind, width=(delta_t*2, 1), dm=dm,
                  scat_factor=scat_factor, background_noise=vdif_in,
                  delta_t=delta_t, freq=freq, f_ref=FREQ_REF)
    event.plot()

