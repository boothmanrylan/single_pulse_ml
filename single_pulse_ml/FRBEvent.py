import random
import glob
import numpy as np
from baseband import vdif
from baseband.helpers import sequentialfile as sf
import astropy.units as u
from scipy import signal
import matplotlib.pyplot as plt

k_dm = 4.148808e3 * (u.MHz**2 * u.cm**3 * u.ms / u.pc)

class FRBEvent(object):
    """
    Class to generate a realistic fast radio burst and add the event to data,
    including scintillation, temporal scattering, spectral index variation, and
    DM smearing .This class was expanded from real-time FRB injection in
    Kiyoshi Masui's https://github.com/kiyo-masui/burst\_search
    """
    def __init__(self, t_ref=0*u.ms, f_ref=600*u.MHz, NFREQ=1024, NTIME=2**15,
                 delta_t=0.0016*u.ms, dm=1000*(u.pc/u.cm**3), disp_ind=2.,
                 fluence=12*(u.Jy*u.ms), freq=(800., 400.)*u.MHz,
                 rate=(400/1024)*u.MHz, scat_factor=(-4, -0.5), width=None,
                 scintillate=True, spec_ind=(-4.,4), background=None,):
        """
        t_ref :         The reference time used when computing the DM.
        f_ref :         The reference frequency used when computing the DM.
        NFREQ :         The number of frequency bins.
        NTIME :         The number of time bins.
        delta_t :       The time step between two time bins.
        dm :            The dispersion measure (DM) of the pulse.
        fluence :       The fluence of the pulse.
        freq :          The range of freqeuncies in the data.
        rate :          The rate at which samples were taken.
        scat_factor :   How much scattering occurs in the pulse.
        width :         The width in time of the pulse.
        disp_ind :      The dispersion index
        scintillate :   If true scintills are added to the data.
        spec_ind :      The spectral index.
        background :    The background the pulse will be injected into.
        """
        self.t_ref = t_ref.to(u.ms)
        self.f_ref = f_ref.to(u.MHz)
        self.scintillate = scintillate
        self.bandwidth = (max(freq) - min(freq)).to(u.MHz)
        self.rate=rate.to(u.MHz)

        if rate is None:
            self.delta_t = delta_t.to(u.ms)
        else:
            self.delta_t = (1/rate).to(u.ms)

        self.make_background(background, NFREQ, NTIME, rate)

        self.stds = np.std(self.background)

        try:
            self.dm = random.uniform(*dm).to(u.pc/u.cm**3)
        except:
            self.dm = dm.to(u.pc/u.cm**3)

        fluence_units = fluence.unit
        fluence = fluence.value
        try:
            # There is a weird effect when the bracets around random are
            # removed changing the order of operations
            self.fluence = 1e3*(random.uniform(*fluence)**4/3)
        except:
            self.fluence = 1e3*(fluence**4/3)
        self.fluence = self.fluence * fluence_units

        width =  width or (3, 3) * self.delta_t
        width_units = width.unit
        width = width.value
        try:
            self.width = np.random.lognormal(np.log(width[0]), width[1])
            width = width[0]
        except:
             self.width = np.random.lognormal(np.log(width.value), width)
        self.width = max(min(self.width, 100*width), 0.5*width) * width_units

        self.time_index = self.calc_width(tau=0)

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

        self.freq = np.linspace(freq[0], freq[1], self.NFREQ).to(u.MHz)
        self.simulated_frb = self.simulate_frb()

    def make_background(self, background, NFREQ, NTIME, rate):
        try:
            self.background = background
            self.NFREQ = background.shape[0]
            self.NTIME = background.shape[1]
            self.header0 = None
        except:
            try:
                with vdif.open(background, 'rs', sample_rate=rate) as fh:
                    data = fh.read()
                    self.header0 = fh.header0
                # Get the power from data
                data = (np.abs(data)**2).mean(1)
                data = data - np.nanmean(data, axis=1, keepdims=True)
                self.background = data.T
                self.NFREQ = self.background.shape[0]
                self.NTIME = self.background.shape[1]
            except:
                self.background = np.random.normal(0, 1, size=(NFREQ, NTIME))
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

        ti = self.width.to(u.ms)
        tsamp = self.delta_t.to(u.ms)
        delta_freq = self.bandwidth/self.NFREQ

        # taudm in milliseconds
        tdm = (2*k_dm)*(10**-6) * self.dm * delta_freq / (self.f_ref*1e-3)**3
        tI = np.sqrt(ti**2 + tsamp**2 + tdm**2 + tau**2)
        t_index = max(1, int(tI / self.delta_t))

        return t_index

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

    def gaussian_profile(self, t0):
        """
        Use a normalized Gaussian window for the pulse, rather than a boxcar.
        """
        t = np.linspace(-self.NTIME//2, self.NTIME//2, self.NTIME)
        g = np.exp(-(t-t0)**2 / self.time_index**2)

        if not np.all(g > 0):
            g += 1e-18

        g /= g.max()

        return g

    def scat_profile(self, f):
        """
        Include exponential scattering profile.
        """
        tau_nu = self.scat_factor * (f / self.f_ref)**-4.
        t = np.linspace(0., self.NTIME//2, self.NTIME)

        prof = 1 / tau_nu * np.exp(-t / tau_nu)
        return prof / prof.max()

    def pulse_profile(self, f, t0):
        """
        Convolve the gaussian and scattering profiles for final pulse shape at
        frequency channel f.
        """
        gaus_prof = self.gaussian_profile(t0)
        scat_prof = self.scat_profile(f)
        pp = signal.fftconvolve(gaus_prof, scat_prof)[:self.NTIME]
        pp /= (pp.max()*self.stds)
        pp *= self.fluence.value
        pp /= (self.time_index)
        pp = pp * (f / self.f_ref).value ** self.spec_ind

        return pp

    def simulate_frb(self):
        """
        Method to add already-dedispersed pulse to background noise data.
        Includes frequency-dependent width (smearing, scattering, etc.) and
        amplitude (scintillation, spectral index).
        """
        data = np.copy(self.background)
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
        f, axis = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(18, 30))
        axis[0].imshow(self.background, vmin=-1.0, vmax=1.0, interpolation="nearest",
                       aspect="auto")
        axis[0].set_title("Background Noise")
        axis[0].set_xlabel("Time ({})".format(self.delta_t.unit))
        axis[0].set_ylabel("Frequency ({})".format(self.freq.unit))
        yticks = np.linspace(0, self.NFREQ, 10)
        ylabels = max(self.freq)-(self.bandwidth/self.NFREQ*yticks).astype(int)
        ylabels = ylabels.value
        xticks = np.linspace(0, self.NTIME, 10)
        xlabels = (self.delta_t * xticks).astype(int)
        xlabels = xlabels.value
        axis[0].set_yticks(yticks)
        axis[0].set_yticklabels(ylabels)
        axis[0].set_xticks(xticks)
        axis[0].set_xticklabels(xlabels)
        axis[1].imshow(self.simulated_frb, vmin=-1.0, vmax=1.0, interpolation="nearest",
                   aspect="auto")
        axis[1].set_title("Simulated FRB")
        axis[1].set_xlabel("Time (ms)")
        axis[1].set_xticks(xticks)
        axis[1].set_xticklabels(xlabels)

        if save is None:
            plt.show()
        else:
            plt.savefig(save)

if __name__ == "__main__":
    f = np.sort(glob.glob('/home/rylan/dunlap/data/natasha_vdif/0000012.vdif'))
    vdif_in = sf.open(f)
    event = Event(background=vdif_in, rate=(400/1024)*u.MHz, width=None)
    event.plot()

