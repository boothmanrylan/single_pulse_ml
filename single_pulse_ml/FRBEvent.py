import random
import glob
import numpy as np
from baseband import vdif
from baseband.helpers import sequentialfile as sf
import astropy.units as u
from scipy import signal
import matplotlib.pyplot as plt

k_dm = 4.148808e-1 * (u.GHz**2 * u.cm**3 * u.ms / u.pc)

class FRBEvent(object):
    """
    Class to generate a realistic fast radio burst and add the event to data,
    including scintillation, temporal scattering, spectral index variation, and
    DM smearing .This class was expanded from real-time FRB injection in
    Kiyoshi Masui's https://github.com/kiyo-masui/burst\_search
    """
    def __init__(self, t_ref=0*u.ms, f_ref=None, NFREQ=1024, NTIME=2**15,
                 delta_t=0.0016*u.ms, dm=(2e1, 2e2)*(u.pc/u.cm**3),
                 fluence=(1e6, 1e7)*(u.Jy*u.ms), freq=(0.8, 0.4)*u.GHz,
                 rate=(0.4/1024)*u.GHz, scat_factor=(-4), width=None,
                 scintillate=True, spec_ind=(-4.), background=None,):
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
        scintillate :   If true scintills are added to the data.
        spec_ind :      The spectral index.
        background :    The background the pulse will be injected into.
        """
        self.t_ref = t_ref.to(u.ms)
        self.scintillate = scintillate
        self.bandwidth = (max(freq) - min(freq)).to(u.GHz)
        self.rate=rate.to(u.GHz)

        if f_ref is None:
            f_ref = (max(freq.value)*0.9, min(freq.value)*1.1) * freq.unit
            f_ref = random.uniform(*f_ref)
        self.f_ref = f_ref.to(u.GHz)

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
            self.fluence = random.uniform(*fluence)**(4/3) * fluence_units
        except:
            self.fluence = fluence**(4/3) * fluence_units
        self.fluence /= np.sqrt(self.NFREQ)

        width =  width or (10, 10) * self.delta_t
        units = width.unit
        value = width.value
        try:
            logvalue = np.random.lognormal(np.log(value[0]), value[1])
            value = value[0]
        except:
             logvalue = np.random.lognormal(np.log(value), value)
        width = max(min(logvalue, 100*value), 0.5*value) * units
        self.width = self.calc_width(width)

        try:
            self.spec_ind = random.uniform(*spec_ind)
        except:
            self.spec_ind = spec_ind

        try:
            self.scat_factor = np.exp(random.uniform(*scat_factor))
        except:
            self.scat_factor = np.exp(scat_factor)
        self.scat_factor = min(1, self.scat_factor + 1e-18) # quick bug fix hack

        self.freq = np.linspace(freq[0], freq[1], self.NFREQ).to(u.GHz)
        self.simulated_frb = self.simulate_frb()
#         self.frb_dm = self.dm_transform(self.simulated_frb)
#         self.bg_dm = self.dm_transform(self.background)

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
        return k_dm * self.dm * (f**-2)

    def arrival_time(self, f):
        t = self.disp_delay(f)
        t = t - self.disp_delay(self.f_ref)
        return self.t_ref + t

    def calc_width(self, width):
        """
        Calculated effective width of pulse including DM smearing, sample time,
        etc.  Input/output times are in seconds.
        """

        delta_freq = self.bandwidth/self.NFREQ

        # taudm in milliseconds
        tau_dm = 2*k_dm * self.dm * delta_freq / (self.f_ref)**3
        tI = np.sqrt(width**2 + self.delta_t**2 + tau_dm**2)
        t_index = max(1, int(tI / self.delta_t))

        return t_index

    def scintillation(self, freq):
        """
        Include spectral scintillation across the band. Apulse_profroximate effect as
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

    def gaussian_profile(self, t):
        """
        Use a normalized Gaussian window for the pulse, rather than a boxcar.
        """
        tlin = np.linspace(-self.NTIME//2, self.NTIME//2, self.NTIME)
        g = np.exp(-(tlin-t)**2 / self.width**2)

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

    def pulse_profile(self, f, t):
        """
        Convolve the gaussian and scattering profiles for final pulse shape at
        frequency channel f.
        """
        gaus_prof = self.gaussian_profile(t)
        scat_prof = self.scat_profile(f)
        pulse_prof = signal.fftconvolve(gaus_prof, scat_prof)[:self.NTIME]
        pulse_prof /= (pulse_prof.max()*self.stds)
        pulse_prof *= self.fluence.value
        pulse_prof /= (self.width / self.delta_t.value)

        pulse_prof = pulse_prof * (f / self.f_ref).value ** self.spec_ind

        return pulse_prof

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

        for ii, f in enumerate(self.freq):
            t = int(self.arrival_time(f) / self.delta_t)

            if abs(t) >= tmid:
                # ensure that edges of data are not crossed
                continue

            p = self.pulse_profile(f, t)

            if self.scintillate is True:
                p *= scint_amp[ii]

            data[ii] += p
        return data

    def dm_transform(self, data, NDM=50):
        dm = np.linspace(-self.dm, self.dm, NDM)
        dm_data = np.zeros([NDM, self.NTIME])

        for ii, dm in enumerate(dm):
            for jj, f in enumerate(self.freq):
                t = int(self.arrival_time(f) / self.delta_t)
                data_rot = np.roll(data[jj], t, axis=-1)
                dm_data[ii] += data_rot

        return data

    def plot(self, save=None):
        f, axis = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(18, 30))
        axis[0].imshow(self.background, vmin=-1.0, vmax=1.0, interpolation="nearest",
                       aspect="auto")
        axis[0].set_title("Background Noise")
        axis[0].set_xlabel("Time ({})".format(self.delta_t.unit))
        axis[0].set_ylabel("Frequency ({})".format(self.freq.unit))
        yticks = np.linspace(0, self.NFREQ, 10)
        ylabels = max(self.freq)-(self.bandwidth/self.NFREQ*yticks)
        ylabels = np.around(ylabels.value, 2)
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
    f = np.sort(glob.glob('/home/rylan/dunlap/data/natasha_vdif/0000008.vdif'))
    vdif_in = sf.open(f)
    event = FRBEvent(background=vdif_in)
    event.plot()

