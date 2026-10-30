"""A short synthesized burst, played without blocking camera processing."""
import numpy as np


def burst_samples(rate=22050):
    time = np.arange(int(rate*.65))/rate
    noise = np.random.default_rng(7).normal(0, 1, len(time))
    noise = np.convolve(noise, np.ones(5)/5, mode='same')
    low = np.sin(2*np.pi*(125*time-70*time*time))
    crack = np.sin(2*np.pi*(1700*time-1000*time*time))*np.exp(-time*25)
    envelope = (1-np.exp(-time*180))*np.exp(-time*7)
    samples = (low*.45+noise*.6+crack*.16)*envelope
    samples *= np.minimum(1, (.65-time)/.04)
    return np.asarray(np.clip(samples, -.65, .65), dtype=np.float32)


class Sound:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.samples = burst_samples()
        self.device = None
        self.failed = False

    def play(self):
        if not self.enabled or self.failed:
            return
        try:
            if self.device is None:
                import sounddevice
                self.device = sounddevice
            self.device.play(self.samples, samplerate=22050, blocking=False)
        except Exception as error:
            self.failed = True
            print(f'Sound unavailable; continuing silently: {error}')

    def stop(self):
        if self.device is not None:
            self.device.stop()
