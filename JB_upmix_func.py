import numpy as np
import erb as erb
import librosa
import scipy
import nussl

class JBupmix:
    """Implements the Jeroen Breebaart's Phantom materialization methods.
    References:
        * Breebaart J, Schuijers E. Phantom materialization:
          A novel method to enhance stereo audio reproduction on headphones.
          IEEE transactions on audio, speech, and language processing,
          2008, 16(8): 1503-1511.
    """

    _window_length = 512
    _window_type = 'hann'
    _num_bands = 34
    _sample_rate = 48000

    def __init__(self, input_audio_signal):
        self.background = None
        self.foreground = None
        self._input_signal = input_audio_signal
        self._hop_length = self._window_length // 2
        self._n_fft_bins = self._window_length * 2
        self._num_bands_p2 = self._num_bands + 2
        freqs_length = self._n_fft_bins // 2 + 1

        self._sin_window = np.sqrt(scipy.signal.get_window(self._window_type, self._window_length))

        # Equivalent Rectangular Bandwidth
        # Create an instance of the ERB filter bank class
        erb_bank = erb.EquivalentRectangularBandwidth(freqs_length, self._sample_rate, self._num_bands, 50, 22000)
        erb_filters = erb_bank.filters
        
        self._filters2 = np.power(erb_filters, 2)

    def run(self):
        # load file
        sig, sr = librosa.load(self._input_signal, sr=self._sample_rate, mono=False)
        SL = librosa.stft(sig[0,:], n_fft=self._n_fft_bins, hop_length=self._hop_length, win_length=self._window_length, window=self._sin_window)
        SR = librosa.stft(sig[1,:], n_fft=self._n_fft_bins, hop_length=self._hop_length, win_length=self._window_length, window=self._sin_window)

        M = SL.shape[1]
        B2 = self._num_bands_p2
        freqs_length = SL.shape[0]
        SD_BS = np.zeros((SL.shape[0], B2, M, 2), dtype=np.complex_)
        SD_syn = np.zeros((freqs_length, M, 2), dtype=np.complex_)

        EPS = 0.0001
        g_default = np.arctan(1.0)
        filters2 = self._filters2

        # spatial analysis
        for m in range(M):
            L = SL[:, m]
            R = SR[:, m]
            EL = np.real(L * np.conjugate(L))
            ER = np.real(R * np.conjugate(R))
            LR = L * np.conjugate(R)
            for b in range(B2):
                a_BS_L = np.sqrt(np.abs(np.dot(filters2[:, b], EL)))
                a_BS_R = np.sqrt(np.abs(np.dot(filters2[:, b], ER)))
                g_BS = g_default
                if (a_BS_L * a_BS_R) > EPS:
                    rho_BS = np.real( np.dot(filters2[:, b], LR) / (a_BS_L * a_BS_R) )
                    if  np.abs(rho_BS) < 1.0:
                        v_BS = 0.5 * np.arccos( rho_BS )
                        b_BS = np.tan(((a_BS_R - a_BS_L) / (a_BS_R + a_BS_L)) * np.arctan(v_BS))
                        g_BS = np.arctan((a_BS_L * np.cos(v_BS + b_BS)) / (a_BS_R * np.cos(-v_BS + b_BS)))

                SD_BS[:, b, m, 0] = (L + R) / (np.sin(g_BS) + np.cos(g_BS))
                SD_BS[:, b, m, 1] = L - np.sin(g_BS) * SD_BS[:, b, m, 0]

                SD_BS[:, b, m, 0] = SD_BS[:, b, m, 0] * filters2[:, b]
                SD_BS[:, b, m, 1] = SD_BS[:, b, m, 1] * filters2[:, b]

                SD_syn[:, m, 0] = SD_syn[:, m, 0] + SD_BS[:, b, m, 0]
                SD_syn[:, m, 1] = SD_syn[:, m, 1] + SD_BS[:, b, m, 1]

        self._fg_out = librosa.istft(SD_syn[:, :, 0], hop_length=self._hop_length, win_length=self._window_length, window=self._sin_window)
        self._bg_out = librosa.istft(SD_syn[:, :, 1], hop_length=self._hop_length, win_length=self._window_length, window=self._sin_window)

    def make_audio_signals(self):
        self.foreground = nussl.AudioSignal(audio_data_array=self._fg_out, sample_rate=self._sample_rate)
        self.background = nussl.AudioSignal(audio_data_array=self._bg_out, sample_rate=self._sample_rate)
        return [self.background, self.foreground]



