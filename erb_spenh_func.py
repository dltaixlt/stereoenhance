import numpy as np
import erb as erb
import librosa
import scipy
import nussl

class ERBEnhance:
    """Implements the speech enhancement methods.
    References:
        * Geiger J T, Grosche P, Parodi Y L. Dialogue enhancement
          of stereo sound[C]//2015 23rd European Signal Processing
          Conference (EUSIPCO). IEEE, 2015: 869-873.
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
        self._erb_filters = erb_bank.filters

        self._filters2 = np.power(self._erb_filters, 2)
        #filters2_tr = np.transpose(self._filters2)
        #filters2_pseudo = np.matmul(np.linalg.inv(np.matmul(filters2_tr, self._filters2)), filters2_tr)
        #self._filters2_pseudo_tr = np.transpose(filters2_pseudo)

        self._speechEnh = False

    def set_speech_enhance(self, enhance=True):
        self._speechEnh = enhance

    def run(self):
        # load file
        sig, sr = librosa.load(self._input_signal, sr=self._sample_rate, mono=False)
        SL = librosa.stft(sig[0,:], n_fft=self._n_fft_bins, hop_length=self._hop_length, win_length=self._window_length, window=self._sin_window)
        SR = librosa.stft(sig[1,:], n_fft=self._n_fft_bins, hop_length=self._hop_length, win_length=self._window_length, window=self._sin_window)

        # center channel extract (based on ERB and smooth)
        sm = 0.8
        B2 = self._num_bands_p2
        M = SL.shape[1]
        CLR = np.zeros((SL.shape[0], SL.shape[1], 3), dtype=np.complex_)
        a = np.zeros(B2)
        filters = self._filters2
        EPS = 0.00001
        for m in range(M):
            for b in range(B2):
                L = SL[:, m] * filters[:, b]
                R = SR[:, m] * filters[:, b]
                D = L - R
                S = L + R
                if False:
                    AS = np.real(np.vdot(S, S))
                    if AS < EPS:
                        alpha = 0.0
                    else:
                        alpha = 0.5 * (1 - np.sqrt( np.real(np.vdot(D, D)) / AS ) )

                    if m > 0:
                        a[b] = sm * a[b] + (1 - sm) * alpha
                    else:
                        a[b] = alpha
                else:
                    a[b] = 0.5

                Ce = a[b] * S
                CLR[:, m, 0] = CLR[:, m, 0] + Ce
                CLR[:, m, 1] = CLR[:, m, 1] + L - Ce
                CLR[:, m, 2] = CLR[:, m, 2] + R - Ce

        if self._speechEnh:
            # speech enhance by Wiener filter (based on ERB)
            CLR_enh = np.zeros(CLR.shape, dtype=np.complex_)
            G = np.zeros(B2)
            #filters = self._filters2_pseudo_tr
            for m in range(M):
                for b in range(B2):
                    CeB = CLR[:, m, 0] * filters[:, b]
                    DB = (CLR[:, m, 1] - CLR[:, m, 2]) * filters[:, b]
                    e_CeB = np.real(np.vdot(CeB, CeB))
                    e_DB = np.real(np.vdot(DB, DB))
                    S = e_CeB + e_DB
                    if S < EPS:
                        G[b] = 1.0
                    else:
                        if m > 0:
                            G[b] = sm * G[b] + (1 - sm) * e_CeB / (e_CeB + e_DB)
                        else:
                            G[b] = e_CeB / (e_CeB + e_DB)

                    if np.abs(G[b]) > 1.0:
                        G[b] = 1.0

                    # enhance dialogue in foreground
                    CLR_enh[:, m, 0] = CLR_enh[:, m, 0] + CLR[:, m, 0] * filters[:, b] * G[b]
                    # suppress dialogue in background
                    CLR_enh[:, m, 1] = CLR_enh[:, m, 1] + CLR[:, m, 1] * filters[:, b] * (1 - G[b])
                    CLR_enh[:, m, 2] = CLR_enh[:, m, 2] + CLR[:, m, 2] * filters[:, b] * (1 - G[b])

            CLR_ori = np.copy(CLR)
            # add background dialogue back to foreground
            CLR[:, :, 0] = CLR_enh[:, :, 0] + CLR_ori[:, :, 1] - CLR_enh[:, :, 1] + CLR_ori[:, :, 2] - CLR_enh[:, :, 2]
            # add foreground ambience back to background
            CLR[:, :, 1] = CLR_enh[:, :, 1] + (CLR_ori[:, :, 0] - CLR_enh[:, :, 0]) * 0.5
            CLR[:, :, 2] = CLR_enh[:, :, 2] + (CLR_ori[:, :, 0] - CLR_enh[:, :, 0]) * 0.5

        self._fg_out = librosa.istft(CLR[:, :, 0], hop_length=self._hop_length, win_length=self._window_length, window=self._sin_window)
        bg_left = librosa.istft(CLR[:, :, 1], hop_length=self._hop_length, win_length=self._window_length, window=self._sin_window)
        bg_right = librosa.istft(CLR[:, :, 2], hop_length=self._hop_length, win_length=self._window_length, window=self._sin_window)
        self._bg_out = np.array([bg_left, bg_right])


    def make_audio_signals(self):
        self.foreground = nussl.AudioSignal(audio_data_array=self._fg_out, sample_rate=self._sample_rate)
        self.background = nussl.AudioSignal(audio_data_array=self._bg_out, sample_rate=self._sample_rate)
        return [self.background, self.foreground]


if __name__ == "__main__":
    f1 = 'test'
    #f1 = 'tv'
    upmixfunc = 'erbenh8'
    infile = 'src/' + f1 + '.wav '
    bk = 'tmp/' + f1 + '_%s_bk.wav ' % (upmixfunc)
    fg = 'tmp/' + f1 + '_%s_fg.wav ' % (upmixfunc)

    erbenh = ERBEnhance(infile)
    #erbenh.set_speech_enhance()
    erbenh.run()
    background, foreground = erbenh.make_audio_signals()

    foreground.write_audio_to_file(fg)
    background.write_audio_to_file(bk)
