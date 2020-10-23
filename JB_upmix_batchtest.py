from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import erb as erb
import librosa
import scipy
import nussl

# set parameters
window_length = 1024
hop_length = window_length // 2
window_type = 'hann'
n_fft_bins = window_length
B = 34
B2 = B+2
freqs_length = n_fft_bins // 2 + 1
sample_rate = 48000

# get windows
sin_window = np.sqrt(scipy.signal.get_window('hann', n_fft_bins))

# Equivalent Rectangular Bandwidth
# Create an instance of the ERB filter bank class
erb_bank = erb.EquivalentRectangularBandwidth(freqs_length, sample_rate, B, 50, 22000)
filters2 = erb_bank.filters
np.power(filters2, 2)

# preprocess BRIR
#BRIRs = ['BRIR_R01_P1_E0_A0.wav', 'BRIR_R01_P1_E0_A-60.wav', 'BRIR_R01_P1_E0_A60.wav', 'BRIR_R01_P1_E0_A-30.wav', 'BRIR_R01_P1_E0_A30.wav']
#BRIR_gain = 0.6
#BRIRs = ['IIS_BRIR_A+000_E+00.wav', 'IIS_BRIR_A+090_E+00.wav', 'IIS_BRIR_A-090_E+00.wav', 'IIS_BRIR_A+030_E+00.wav', 'IIS_BRIR_A-030_E+00.wav']
BRIRs = ['IIS_BRIR_A+000_E+00.wav', 'IIS_BRIR_A+060_E+00.wav', 'IIS_BRIR_A-060_E+00.wav', 'IIS_BRIR_A+030_E+00.wav', 'IIS_BRIR_A-030_E+00.wav']
BRIRs_path = 'brirs/'
BRIR_gain = 4.45

def brir_para(brirL, brirR):
    K = brirL.shape[1] // 4
    pFC = np.zeros((B2, 3))

    for b in range(B2):
        corr = 0*1j
        for k in range(K):
            L = brirL[:, k] * filters2[:, b]
            R = brirR[:, k] * filters2[:, b]

            pFC[b, 0] = pFC[b, 0] + np.real(np.vdot(L, L))
            pFC[b, 1] = pFC[b, 1] + np.real(np.vdot(R, R))
            corr = corr + np.vdot(L, R)

        pFC[b, 0] = pFC[b, 0] / K
        pFC[b, 1] = pFC[b, 1] / K
        pFC[b, 2] = np.angle(corr)

    pFC[:, 0] = np.sqrt(pFC[:, 0])
    pFC[:, 1] = np.sqrt(pFC[:, 1])
    pFC[:, 2] = np.unwrap(pFC[:, 2])
    return pFC


brir = np.zeros((B2, 3, 5))

i = 0
for brir_file in BRIRs:
    brir_file = BRIRs_path + brir_file
    y, sr = librosa.load(brir_file, sr=sample_rate, mono=False)
    brirL_stft = librosa.stft(y[0,:], n_fft=n_fft_bins, hop_length=hop_length, win_length=window_length, window=sin_window)
    brirR_stft = librosa.stft(y[1,:], n_fft=n_fft_bins, hop_length=hop_length, win_length=window_length, window=sin_window)

    brir[:,:,i] = brir_para(brirL_stft, brirR_stft)
    i = i + 1

# batch test
src_path = 'src/'
dst_path = 'out/'
ver = '_0325a02'
test_files = ['test1']

for filename in test_files:
    # load test file
    file_path = src_path + filename + '.wav'
    sig, sr = librosa.load(file_path, sr=sample_rate, mono=False)
    SL = librosa.stft(sig[0,:], n_fft=n_fft_bins, hop_length=hop_length, win_length=window_length, window=sin_window)
    SR = librosa.stft(sig[1,:], n_fft=n_fft_bins, hop_length=hop_length, win_length=window_length, window=sin_window)

    M = SL.shape[1]

    # Binaural rendering by J. Breebaart
    a_BS = np.zeros((B2, M, 2))
    v_BS = np.zeros((B2, M))
    b_BS = np.zeros((B2, M))
    g_BS = np.zeros((B2, M))
    SD_BS = np.zeros((SL.shape[0], B2, M, 2), dtype=np.complex_)

    EPS = 0.00001

    # spatial analysis
    for m in range(M):
        for b in range(B2):
            L = SL[:, m] * filters2[:, b]
            R = SR[:, m] * filters2[:, b]
            a_BS[b, m, 0] = np.sqrt(np.real(np.vdot(L, L)))
            a_BS[b, m, 1] = np.sqrt(np.real(np.vdot(R, R)))
            v_BS[b, m] = 0.5 * np.arccos(np.real(np.vdot(L, R)) / (a_BS[b, m, 0] * a_BS[b, m, 1] + EPS))
            b_BS[b, m] = np.tan(((a_BS[b, m, 1] - a_BS[b, m, 0]) / (a_BS[b, m, 1] + a_BS[b, m, 0] + EPS)) * np.arctan(v_BS[b,m]))
            g_BS[b, m] = np.arctan((a_BS[b, m, 0] * np.cos(v_BS[b, m] + b_BS[b, m])) / (a_BS[b, m, 1] * np.cos(-v_BS[b, m] + b_BS[b, m]) + EPS))

            SD_BS[:, b, m, 0] = (L + R) / (np.sin(g_BS[b, m]) + np.cos(g_BS[b, m]))
            SD_BS[:, b, m, 1] = L - np.sin(g_BS[b, m]) * SD_BS[:, b, m, 0]

    # EQ
    EQ1 = np.ones(B2)
    EQ1[0] = 2.2
    EQ1[1] = 2.2
    EQ1[2] = 2.2
    EQ1[3] = 2.0
    EQ1[4] = 2.0
    EQ1[5] = 1.8
    EQ1[6] = 1.7
    EQ1[7] = 1.5
    EQ1[8] = 1.3
    EQ1[9] = 1.2
    EQ2 = np.ones(B2)
    for b in range(10):
        EQ2[b] = EQ1[b]
    EQ2[10] = 1.7
    EQ2[11] = 2.2
    EQ2[12] = 2.4
    EQ2[13] = 2.7
    EQ2[14] = 3.0
    EQ2[15] = 2.8
    EQ2[16] = 2.7
    EQ2[17] = 2.6
    EQ2[18] = 2.5
    EQ2[19] = 2.0
    EQ2[20] = 1.8
    EQ2[21] = 1.5
    EQ2[22] = 1.1


    for m in range(M):
        for b in range(B2):
            SD_BS[:, b, m, 0] = SD_BS[:, b, m, 0] * EQ1[b] * filters2[:, b]
            SD_BS[:, b, m, 1] = SD_BS[:, b, m, 1] * EQ2[b] * filters2[:, b]

    # angle compute
    ang_BS = np.abs(g_BS * 180.0 / np.pi)
    ang_BS = 60 - ang_BS * (120.0) / 90.0

    # spatial synthesis
    (FC, FL, FR, FLc, FRc) = [0, 1, 2, 3, 4]
    Y_out = np.zeros((SL.shape[0], M, 2), dtype=np.complex_)

    for m in range(M):
        for b in range(B2):
            if ang_BS[b, m] < -30:
                S_brir = FL
            elif ang_BS[b, m] < -10:
                S_brir = FLc
            elif ang_BS[b, m] < 10:
                S_brir = FC
            elif ang_BS[b, m] < 30:
                S_brir = FRc
            else:
                S_brir = FR

            S = SD_BS[:, b, m, 0]
            D = SD_BS[:, b, m, 1]

            Y_out[:, m, 0] = Y_out[:, m, 0] + S * brir[b, 0, S_brir] * np.exp(-1j * brir[b, 2, S_brir] * 0.5)
            Y_out[:, m, 0] = Y_out[:, m, 0] + D * brir[b, 0, FL] * np.exp(-1j * brir[b, 2, FL] * 0.5)
            Y_out[:, m, 0] = Y_out[:, m, 0] - D * brir[b, 0, FR] * np.exp(-1j * brir[b, 2, FR] * 0.5)

            Y_out[:, m, 1] = Y_out[:, m, 1] + S * brir[b, 1, S_brir] * np.exp(1j * brir[b, 2, S_brir] * 0.5)
            Y_out[:, m, 1] = Y_out[:, m, 1] + D * brir[b, 1, FL] * np.exp(1j * brir[b, 2, FL] * 0.5)
            Y_out[:, m, 1] = Y_out[:, m, 1] - D * brir[b, 1, FR] * np.exp(1j * brir[b, 2, FR] * 0.5)

    y_out_L = librosa.istft(Y_out[:, :, 0], hop_length=hop_length, win_length=window_length, window=sin_window)
    y_out_R = librosa.istft(Y_out[:, :, 1], hop_length=hop_length, win_length=window_length, window=sin_window)

    y = np.array([y_out_L, y_out_R])

    y = y * BRIR_gain
    sig_out = nussl.AudioSignal(audio_data_array=y, sample_rate=sample_rate)

    sig_out.write_audio_to_file(dst_path + filename + ver + '.wav')
