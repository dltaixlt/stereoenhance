import os
import wave

def wavconcate(wav_a, wav_b, wav_out, a_time_sec, b_time_sec):
    in_a = wave.open(wav_a, "rb")
    in_b = wave.open(wav_b, "rb")
    out = wave.open(wav_out, "wb")

    if not in_a or not in_b or not out:
        print("wav open error!")
        return []

    if not (in_a.getframerate() == in_b.getframerate()):
        print("wav sample rate not equal! (%d, %d)" % (in_a.getframerate(), in_b.getframerate()))
        return []

    if not (in_a.getnchannels() == in_b.getnchannels()):
        print("wav channels not equal! (%d, %d)" % (in_a.getnchannels(), in_b.getnchannels()))
        return []

    if not (in_a.getnframes() == in_b.getnframes()):
        print("wav frames not equal! (%d, %d)" % (in_a.getnframes(), in_b.getnframes()))
        #return []

    if not (in_a.getsampwidth() == in_b.getsampwidth()):
        print("wav sampwidth not equal! (%d, %d)" % (in_a.getsampwidth(), in_b.getsampwidth()))
        return []

    samplerate = in_a.getframerate()
    nchannels = in_a.getnchannels()
    nframes = min(in_a.getnframes(), in_b.getnframes())
    sampwidth = in_a.getsampwidth()
    print("samplerate: %d" % samplerate)
    print("nchannels: %d" % nchannels)
    print("nframes: %d" % nframes)
    print("sampwidth: %d" % sampwidth)

    a_time_frame = samplerate * a_time_sec
    b_time_frame = samplerate * b_time_sec

    out.setnchannels(nchannels)
    out.setframerate(samplerate)
    out.setsampwidth(sampwidth)
    out.setnframes(nframes)

    pos = 0
    while pos < nframes:
        nread = min(nframes - pos, a_time_frame)
        out.writeframes(in_a.readframes(nread))
        in_b.readframes(nread)
        pos = pos + nread

        nread = min(nframes - pos, b_time_frame)
        out.writeframes(in_b.readframes(nread))
        in_a.readframes(nread)
        pos = pos + nread

    in_a.close()
    in_b.close()
    out.close()

    # check
    out_check = wave.open(wav_out, "rb")
    if not (out_check.getnframes() == nframes):
        print("nframe of out wav is error! (%d, %d)" % (out_check.getnframes(), nframes))

if __name__ == "__main__":
    wav_1 = 'out/music.erbenh4.R09R10.wav'
    wav_2 = 'src/music.wav'
    wav_out = 'tmp/music_interlaced20.wav'

    wavconcate(wav_1, wav_2, wav_out, 20, 10)

