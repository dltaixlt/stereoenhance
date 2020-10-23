import subprocess
from stereo_speech_enhance import StereoSpeechEnhance

def do_cmd(s_cmd, log=False, out=False, err=False):
    dict = {False: None, True: subprocess.PIPE}

    child = subprocess.Popen(
        s_cmd,
        shell=True,
        stdout=dict.get(out, None),
        stderr=dict.get(err, None))
    res = child.communicate()  # stdout stderr include '\n'
    ret = child.returncode

    return ret, res[0], res[1]

fg_brir_size = 2048
bk_brir_size = 4096
fg_gain = 0.7
bk_gain = 0.8
total_gain = 0.7
COPYAUDIO = 'copyaudio '
BRIRCONV = 'brirconvSL '

#filelist = ['v0033x24lxf.1','v0033x24lxf.2','v0033x24lxf.3','v0033x24lxf.4','v0033x24lxf.5','v0033x24lxf.6','v0033x24lxf.7','v0033x24lxf.8','v0033x24lxf.9','v0033x24lxf.10']
filelist = ['in']
funcversion = '0109'

for f1 in filelist:
    infile = 'src/' + f1 + '.wav '
    fg = 'tmp/' + f1 + '_%s_fg.wav ' % (funcversion)
    fg_m = 'tmp/' + f1 + '_fg.m.wav '
    fg_m_L = 'tmp/' + f1 + '_fg.m.L.wav '
    fg_m_R = 'tmp/' + f1 + '_fg.m.R.wav '
    fg_2ch = 'out/' + f1 + '_fg.%s.wav ' % (funcversion)
    bk = 'tmp/' + f1 + '_%s_bk.wav ' % (funcversion)
    bk_left = 'tmp/' + f1 + '_bk.left.wav '
    bk_left_L = 'tmp/' + f1 + '_bk.left.L.wav '
    bk_left_R = 'tmp/' + f1 + '_bk.left.R.wav '
    bk_right = 'tmp/' + f1 + '_bk.right.wav '
    bk_right_L = 'tmp/' + f1 + '_bk.right.L.wav '
    bk_right_R = 'tmp/' + f1 + '_bk.right.R.wav '
    bg_2ch = 'out/' + f1 + '_bk.%s.wav' % (funcversion)
    comb_6ch = 'tmp/' + f1 + '_6ch.wav '
    outfile = 'out/' + f1 + '.%s.%.1f.wav ' % (funcversion, bk_gain)

    # foreground/background separation
    if True:
        erbenh = StereoSpeechEnhance(infile)
        erbenh.run()
        background, foreground = erbenh.make_audio_signals()
        foreground.write_audio_to_file(fg)
        background.write_audio_to_file(bk)

    # brir processing
    # extract direct_mono and diffuse left and right
    cmd_extr = COPYAUDIO + ' --chanA="1.0*A" ' + fg + ' ' + fg_m
    do_cmd(cmd_extr)
    cmd_extr = COPYAUDIO + ' --chanA="1.0*A" ' + bk + ' ' + bk_left
    do_cmd(cmd_extr)
    cmd_extr = COPYAUDIO + ' --chanA="1.0*B" ' + bk + ' ' + bk_right
    do_cmd(cmd_extr)

    # process direct_mono
    cmd_brir = BRIRCONV + fg_m + ' brir/brir_FC2_l_48.wav ' + str(fg_brir_size) + ' ' + fg_m_L + str(fg_gain)
    do_cmd(cmd_brir)
    cmd_brir = BRIRCONV + fg_m + ' brir/brir_FC2_r_48.wav ' + str(fg_brir_size) + ' ' + fg_m_R + str(fg_gain)
    do_cmd(cmd_brir)
    cmd_comb = COPYAUDIO + ' -c ' + fg_m_L + fg_m_R + fg_2ch
    do_cmd(cmd_comb)

    # process diffuse_left & right
    cmd_brir = BRIRCONV + bk_left + ' brir/brir_FL2_l_48.wav ' + str(bk_brir_size) + ' ' + bk_left_L + str(bk_gain)
    do_cmd(cmd_brir)
    cmd_brir = BRIRCONV + bk_left + ' brir/brir_FL2_r_48.wav ' + str(bk_brir_size) + ' ' + bk_left_R + str(bk_gain)
    do_cmd(cmd_brir)
    cmd_brir = BRIRCONV + bk_right + ' brir/brir_FR2_l_48.wav ' + str(bk_brir_size) + ' ' + bk_right_L + str(bk_gain)
    do_cmd(cmd_brir)
    cmd_brir = BRIRCONV + bk_right + ' brir/brir_FR2_r_48.wav ' + str(bk_brir_size) + ' ' + bk_right_R + str(bk_gain)
    do_cmd(cmd_brir)

    # combine
    cmd_comb = COPYAUDIO + ' -c ' + fg_m_L + fg_m_R + bk_left_L + bk_left_R + bk_right_L + bk_right_R + comb_6ch
    do_cmd(cmd_comb)
    cmd_comb = COPYAUDIO + ' --chanA="1.0*C+1.0*E" --chanB="1.0*D+1.0*F" ' + comb_6ch + bg_2ch
    do_cmd(cmd_comb)
    cmd_comb = COPYAUDIO + ' -g ' + str(total_gain) + ' --chanA="1.0*A+1.0*C+1.0*E" --chanB="1.0*B+1.0*D+1.0*F" ' + comb_6ch + outfile
    do_cmd(cmd_comb)