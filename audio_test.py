# -*- encoding:utf-8 -*-
import numpy as np
import numpy.fft as nf
import scipy.io.wavfile as wf
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import python_speech_features as sf
import matplotlib.pyplot as mp



fs, sigs = wf.read('FC_l1.wav')
print(fs)      # 采样率
print(sigs.shape)
sigs = sigs / (2 ** 15) # 归一化
times = np.arange(len(sigs)) / fs
freqs = nf.fftfreq(sigs.size, 1 / fs)
ffts = nf.fft(sigs)
pows = np.abs(ffts)



#梅尔频率倒谱系数
mfcc = sf.mfcc(sigs, fs)

mp.matshow(mfcc.T, cmap='gist_rainbow')
mp.show()

print(type(sigs))
print(sigs.shape)
plt.plot(sigs)

#
# #绘制语音时域频域波形
# plt.figure('Audio')
# plt.subplot(121)
# plt.title('Time Domain')
# plt.xlabel('Time', fontsize=12)
# plt.ylabel('Signal', fontsize=12)
# plt.tick_params(labelsize=10)
# plt.grid(linestyle=':')
# plt.plot(times, sigs, c='dodgerblue', label='Signal')
# plt.legend()
# plt.subplot(122)
# plt.title('Frequency Domain')
# plt.xlabel('Frequency', fontsize=12)
# plt.ylabel('Power', fontsize=12)
# plt.tick_params(labelsize=10)
# plt.grid(linestyle=':')
# plt.plot(freqs[freqs >= 0], pows[freqs >= 0], c='orangered', label='Power')
# plt.legend()
# plt.tight_layout()
# plt.show()



#构造hamming窗
x=np.linspace(0, 400 - 1, 400, dtype = np.int64)#返回区间内的均匀数字
# print(x)
w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (400 - 1))
plt.plot(w)
plt.show()

#分帧提取
'''
帧长： 25ms
帧移： 10ms
采样点（s） = fs
采样点（ms）= fs / 1000
采样点（帧）= fs / 1000 * 帧长
'''
time_window = 25
window_length = fs // 1000 * time_window
#保持window

# 分帧
p_begin = 0
p_end = p_begin + window_length
frame = sigs[p_begin:p_end]

plt.figure(figsize=(15, 5))
ax4 = plt.subplot(121)
plt.title('the original picture of one frame')
plt.plot(frame)

# plt.show()
# 加窗

frame = np.dot(frame,[w,w])
ax5 = plt.subplot(122)
plt.title('after hanmming')
plt.plot(frame)
plt.show()


#快速傅里叶变换
frame_fft = np.abs(fft(frame))[:200]
plt.plot(frame_fft)
plt.show()

# 取对数，求db
frame_log = np.log(frame_fft)
plt.plot(frame_log)
plt.show()



