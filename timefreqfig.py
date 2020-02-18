import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

import numpy as np
from scipy.fftpack import fft


# 输出音频波形
filepath = 'test.wav'
#fs采样频率 wavsigal音频数据
fs, wavsignal = wav.read(filepath)
print(type(wavsignal))
print(wavsignal.shape)
plt.plot(wavsignal)
plt.show()

#构造hamming window
x=np.linspace(0, 400 - 1, 400, dtype = np.int64)#返回区间内的均匀数字
# print(x)
w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (400 - 1))
plt.plot(w)
plt.show()

#对数据分帧
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
frame = wavsignal[p_begin:p_end]

plt.figure(figsize=(15, 5))
ax4 = plt.subplot(121)
plt.title('the original picture of one frame')
plt.plot(frame)

# plt.show()
# 加窗

frame = frame * w
ax5 = plt.subplot(122)
plt.title('after hanmming')
plt.plot(frame)
plt.show()

# 进行快速傅里叶变换
frame_fft = np.abs(fft(frame))[:200]
plt.plot(frame_fft)
plt.show()

# 取对数，求db
frame_log = np.log(frame_fft)
plt.plot(frame_log)
plt.show()

# 获取信号的时频图
def compute_fbank(file):
    x=np.linspace(0, 400 - 1, 400, dtype = np.int64)
    # 汉明窗
    w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (400 - 1) )
    fs, wavsignal = wav.read(file)
    # wav波形 加时间窗以及时移10ms
    time_window = 25 # 单位ms
    # 计算窗长度的公式，目前全部为400固定值
    window_length = fs / 1000 * time_window
    wav_arr = np.array(wavsignal)
    wav_length = len(wavsignal)
    # 计算循环终止的位置，也就是最终生成的窗数
    range0_end = int(len(wavsignal)/fs*1000 - time_window) // 10
    # 	print(range0_end)
    # 用于存放最终的频率特征数据
    data_input = np.zeros((range0_end, 200), dtype = np.float)
    # 窗口内的数据
    data_line = np.zeros((1, 400), dtype = np.float)
    for i in range(0, range0_end):
        p_start = i * 160  # 步长10ms
        p_end = p_start + 400  # 窗口长25ms
        data_line = wav_arr[p_start:p_end]
        data_line = data_line * w  # 加窗
        data_line = np.abs(fft(data_line))
        data_input[i]=data_line[0:200]    # 设置为400除以2的值（即200）是取一半数据，因为是对称的
    data_input = np.log(data_input + 1)
    #data_input = data_input[::]
    return data_input




a = compute_fbank(filepath)
print(a.shape)
plt.imshow(a.T, origin = 'lower')
plt.show()
