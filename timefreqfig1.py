import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.fftpack import fft
from timefreqfig import compute_fbank
from random import shuffle
import keras
from keras import backend as K
from languagemodel import ModelLanguage


source_file = 'data_thchs30'


def source_get(source_file):
    train_file = source_file + '/data'
    label_lst = []
    wav_lst = []
    for root, dirs, files in os.walk(train_file):
        for file in files:
            if file.endswith('.wav') or file.endswith('.WAV'):
                wav_file = os.sep.join([root, file])
                label_file = wav_file + '.trn'
                wav_lst.append(wav_file)
                label_lst.append(label_file)

    return label_lst, wav_lst


label_lst, wav_lst = source_get(source_file)

print(label_lst[:10])
print(wav_lst[:10])
#确认相同id对应的音频文件和标签文件相同
for i in range(10000):
    wavname = (wav_lst[i].split('/')[-1]).split('.')[0]
    labelname = (label_lst[i].split('/')[-1]).split('.')[0]
    if wavname != labelname:
        print('error')


def read_label(label_file):
    with open(label_file, 'r', encoding='utf8') as f:
        data = f.readlines()
        return data[1]


print(read_label(label_lst[0]))


def gen_label_data(label_lst):
    label_data = []
    for label_file in label_lst:
        pny = read_label(label_file)
        label_data.append(pny.strip('\n'))
    return label_data


label_data = gen_label_data(label_lst)
print(len(label_data))


#建立词典
def mk_vocab(label_data):
    vocab = []
    for line in label_data:
        line = line.split(' ')
        for pny in line:
            if pny not in vocab:
                vocab.append(pny)
    vocab.append('_')
    return vocab

vocab = mk_vocab(label_data)
# print(len(vocab))


def word2id(line, vocab):
    return [vocab.index(pny) for pny in line.split(' ')]


def get_batch(batch_size, shuffle_list, wav_lst, label_data, vocab):
    for i in range(10000//batch_size):
        wav_data_lst = []
        label_data_lst = []
        begin = i * batch_size
        end = begin + batch_size
        sub_list = shuffle_list[begin:end]
        for index in sub_list:
            fbank = compute_fbank(wav_lst[index])
            fbank = fbank[:fbank.shape[0] // 8 * 8, :]
            label = word2id(label_data[index], vocab)
            wav_data_lst.append(fbank)
            label_data_lst.append(label)
        yield wav_data_lst, label_data_lst



def wav_padding(wav_data_lst):
    wav_lens = [len(data) for data in wav_data_lst]
    wav_max_len = max(wav_lens)
    wav_lens = np.array([leng//8 for leng in wav_lens])
    new_wav_data_lst = np.zeros((len(wav_data_lst), wav_max_len, 200, 1))
    for i in range(len(wav_data_lst)):
        new_wav_data_lst[i, :wav_data_lst[i].shape[0], :, 0] = wav_data_lst[i]
    return new_wav_data_lst, wav_lens


def label_padding(label_data_lst):
    label_lens = np.array([len(label) for label in label_data_lst])
    max_label_len = max(label_lens)
    new_label_data_lst = np.zeros((len(label_data_lst), max_label_len))
    for i in range(len(label_data_lst)):
        new_label_data_lst[i][:len(label_data_lst[i])] = label_data_lst[i]
    return new_label_data_lst, label_lens




def data_generator(batch_size, shuffle_list, wav_lst, label_data, vocab):
    for i in range(len(wav_lst)//batch_size):
        wav_data_lst = []
        label_data_lst = []
        begin = i * batch_size
        end = begin + batch_size
        sub_list = shuffle_list[begin:end]
        for index in sub_list:
            fbank = compute_fbank(wav_lst[index])
#             print(wav_lst[index])
            pad_fbank = np.zeros((fbank.shape[0]//8*8+8, fbank.shape[1]))
            pad_fbank[:fbank.shape[0], :] = fbank
            label = word2id(label_data[index], vocab)
            wav_data_lst.append(pad_fbank)
            label_data_lst.append(label)
#             break
        pad_wav_data, input_length = wav_padding(wav_data_lst)
        pad_label_data, label_length = label_padding(label_data_lst)
        inputs = {'the_inputs': pad_wav_data,
                  'the_labels': pad_label_data,
                  'input_length': input_length,
                  'label_length': label_length,
                 }
#         print(pad_wav_data,pad_label_data,input_length,label_length)
        outputs = {'ctc': np.zeros(pad_wav_data.shape[0],)}
        yield inputs, outputs





label_id = word2id(label_data[0], vocab)
print(label_data[0])
print(label_id)

#词典检验
print('---------词典检验-----------')
print(vocab[:15])
print(label_data[10])
print(word2id(label_data[10], vocab))

fbank = compute_fbank(wav_lst[0])
print(wav_lst[0])
print(fbank.shape)

plt.imshow(fbank.T, origin = 'lower')
plt.show()

#三个max pooling
fbank = fbank[:fbank.shape[0]//8*8, :]
print(fbank.shape)


#生成训练数据

total_nums = 10000
batch_size = 4
batch_num = total_nums // batch_size

shuffle_list = [i for i in range(10000)]
shuffle(shuffle_list)


batch = get_batch(4, shuffle_list, wav_lst, label_data, vocab)

wav_data_lst, label_data_lst = next(batch)
for wav_data in wav_data_lst:
    print('Shape of wave data',wav_data.shape)
for label_data in label_data_lst:
    print('label data:',label_data)

lens = [len(wav) for wav in wav_data_lst]
print(max(lens))
print(lens)

pad_wav_data_lst, wav_lens = wav_padding(wav_data_lst)
print('Shape of wave data pad',pad_wav_data_lst.shape)
print('wave lens:',wav_lens)





with open('model_def.json') as ff:
    model_json=ff.read()
    model=keras.models.model_from_json(model_json)
model.load_weights('res2.h5')

result, text = decode_ctc(preds, py)

print(result,text)

preds=model.predict(inputdata)
ml = ModelLanguage('model_language')
ml.LoadModel()
str_pinyin = text
r = ml.SpeechToText(str_pinyin)
print('语音识别结果：\n',r)