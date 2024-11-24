from os import O_TMPFILE
import pywt
import neurokit2 as nk
import numpy as np
import logging
import math

logger = logging.getLogger(__name__)

leads_idx = {'I': 0, 'II': 1, 'III':2, 'aVR': 3, 'aVL':4, 'aVF':5, 'V1':6, 'V2':7, 'V3':8, 'V4':9, 'V5':10, 'V6':11}

def leading_rythm(bpm):
    if(bpm < 50):
       return -1
    elif (50< bpm < 100):
        return 0
    else:
        return 1



def get_wavelet_orientation(onset, peak, offset):
    if peak > (onset+offset)/2 :
        return 1
    else:
        return -1

#R->L (correct) return 1, L->R (incorrect) return -1 LEAD I
def get_right_left_activation_leadI(p_complexes):
    if len(p_complexes) > 0:
        return cleanse_data_mean([get_wavelet_orientation(*px) for px in p_complexes])
    else:
        return 0

#Correct sinus return 1 (pwave positive), Extra sinus orign return -1 (pwave negative) on lead II or lead III or aVF
def get_vertical_orientation(p_complexes):
    if len(p_complexes) > 0:
        return cleanse_data_mean([get_wavelet_orientation(*px) for px in p_complexes])
    else:
        return 0

def get_p_complex(signals, info):
    num_peaks = len(info['ECG_P_Peaks'])
    result=[]
    for i in range(num_peaks):
        p_on = info['ECG_P_Onsets'][i]
        p = info['ECG_P_Peaks'][i]
        p_off = info['ECG_P_Offsets'][i]

        if np.isnan([p_on, p, p_off]).any():
            continue

        p_complex = [signals[p_on], signals[p], signals[p_off]]

        if np.isnan(p_complex).any():
            continue
        else:
            result.append(p_complex)
    return result



#          | R   |  L
#---------------------
# top-down | 1 1 | 1 -1
#---------------------
# down-top | -1 1 | -1 -1
def get_rhythm_origin(signalsI, infoI, signalsII, infoII):
    I_pcomplexes = get_p_complex(signalsI, infoI)
    II_pcomplexes = get_p_complex(signalsII, infoII)

    return get_vertical_orientation(II_pcomplexes), get_right_left_activation_leadI(I_pcomplexes)




#https://www.cmj.hr/1999/40/1/9933900.htm
def get_heart_axis(leadI_QRS, leadaVF_QRS):
    results=[]
    lenI = len(leadI_QRS)
    lenaVF = len(leadaVF_QRS)
    target_len = 0
    if lenI >= lenaVF:
        target_len = lenaVF
    else:
        target_len = lenI

    for i in range(target_len):
        aVF_QRS=leadaVF_QRS[i]
        I_QRS=leadI_QRS[i]
        altitudeAVF=sum(aVF_QRS)
        altitudeI=sum(I_QRS)
        results.append(math.degrees(math.atan2((2*altitudeAVF),(math.sqrt(3)*altitudeI))))
    return results



def get_QRS_from_lead(signal, info):
    num_peaks = len(info['ECG_R_Peaks'])
    result = []
    for i in range(num_peaks):
        Q = info['ECG_Q_Peaks'][i]
        R = info['ECG_R_Peaks'][i]
        S = info['ECG_S_Peaks'][i]

        QRS_ts= [Q, R, S]
        if np.isnan(QRS_ts).any():
            continue

        QRS = [signal[Q], signal[R], signal[S]]
        if np.isnan(QRS).any():
            continue
        else:
            result.append(QRS)

    return result


#Check if there are missing QRS complexes, if so we diagnose atrioventricular block
def has_missing_qrs(signals, info):
    R_peaks = info['ECG_R_Peaks']
    distances = np.diff(R_peaks)
    quantile90=np.quantile(distances,0.9)
    quantile10=np.quantile(distances,0.1)
    outliers_removed=[d for d in distances if (d>quantile10 and d<quantile90)]
    if len(outliers_removed) > 0:
        mean_without_outliers = cleanse_data_mean(outliers_removed)
        is_missing_qrs = distances > (mean_without_outliers * 1.5)
        return any(is_missing_qrs)
    else:
        return -1


def has_missing_p(signals, info):
    num_of_p = np.count_nonzero(~np.isnan(info['ECG_P_Peaks']))
    num_of_beats = len(info['ECG_R_Peaks'])
    return num_of_p < (num_of_beats - 1)


def get_QRS_duration(signals, info, freq=500):
    num_peaks = min([len(info['ECG_R_Peaks']), len(info['ECG_Q_Peaks']), len(info['ECG_S_Peaks'])])
    result = []
    for i in range(num_peaks):
        Q = info['ECG_Q_Peaks'][i]
        R = info['ECG_R_Peaks'][i]
        S = info['ECG_S_Peaks'][i]
        if any(np.isnan([Q,R,S])):
            continue
        else:
            result.append((S-Q)/freq)

    return result

def get_S_duration(signals, info, freq=500):
    num_peaks = len(info['ECG_R_Peaks'])
    result = []
    for i in range(num_peaks):
        s = info['ECG_S_Peaks'][i]
        r = info['ECG_R_Peaks'][i]
        if any(np.isnan([r,s])):
            continue
        else:
            result.append((s-r)/freq)
    return result


def get_R_duration(signals, info, freq=500):
    num_peaks = len(info['ECG_R_Peaks'])
    result = []
    for i in range(num_peaks):
        R = info['ECG_R_Peaks'][i]
        R_on = info['ECG_R_Onsets'][i]
        R_off = info['ECG_R_Offsets'][i]
        S = info['ECG_S_Peaks'][i]
        if any(np.isnan([R,R_on, R_off, S])):
            continue
        else:
            if S < R_off:
                result.append((S - R_on)/freq)
            else:
                result.append((R_off - R_on)/freq)

    return result


def get_0_crossings(biorcD, beg_qrs, end_qrs, threshold=15, show=False, **kwargs):
    bior_qrs_beg = beg_qrs // 2
    bior_qrs_end = end_qrs // 2

    biorcD_widnows_beat = [biorcD[bior_qrs_beg[i]:bior_qrs_end[i]] for i in range(len(bior_qrs_beg)) if bior_qrs_end[i] - bior_qrs_beg[i] > threshold ]
    biorcD_min_max_beat = []

    for wind in biorcD_widnows_beat :
        argmin = np.argmin(wind)
        argmax = np.argmax(wind)
        if argmax > argmin:
            biorcD_min_max_beat.append(wind[argmin:argmax])
        else:
            biorcD_min_max_beat.append(wind[argmax:argmin])

    crossing_0 = [((window[:-1] * window[1:]) < 0).sum() for window in biorcD_min_max_beat]

    if show:
        figure, axis = plt.subplots(5, 1)
        figure.set_size_inches(24,10)
        for i in range(len(biorcD_widnows_beat)):
            beat = biorcD_widnows_beat[i]
            x= range(len(beat))
            axis[i].bar(x, beat)
            axis[i].bar(x, np.zeros(len(x)), linestyle='dashed')

    return crossing_0




def analyse_notched_signal(signal, info, recording, threshold=1.5, peaks=None, **kwargs):
    #list_of_qrs = get_QRS_from_lead(signal, info) #get_qrs_beginning_and_end(signal['ECG_Raw'], **kwargs)
    N = len(recording)
    beginning_key = 'ECG_Q_Peaks'
    ending_key = 'ECG_S_Peaks'
    window = None
    if 'ECG_Q_Peaks' not in info or 'ECG_S_Peaks' not in info:
        if 'ECG_R_Onsets' in info or 'ECG_R_Offsets' not in info:
            window = 100
        else:
            beginning_key = 'ECG_R_Onsets'
            ending_key = 'ECG_R_Offsets'

    list_of_qrs = []
    if 'ECG_R_Peaks' not in info and peaks is not None:
        window = 100
        for peak in peaks:
            if peak < (window // 2) + 1:
                list_of_qrs.append([0, window])
            elif peak > ( N - window//2):
                list_of_qrs.append([N-window-1,N-1])
            else:
                list_of_qrs.append([(peak-window//2), (peak+window//2)])
    else:
        beg = info[beginning_key]
        end = info[ending_key]

        for i in range(len(beg)):
            b = beg[i]
            e = end[i]
            if np.isnan([b,e]).any():
                continue
            list_of_qrs.append([b, e])

   # list_of_qrs = [[info[beginning_key][i], info[beginning_key][i]] for i in len(info['ECG_R_Peaks'])]
    if len(list_of_qrs) ==0:
        return -1

    list_of_qrs = np.array(list_of_qrs)

    beg_qrs = list_of_qrs[:, 0]
    end_qrs = list_of_qrs[:, 1]
    (_, cD) = pywt.dwt(recording, 'bior1.1')
    crossing_0 = get_0_crossings(cD, beg_qrs, end_qrs, **kwargs)
    if len(crossing_0) > 0:
        return cleanse_data_mean(crossing_0)
    else:
        return -1






def cleanse_data_mean(array):
    if len(array) > 0 and array is not None:
        result = np.nan_to_num(array, posinf=99999, neginf=-99999)
        return  np.mean(result)
    else:
        return -1



def analyse_recording(rec, signals, infos, rates,  pantompkins_peaks=None, label=None, leads_idxs=leads_idx, sampling_rate=500):
    logger.debug("Entering analysed_results")
    analysed_results = {}
    for lead_name, idx in leads_idxs.items():
        signal = signals[lead_name]
        info = infos[lead_name] #nk.ecg_process(rec[idx], sampling_rate=sampling_rate)
        if signl is None or info is None:
            analysed_results[lead_name]={
                'signal': rec[idx],
                'info': None,
                'bpm': -1,
                'missing_qrs': -1,
                'missing_p': -1,
                'qrs_duration': -1,
                's_duration': -1,
                'rhythm': None,
                # 'has_rsr': rsr,
                'notched': -1,
            }
            continue

        bpm = -1
        if lead_name in rates:
            bpm = cleanse_data_mean(rates[lead_name])
        missing_qrs = has_missing_qrs(signal, info)
        missing_p = has_missing_p(signal, info)
        qrs_duration = cleanse_data_mean(get_QRS_duration(signal, info))
        s_duration = cleanse_data_mean(get_S_duration(signal, info))
        rhythm = leading_rythm(bpm)
        # rsr = has_rsR_complex(rec[idx], sampling_rate)
        notched = analyse_notched_signal(signal,info, rec[idx], peaks=pantompkins_peaks)

        analysed_results[lead_name]={
            'signal': rec[idx],
            'info': info,
            'bpm': bpm,
            'missing_qrs':missing_qrs,
            'missing_p': missing_p,
            'qrs_duration':qrs_duration,
            's_duration':s_duration,
            'rhythm': rhythm,
            # 'has_rsr': rsr,
            'notched': notched,
        }

    heart_axis = None
    rhythm_origin = None
    if 'I' in leads_idx and analysed_results['I']['info'] is not None:
        if 'II' in leads_idx and analysed_results['II']['info'] is not None:
            rhythm_origin = get_rhythm_origin(analysed_results['I']['signal'], analysed_results['I']['info'], analysed_results['II']['signal'], analysed_results['II']['info'])
        if 'aVF' in leads_idx and analysed_results['aVF']['info'] is not None:
            if 'II' not in leads_idx:
                rhythm_origin = get_rhythm_origin(analysed_results['I']['signal'], analysed_results['I']['info'], analysed_results['aVF']['signal'], analysed_results['aVF']['info'])
            heart_axis = get_heart_axis(get_QRS_from_lead(analysed_results['I']['signal'], analysed_results['I']['info']), get_QRS_from_lead(analysed_results['aVF']['signal'], analysed_results['aVF']['info']))


    if heart_axis:
        analysed_results['heart_axis']=cleanse_data_mean(heart_axis)

    if rhythm_origin:
        analysed_results['rhythm_origin_vertical']=rhythm_origin[0]
        analysed_results['rhythm_origin_horizontal']=rhythm_origin[1]



    analysed_results['label'] = label

    for lead_name, idx in leads_idxs.items():
        analysed_results[lead_name].pop('signal', None)
        analysed_results[lead_name].pop('info', None)


    return analysed_results

def analysis_dict_to_array(analysis_dict, leads_idxs=leads_idx):
    result = []
    logger.debug(analysis_dict)
    signals_to_extract = ['bpm', 'missing_qrs', 'missing_p', 'qrs_duration', 's_duration', 'rhythm', 'notched', 'heart_axis', 'rhythm_origin_vertical', 'rhythm_origin_horizontal']
    for lead_name, idx in leads_idxs.items():
        tmp_result = []
        for key in signals_to_extract:
            if key in analysis_dict[lead_name]:
                tmp_result.append(analysis_dict[lead_name][key])
            else:
                tmp_result.append(0)
        result.append(tmp_result)

    return np.array(result, dtype=np.float64)
