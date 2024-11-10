import pywt
import neurokit2 as nk
import numpy as np
import math


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
    return np.mean([get_wavelet_orientation(*px) for px in p_complexes])

#Correct sinus return 1 (pwave positive), Extra sinus orign return -1 (pwave negative) on lead II or lead III or aVF
def get_vertical_orientation(p_complexes):
    return np.mean([get_wavelet_orientation(*px) for px in p_complexes])

def get_p_complex(signals, info):
    num_peaks = len(info['ECG_P_Peaks'])
    result=[]
    for i in range(num_peaks):
        p_on = info['ECG_P_Onsets'][i]
        p = info['ECG_P_Peaks'][i]
        p_off = info['ECG_P_Offsets'][i]

        if np.isnan([p_on, p, p_off]).any():
            continue

        p_complex = [signals.iloc[p_on]['ECG_Raw'], signals.iloc[p]['ECG_Raw'], signals.iloc[p_off]['ECG_Raw']]

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



def get_QRS_from_lead(signals, info):
    num_peaks = len(info['ECG_R_Peaks'])
    result = []
    for i in range(num_peaks):
        Q = info['ECG_Q_Peaks'][i]
        R = info['ECG_R_Peaks'][i]
        S = info['ECG_S_Peaks'][i]

        QRS_ts= [Q, R, S]
        if np.isnan(QRS_ts).any():
            continue

        QRS = [signals.iloc[Q]['ECG_Raw'], signals.iloc[R]['ECG_Raw'], signals.iloc[S]['ECG_Raw']]
        if np.isnan(QRS).any():
            continue
        else:
            result.append(QRS)

    return result


#Check if there are missing QRS complexes, if so we diagnose atrioventricular block
def has_missing_qrs(signals, info):
    R_peaks = info['ECG_R_Peaks']
    distances = [R_peaks[i] - R_peaks[i-1] for i in range(1, len(R_peaks))]
    quantile90=np.quantile(distances,0.9)
    quantile10=np.quantile(distances,0.1)
    mean_without_outliers = np.mean([d for d in distances if (d>quantile10 and d<quantile90)])
    is_missing_qrs = distances > (mean_without_outliers * 1.5)
    return any(is_missing_qrs)


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


def get_qrs_beginning_and_end(recording, smoothwindow=0.1, avgwindow=0.75, gradthreshweight=1.5, minlenweight=0.4, mindelay=0.3, sampling_rate=500, **kwargs):
    ecg_clean = nk.ecg_clean(recording, sampling_rate=sampling_rate, method='biosppy')
    signal_gradient = np.gradient(ecg_clean)
    absgrad = np.abs(signal_gradient)
    smooth_kernel = int(np.rint(smoothwindow * sampling_rate))
    avg_kernel = int(np.rint(avgwindow * sampling_rate))
    smoothgrad = nk.signal.signal_smooth(absgrad, kernel="boxcar", size=smooth_kernel)
    avggrad = nk.signal.signal_smooth(smoothgrad, kernel="boxcar", size=avg_kernel)
    gradthreshold = gradthreshweight * avggrad
    mindelay = int(np.rint(sampling_rate * mindelay))
    qrs = smoothgrad > gradthreshold
    beg_qrs_tmp = np.where(np.logical_and(np.logical_not(qrs[0:-1]), qrs[1:]))[0]
    end_qrs_tmp = np.where(np.logical_and(qrs[0:-1], np.logical_not(qrs[1:])))[0]
    # Throw out QRS-ends that precede first QRS-start.
    end_qrs_tmp = end_qrs_tmp[end_qrs_tmp > beg_qrs_tmp[0]]

     # Identify R-peaks within QRS (ignore QRS that are too short).
    num_qrs = min(beg_qrs_tmp.size, end_qrs_tmp.size)
    min_len = np.mean(end_qrs_tmp[:num_qrs] - beg_qrs_tmp[:num_qrs]) * minlenweight
    beg_qrs = []
    end_qrs = []

    for i in range(num_qrs):
        beg = beg_qrs_tmp[i]
        end = end_qrs_tmp[i]
        len_qrs = end - beg

        if len_qrs < min_len:
            continue
        else:
            beg_qrs.append(beg)
            end_qrs.append(end)

    return np.array(beg_qrs), np.array(end_qrs)




def analyse_notched_signal(recording, **kwargs):
    beg_qrs, end_qrs = get_qrs_beginning_and_end(recording, **kwargs)
    (cA, cD) = pywt.dwt(recording, 'bior1.1')
    avg_0_crossing = np.mean(get_0_crossings(cD, beg_qrs, end_qrs, **kwargs))
    if avg_0_crossing > 1.5:
        return 1
    else:
        return 0



def has_rsR_complex(recording, threshold=20):
    ecg, inverted = nk.ecg_invert(recording, sampling_rate=500)
    coeff_bigger_20 = []
    #For V1 with rSR there should be no inversion. If V1 is healthy inversion would occurr
    if not inverted:
        signals, info = nk.ecg_process(ecg, sampling_rate=500)
        num_peaks = len(info['ECG_R_Peaks'])
        result = []
        for i in range(num_peaks):
            r = info['ECG_R_Peaks'][i]
            r_on = info['ECG_R_Onsets'][i]
            r_off = info['ECG_R_Offsets'][i]
            p_on = info['ECG_P_Onsets'][i]
            t_off = info['ECG_T_Offsets'][i]

            if np.isnan([r, r_on, r_off, p_on, t_off]).any():
                continue
            else:
                # if signals.iloc[r_on]['ECG_Raw'] > 0:
                #     result.append(any(signals.iloc[r_on-threshold:r]['ECG_Raw'] < 0))
                # else:
                    #TODO? What if R_onset is not positive?
                    # Lets see how many times we crossed from positive to negative again, if > 0 between ron and r then its rsr
                    # pos_to_neg_changes = sum((np.diff(signals.iloc[r_on:r]['ECG_Raw']) > 0).astype(int))<0
                    # print((np.diff(signals.iloc[r_on:r]['ECG_Raw']) > 0).astype(int))
                pos_to_neg_changes = sum(np.diff(np.diff(signals.iloc[r_on-threshold:r]['ECG_Raw']) > 0).astype(int))
                (db_cA, db_cD) = pywt.dwt(signals.iloc[p_on: t_off]['ECG_Clean'], 'db2')
                coeff_bigger_20.append(sum(db_cD > 2.5))

                result.append(pos_to_neg_changes>5)

        return int(any(result))
    else:
        rec_clean = nk.ecg_clean(recording, method='pantompkins1985', sampling_rate=500)
        signals, info = nk.ecg_process(rec_clean, sampling_rate=500)
        num_peaks = len(info['ECG_R_Peaks'])
        result = []
        for i in range(num_peaks):
            r = info['ECG_R_Peaks'][i]
            r_on = info['ECG_R_Onsets'][i]
            r_off = info['ECG_R_Offsets'][i]
            p_on = info['ECG_P_Onsets'][i]
            t_off = info['ECG_T_Offsets'][i]
  
            if np.isnan([r, r_on, r_off, p_on, t_off]).any():
                continue
            else:
                # if signals.iloc[r_on]['ECG_Raw'] > 0:
                #     result.append(any(signals.iloc[r_on-threshold:r]['ECG_Raw'] < 0))
                # else:
                    #TODO? What if R_onset is not positive?
                    # Lets see how many times we crossed from positive to negative again, if > 0 between ron and r then its rsr
                    # pos_to_neg_changes = sum((np.diff(signals.iloc[r_on:r]['ECG_Raw']) > 0).astype(int))<0
                    # print((np.diff(signals.iloc[r_on:r]['ECG_Raw']) > 0).astype(int))
                pos_to_neg_changes = sum(np.diff(np.diff(signals.iloc[r: r_off]['ECG_Raw']) > 0).astype(int)) 
                result.append(pos_to_neg_changes>7)
                (db_cA, db_cD) = pywt.dwt(signals.iloc[p_on: t_off]['ECG_Clean'], 'db2')
                coeff_bigger_20.append(sum(db_cD > 5) )
              
        return int(any(result))




def analyse_recording(rec, label=None, leads_idxs=leads_idx, sampling_rate=500):
    analysed_results = {}
    for lead_name, idx in leads_idxs.items():
        rec_clean = nk.ecg_clean(rec[idx], method="pantompkins1985", sampling_rate=sampling_rate)
        signal, info =nk.ecg_process(rec_clean, sampling_rate=sampling_rate)
        bpm = np.mean(nk.ecg_rate(signal, sampling_rate))
        missing_qrs = has_missing_qrs(signal, info)
        missing_p = has_missing_p(signal, info)
        qrs_duration = np.mean(get_QRS_duration(signal, info))
        s_duration = np.mean(get_S_duration(signal, info))
        rhythm = leading_rythm(bpm)
        # rsr = has_rsR_complex(rec[idx], sampling_rate)
        notched = analyse_notched_signal(rec[idx])

        analysed_results[lead_name]={
            'signal': signal,
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
    if 'I' in leads_idx:
        if 'II' in leads_idx:
            rhythm_origin = get_rhythm_origin(analysed_results['I']['signal'], analysed_results['I']['info'], analysed_results['II']['signal'], analysed_results['II']['info'])
        if 'aVF' in leads_idx:
            if 'II' not in leads_idx:
                rhythm_origin = get_rhythm_origin(analysed_results['I']['signal'], analysed_results['I']['info'], analysed_results['aVF']['signal'], analysed_results['aVF']['info'])
            heart_axis = get_heart_axis(get_QRS_from_lead(analysed_results['I']['signal'], analysed_results['I']['info']), get_QRS_from_lead(analysed_results['aVF']['signal'], analysed_results['aVF']['info']))


    if heart_axis:
        analysed_results['heart_axis']=np.mean(heart_axis)

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
    for lead_name, idx in leads_idxs.items():
       result.append([analysis_dict[lead_name]['bpm'], analysis_dict[lead_name]['missing_qrs'],analysis_dict[lead_name]['missing_p'],analysis_dict[lead_name]['qrs_duration'], analysis_dict[lead_name]['s_duration'],analysis_dict[lead_name]['rhythm'],analysis_dict[lead_name]['notched'],analysis_dict['heart_axis'],analysis_dict['rhythm_origin_vertical'], analysis_dict['rhythm_origin_horizontal']])

    return np.array(result, dtype=np.float64)
