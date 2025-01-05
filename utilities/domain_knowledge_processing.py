import pywt
import neurokit2 as nk
import numpy as np
import logging
import math
logger = logging.getLogger(__name__)



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



def get_QRS_from_lead(signal, info, with_nans=False):
    num_peaks = len(info['ECG_R_Peaks'])
    result = []
    for i in range(num_peaks):
        Q = np.nan
        if i < len(info['ECG_Q_Peaks']):
            Q = info['ECG_Q_Peaks'][i]
        R = np.nan
        if i < len(info['ECG_R_Peaks']):
            R = info['ECG_R_Peaks'][i]
        S = np.nan
        if i < len(info['ECG_S_Peaks']):
            S = info['ECG_S_Peaks'][i]
        QRS_ts= [Q, R, S]
        if with_nans:
            q_v=0
            r_v=0
            s_v=0
            if not np.isnan(Q):
                q_v = signal[Q]
            if not np.isnan(R):
                r_v = signal[R]
            if not np.isnan(S):
                s_v = signal[R]

            result.append([q_v, r_v, s_v])
        else:
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

def get_R_distances(info):
    R_peaks = info['ECG_R_Peaks']
    distances = np.diff(R_peaks)
    result = np.insert(distances, 0, R_peaks[0], axis=0)
    return result

def has_missing_p(signals, info):
    num_of_p = np.count_nonzero(~np.isnan(info['ECG_P_Peaks']))
    num_of_beats = len(info['ECG_R_Peaks'])
    return num_of_p < (num_of_beats - 1)


def get_QRS_duration(signals, info, freq=500, with_nans=False):
    if with_nans:
        num_peaks=len(info['ECG_R_Peaks'])
    else:
        num_peaks = min([len(info['ECG_R_Peaks']), len(info['ECG_Q_Peaks']), len(info['ECG_S_Peaks'])])
    result = []
    for i in range(num_peaks):
        Q = np.nan
        if i < len(info['ECG_Q_Peaks']):
            Q = info['ECG_Q_Peaks'][i]
        R = np.nan
        if i < len(info['ECG_R_Peaks']):
            R = info['ECG_R_Peaks'][i]
        S = np.nan
        if i < len(info['ECG_S_Peaks']):
            S = info['ECG_S_Peaks'][i]
        if any(np.isnan([Q,R,S])) and not with_nans:
            continue
        if any(np.isnan([Q,R,S])) and with_nans:
            result.append(0)
        else:
            result.append((S-Q)/freq)

    return result

def get_S_duration(signals, info, freq=500, with_nans=False):
    num_peaks = len(info['ECG_R_Peaks'])
    result = []
    for i in range(num_peaks):
        s = info['ECG_S_Peaks'][i]
        r = info['ECG_R_Peaks'][i]
        if any(np.isnan([r,s])) and not with_nans:
            continue
        if any(np.isnan([r,s])) and with_nans:
            result.append(0)
        else:
            result.append((s-r)/freq)
    return result


def get_R_duration(signals, info, freq=500, with_nans=False):
    num_peaks = len(info['ECG_R_Peaks'])
    result = []
    for i in range(num_peaks):
        R = info['ECG_R_Peaks'][i]
        R_on = info['ECG_R_Onsets'][i]
        R_off = info['ECG_R_Offsets'][i]
        S = info['ECG_S_Peaks'][i]
        if any(np.isnan([R,R_on, R_off, S])) and not with_nans:
            continue
        if any(np.isnan([R,R_on, R_off, S])) and with_nans:
            result.append(0)
        else:
            if S < R_off:
                result.append((S - R_on)/freq)
            else:
                result.append((R_off - R_on)/freq)

    return result

def sokolov_lyons_index(V1_info, V1_signal, V5_info, V5_signal):
    if "ECG_S_Peaks" in V1_info and "ECG_R_Peaks" in V5_info:
        V1_QRS = get_QRS_from_lead(V1_signal, V1_info, with_nans=True)
        V5_QRS = get_QRS_from_lead(V5_signal, V5_info, with_nans=True)
        default_length = len(V1_QRS)
        if len(V5_QRS) < default_length:
            default_length=len(V5_QRS)
        return [abs(V1_QRS[i][2]) + abs(V5_QRS[i][1]) for i in range(default_length)]
    else:
        default_length = len(V5_info['ECG_R_Peaks'])
        if len(V1_info['ECG_R_Peaks']) < default_length:
            default_length=len(V1_info['ECG_R_Peaks'])
        return np.zeros(default_length)

def romhilt_index(V2_info, V2_signal, V5_info, V5_signal):
    if "ECG_S_Peaks" in V2_info and "ECG_R_Peaks" in V5_info:
        V2_QRS = get_QRS_from_lead(V2_signal, V2_info, with_nans=True)
        V5_QRS = get_QRS_from_lead(V5_signal, V5_info, with_nans=True)
        default_length = len(V2_QRS)
        if len(V5_QRS) < default_length:
            default_length=len(V5_QRS)
        return [abs(V2_QRS[i][2]) + abs(V5_QRS[i][1]) for i in range(default_length)]
    else:
        default_length = len(V5_info['ECG_R_Peaks'])
        if len(V2_info['ECG_R_Peaks']) < default_length:
            default_length=len(V2_info['ECG_R_Peaks'])
        return np.zeros(default_length)



def cornells_index(V3_info, V3_signal, aVL_info, aVL_signal):
    if "ECG_S_Peaks" in V3_info and "ECG_R_Peaks" in aVL_info:
        V3_QRS = get_QRS_from_lead(V3_signal, V3_info, with_nans=True)
        aVL_QRS = get_QRS_from_lead(aVL_signal, aVL_info, with_nans=True)
        default_length = len(V3_QRS)
        if len(aVL_QRS) < default_length:
            default_length=len(aVL_QRS)
        return [abs(V3_QRS[i][2]) + abs(aVL_QRS[i][1]) for i in range(default_length)]
    else:
        default_length = len(aVL_info['ECG_R_Peaks'])
        if len(V3_info['ECG_R_Peaks']) < default_length:
            default_length=len(V3_info['ECG_R_Peaks'])
        return np.zeros(default_length)

def cornells_product(V3_info, V3_signal, aVL_info, aVL_signal):
    voltages=cornells_index(V3_info, V3_signal, aVL_info, aVL_signal)
    durations=get_QRS_duration(aVL_signal, aVL_info, with_nans=True)
    default_length = len(voltages)
    if len(durations) < default_length:
        default_length=len(durations)
    product = [voltages[i] * durations[i] for i in range(default_length)]
    return product



def lewis_index(III_info, III_signal, I_info, I_signal):
    if "ECG_S_Peaks" in III_info and "ECG_R_Peaks" in I_info:
        III_QRS = get_QRS_from_lead(III_signal, III_info, with_nans=True)
        I_QRS = get_QRS_from_lead(I_signal, I_info, with_nans=True)
        default_length = len(III_QRS)
        if len(I_QRS) < default_length:
            default_length=len(I_QRS)
        return [(III_QRS[i][2] - I_QRS[i][2]) + (I_QRS[i][1]-III_QRS[i][1]) for i in range(default_length)]
    else:
        default_length = len(I_info['ECG_R_Peaks'])
        if len(III_info['ECG_R_Peaks']) < default_length:
            default_length=len(III_info['ECG_R_Peaks'])
        return np.zeros(default_length)

def get_max_param_from_lead(signal, info, param):
    qrs = get_QRS_from_lead(signal, info, with_nans=True)
    if param == "q":
        return abs(max(qrs[:][0], key=abs))
    if param == "r":
        return abs(max(qrs[:][1], key=abs))
    if param == "s":
        return abs(max(qrs[:][2], key=abs))



def mcphie_index(signals, infos):
    max_r = -1000000000
    max_s = -1000000000
    for signal, info in zip(signals, infos):
        tmp_r = get_max_param_from_lead(signal, info, "r")
        tmp_s = get_max_param_from_lead(signal, info, "s")
        if tmp_r > max_r:
            max_r = tmp_r
        if tmp_s > max_s:
            max_s = tmp_s

    result = max_r + max_s

    if result < 0:
        result=0

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




def analyse_notched_signal(signal, info, recording, peaks, threshold=1.5,  **kwargs):
    #list_of_qrs = get_QRS_from_lead(signal, info) #get_qrs_beginning_and_end(signal['ECG_Raw'], **kwargs)
    N = len(recording)
    window = None

    list_of_qrs = []
    window = 150
    for peak in peaks:
        if peak < (window // 2) + 1:
            list_of_qrs.append([0, window])
        elif peak > ( N - window//2):
            list_of_qrs.append([N-window-1,N-1])
        else:
            list_of_qrs.append([(peak-window//2), (peak+window//2)])

   # list_of_qrs = [[info[beginning_key][i], info[beginning_key][i]] for i in len(info['ECG_R_Peaks'])]
    logger.debug(f"List of qrs for zero crossings: {list_of_qrs}")
    if len(list_of_qrs) ==0:
        return -1

    list_of_qrs = np.array(list_of_qrs)
    logger.debug(f"Shape of qrs for zero crossings: {list_of_qrs.shape}")

    beg_qrs = list_of_qrs[:, 0]
    end_qrs = list_of_qrs[:, 1]
    (_, cD) = pywt.dwt(recording, 'bior1.1')
    crossing_0 = get_0_crossings(cD, beg_qrs, end_qrs, **kwargs)
    if len(crossing_0) > 0:
        return crossing_0
    else:
        return -1






def cleanse_data_mean(array):
    if len(array) > 0 and array is not None:
        result = np.nan_to_num(array, posinf=99999, neginf=-99999)
        return  np.mean(result)
    else:
        return -1


def check_for_lead(lead_name, leads_idxs, analysed_results) -> bool:
    return lead_name in leads_idxs and analysed_results[lead_name]['info'] is not None


def analyse_recording(rec, signals, infos, rates, leads_idxs,  pantompkins_peaks=None, label=None,  sampling_rate=500):
    logger.debug("Entering analysed_results")
    analysed_results = {}
    precordial_leads=[]
    for lead_name, idx in leads_idxs.items():
        signal = signals[lead_name]
        info = infos[lead_name] #nk.ecg_process(rec[idx], sampling_rate=sampling_rate)
        if signal is None or info is None:
            analysed_results[lead_name]={
                'signal': rec[idx],
                'info': None,
                'bpm': 0,
                'missing_qrs': -1,
                'missing_p': -1,
                'qrs_duration': 0,
                's_duration': 0,
                'notched': 0,
                'r_distances': 0,
                'romhilt':0,
                'conrell':0,
                'lewis':0,
                'cornell-product':0,
                'mcphie':0,
                'sokolov-lyon':0,
                'heart_axis':0,
                'rhythm_origin_vertical':0,
                'rhythm_origin_horizontal':0,
            }
            continue

        logger.debug(f"Peaks from pantompkings: {pantompkins_peaks}, Peaks from neurokit2: {len(info['ECG_R_Peaks'])}")

        if "V" in lead_name:
            precordial_leads.append(lead_name)
        bpm = -1
        if lead_name in rates:
            bpm = cleanse_data_mean(rates[lead_name])
        missing_qrs = has_missing_qrs(signal, info)
        missing_p = has_missing_p(signal, info)

        qrs_duration = get_QRS_duration(signal, info, with_nans=True)
        s_duration = get_S_duration(signal, info, with_nans=True)
        notched = analyse_notched_signal(signal,info, rec[idx], peaks=pantompkins_peaks)
        r_distances = get_R_distances(info).tolist()

        analysed_results[lead_name]={
            'signal': rec[idx],
            'info': info,
            'bpm': bpm,
            'missing_qrs':missing_qrs,
            'missing_p': missing_p,
            'qrs_duration':qrs_duration,
            's_duration':s_duration,
            'notched': notched,
            'r_distances': r_distances,
        }

    #CrossLead indicators

    heart_axis = None
    rhythm_origin = None
    if check_for_lead('I',leads_idxs,analysed_results):
        if check_for_lead('II',leads_idxs,analysed_results):
            rhythm_origin = get_rhythm_origin(analysed_results['I']['signal'], analysed_results['I']['info'], analysed_results['II']['signal'], analysed_results['II']['info'])
        if check_for_lead('aVF',leads_idxs,analysed_results):
            if 'II' not in leads_idxs:
                rhythm_origin = get_rhythm_origin(analysed_results['I']['signal'], analysed_results['I']['info'], analysed_results['aVF']['signal'], analysed_results['aVF']['info'])
            heart_axis = get_heart_axis(get_QRS_from_lead(analysed_results['I']['signal'], analysed_results['I']['info'], with_nans=True), get_QRS_from_lead(analysed_results['aVF']['signal'], analysed_results['aVF']['info'], with_nans=True))



    #Lewis Index
    if check_for_lead('I', leads_idxs, analysed_results) and check_for_lead('III', leads_idxs, analysed_results):
        analysed_results['lewis']=lewis_index(analysed_results['III']['info'], analysed_results['III']['signal'], analysed_results['I']['info'], analysed_results['I']['signal'])
    else:
        analysed_results['lewis']=0


    #McPhie
    precordial_signals=[analysed_results[l]['signal'] for l in precordial_leads]
    precordial_infos=[analysed_results[l]['info'] for l in precordial_leads]
    if len(precordial_leads) > 0:
        analysed_results['mcphie']=mcphie_index(precordial_signals, precordial_infos)
    else:
        analysed_results['mcphie']=0

    #Sokolov-Lyon
    if check_for_lead('V1', leads_idxs, analysed_results) and check_for_lead('V5', leads_idxs, analysed_results):
        analysed_results['sokolov-lyon']=sokolov_lyons_index(analysed_results['V1']['info'], analysed_results['V1']['signal'], analysed_results['V5']['info'], analysed_results['V5']['signal'])
    else:
        analysed_results['sokolov-lyon']=0

    #Cornell
    if check_for_lead('V3', leads_idxs, analysed_results) and check_for_lead('aVL', leads_idxs, analysed_results):
        analysed_results['cornell']=cornells_index(analysed_results['V3']['info'], analysed_results['V3']['signal'], analysed_results['aVL']['info'], analysed_results['aVL']['signal'])
        analysed_results['cornell-product']=cornells_product(analysed_results['V3']['info'], analysed_results['V3']['signal'], analysed_results['aVL']['info'], analysed_results['aVL']['signal'])
    else:
        analysed_results['cornell']=0
        analysed_results['cornell-product']=0

    #Romhilt
    if check_for_lead('V2', leads_idxs, analysed_results) and check_for_lead('V5', leads_idxs, analysed_results):
        analysed_results['romhilt']=sokolov_lyons_index(analysed_results['V2']['info'], analysed_results['V2']['signal'], analysed_results['V5']['info'], analysed_results['V5']['signal'])
    else:
        analysed_results['romhilt']=0


    if heart_axis:
        analysed_results['heart_axis']=heart_axis

    if rhythm_origin:
        analysed_results['rhythm_origin_vertical']=rhythm_origin[0]
        analysed_results['rhythm_origin_horizontal']=rhythm_origin[1]

    analysed_results['label'] = label

    for lead_name, idx in leads_idxs.items():
        analysed_results[lead_name].pop('signal', None)
        analysed_results[lead_name].pop('info', None)


    return analysed_results

def analysis_dict_to_array(analysis_dict, leads_idxs, peaks_count):
    result = []
    logger.debug(analysis_dict)
    per_lead_parameters = ['bpm', 'missing_qrs', 'missing_p', 'qrs_duration', 's_duration', 'notched', 'r_distances']
    if len(leads_idxs.keys()) == 12:
        cross_lead_parameters = ['heart_axis','rhythm_origin_vertical','rhythm_origin_horizontal', 'romhilt', 'cornell', 'cornell-product', 'sokolov-lyon', 'mcphie', 'lewis']
    elif len(leads_idxs.keys()) == 6:
        cross_lead_parameters = ['heart_axis','rhythm_origin_vertical','rhythm_origin_horizontal', 'lewis']
    elif len(leads_idxs.keys()) == 4:
        cross_lead_parameters = ['heart_axis','rhythm_origin_vertical','rhythm_origin_horizontal']
    else:
        cross_lead_parameters = ['rhythm_origin_vertical','rhythm_origin_horizontal']


    for peak_idx in range(peaks_count):
        tmp_result_lead = []
        for lead_name, idx in leads_idxs.items():
            tmp_result = []
            for key in per_lead_parameters:
                if key in analysis_dict[lead_name]:
                    try:
                        if type(analysis_dict[lead_name][key]) == list:
                            tmp_result.append(analysis_dict[lead_name][key][peak_idx])
                        else:
                            tmp_result.append(analysis_dict[lead_name][key])
                    except Exception as e:
                        logger.error(f"Key {key}, peak_idx {peak_idx}, lead {lead_name}  from result: {analysis_dict[lead_name][key]}")
                        raise e
                else:
                    raise Exception(f"No key {key} in results dict")
            for key in cross_lead_parameters:
                if key in analysis_dict:
                    try:
                        if type(analysis_dict[key]) == list:
                            tmp_result.append(analysis_dict[key][peak_idx])
                        else:
                            tmp_result.append(analysis_dict[key])
                    except Exception as e:
                        logger.error(f"Key {key}, peak_idx {peak_idx}, array from result: {analysis_dict[key]}")
                        raise e
                else:
                    logger.warn(f"No key {key} in results dict")
                    tmp_result.append(0)

            tmp_result_lead.append(tmp_result)
        result.append(tmp_result_lead)
    return np.array(result, dtype=np.float64)

