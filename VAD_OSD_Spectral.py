import os
from pathlib import Path
from diarizer_master.diarizer.xvector import predict
from diarizer_master.diarizer.spectral import sclust
from diarizer_master.diarizer.overlap import pyannote_overlap
from os import walk
import os, sys, time
from pathlib import Path
import torch
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.core import Segment, Timeline, Annotation
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.audio import Model
from pyannote.metrics.diarization import DiarizationPurity, DiarizationCoverage


HYPER_PARAMETERS = {
        "onset": 0.5,
        "offset": 0.5,
        "min_duration_on": 0.0,
        "min_duration_off": 0.0,
    }

denoiser_mode = False
overlap_enable = False
# data_type = "test"
data_type = "dev"
# base = "Aish"
base = "Desraj"
# datasetname = "ami" #/"vox"
# datasetname = "vox" #/"ami"
datasetname = "displace2024" #/"ami"

denoiser_mode = False
overlap_enable = False
data_type = "dev"
base = "Aish"
datasetname = "vox" 

if datasetname == "vox":
    if denoiser_mode:
        EXP_DIR = r"A:\myworkspace"
        if data_type == "test":
            raw_in_wav_dir = os.path.join(EXP_DIR, "voxconverse_test_wav")
            in_wav_dir = os.path.join(EXP_DIR, "denoised_test")
            out_dir_OL = os.path.join(EXP_DIR, "vox_test_ol_denoised")
            vad_out_dir = os.path.join(EXP_DIR, "vox_test_vad_denoised")
            out_xvec_dir = os.path.join(EXP_DIR, "vox_test_xvec_denoised")
            if base == "Aish":
                print("Configs set for Test set and Aishwarya with denoiser enabled")
                out_rttm_dir = r"A:\myworkspace\vox_test_rttm_aish_OL_denoised"
            else:
                print("Configs set for Test set and Desraj with denoiser enabled")
                out_rttm_dir = r"A:\myworkspace\vox_test_rttm_desraj_OL_denoised"
        elif data_type == "dev":
            raw_in_wav_dir = os.path.join(EXP_DIR, "voxconverse_dev_wav")
            in_wav_dir = os.path.join(EXP_DIR, "denoised")
            out_dir_OL = os.path.join(EXP_DIR, "vox_dev_ol_denoised")
            vad_out_dir = os.path.join(EXP_DIR, "vox_dev_vad_denoised")
            out_xvec_dir = os.path.join(EXP_DIR, "vox_dev_xvec_denoised")
            if base == "Aish":
                print("Configs set for Dev set and Aish with denoiser enabled")
                out_rttm_dir = r"A:\myworkspace\vox_dev_rttm_aish_OL_denoised"
            else:
                print("Configs set for Dev set and Desraj with denoiser enabled")
                out_rttm_dir = r"A:\myworkspace\vox_dev_rttm_desraj_OL_denoised"

        in_lab_dir = Path(vad_out_dir)

    else:
        EXP_DIR = r"C:\Users\lenovo\PycharmProjects\pythonProject"
        EXP_DIR = r"A:\myworkspace"
        if overlap_enable:
            EXP_DIR_v1=r"C:\Users\lenovo\PycharmProjects\pythonProject"
            EXP_DIR = r"A:\myworkspace"
            if data_type == "test":
                out_dir_OL = os.path.join(EXP_DIR_v1, "vox_test_ol")
                raw_in_wav_dir = os.path.join(EXP_DIR, "voxconverse_test_wav")
                in_wav_dir = os.path.join(EXP_DIR, "voxconverse_test_wav")
                vad_out_dir = in_lab_dir = os.path.join(EXP_DIR_v1, "vox_test_vad")
                out_xvec_dir = os.path.join(EXP_DIR_v1, "vox_test_xvec")
                if base == "Aish":
                    print("Configs set for Test set and Aishwarya with denoiser disabled")
                    out_rttm_dir = r"C:\Users\lenovo\PycharmProjects\pythonProject\vox_test_rttm_aish_OL"
                else:
                    print("Configs set for Test set and Desraj with denoiser disabled")
                    out_rttm_dir = r"C:\Users\lenovo\PycharmProjects\pythonProject\vox_test_rttm_desraj_OL"
            elif data_type == "dev":
                out_dir_OL = os.path.join(EXP_DIR_v1, "vox_dev_ol")
                raw_in_wav_dir = os.path.join(EXP_DIR, "voxconverse_dev_wav")
                in_wav_dir = os.path.join(EXP_DIR, "voxconverse_dev_wav")
                vad_out_dir = in_lab_dir = os.path.join(EXP_DIR_v1, "vox_dev_vad")
                out_xvec_dir = os.path.join(EXP_DIR_v1, "vox_dev_xvec")
                if base == "Aish":
                    print("Configs set for Dev set and Aish with denoiser disabled")
                    out_rttm_dir = r"C:\Users\lenovo\PycharmProjects\pythonProject\vox_dev_rttm_aish_OL"
                else:
                    print("Configs set for Dev set and Desraj with denoiser disabled")
                    out_rttm_dir = r"C:\Users\lenovo\PycharmProjects\pythonProject\vox_dev_rttm_desraj_OL"

        else:
            EXP_DIR = out_dir_OL = r"C:\Users\lenovo\PycharmProjects\pythonProject"
            if data_type == "test":
                in_wav_dir = r"A:\myworkspace\voxconverse_test_wav"
                vad_out_dir = in_lab_dir = os.path.join(EXP_DIR, "vox_test_vad")
                vad_out_dir = in_lab_dir = Path(vad_out_dir)
                out_xvec_dir = os.path.join(EXP_DIR, "vox_test_xvec")
                if base == 'Aish':
                    out_rttm_dir = r"C:\Users\lenovo\PycharmProjects\pythonProject\vox_test_rttm_aish"
                else:
                    out_rttm_dir = r"C:\Users\lenovo\PycharmProjects\pythonProject\vox_test_rttm_desraj"
            else:
                in_wav_dir = r"A:\myworkspace\voxconverse_dev_wav"
                vad_out_dir = in_lab_dir = os.path.join(EXP_DIR, "vox_dev_vad")
                out_xvec_dir = os.path.join(EXP_DIR, "vox_dev_xvec")
                if base == 'Aish':
                    out_rttm_dir = r"C:\Users\lenovo\PycharmProjects\pythonProject\vox_dev_rttm_aish"
                else:
                    out_rttm_dir = r"C:\Users\lenovo\PycharmProjects\pythonProject\vox_dev_rttm_desraj"
    print("VOX Variables: --------->")
    print("in_wav_dir: ", in_wav_dir)
    print("vad_out_dir: ", vad_out_dir)
    print("in_lab_dir: ", in_lab_dir)
    print("out_xvec_dir: ", out_xvec_dir)
    print("out_rttm_dir: ", out_rttm_dir)


elif datasetname == "ami":
    EXP_DIR = r"C:\Users\lenovo\PycharmProjects\pythonProject"
    if overlap_enable:
        out_dir_OL = os.path.join(EXP_DIR, "%s_ol"%datasetname)
        raw_in_wav_dir = os.path.join(EXP_DIR, "%s_mixed_set_wav"%datasetname)
        in_wav_dir = os.path.join(EXP_DIR, "%s_mixed_set_wav"%datasetname)
        vad_out_dir = in_lab_dir = os.path.join(EXP_DIR, "%s_vad_cuda"%datasetname)
        out_xvec_dir = os.path.join(EXP_DIR, "%s_xvec_cuda"%datasetname)
        if base == "Aish":
            print("Configs set for AMI mixed headset and Aishwarya with denoiser disabled")
            out_rttm_dir = r"C:\Users\lenovo\PycharmProjects\pythonProject\%s_rttm_aish_OL"%datasetname
        else:
            print("Configs set for AMI mixed headset and Desraj with denoiser disabled")
            out_rttm_dir = r"C:\Users\lenovo\PycharmProjects\pythonProject\%s_rttm_desraj_OL"%datasetname
    else:
        out_dir_OL = os.path.join(EXP_DIR, "%s_ol" % datasetname)
        raw_in_wav_dir = os.path.join(EXP_DIR, "%s_mixed_set_wav" % datasetname)
        in_wav_dir = os.path.join(EXP_DIR, "%s_mixed_set_wav" % datasetname)
        vad_out_dir = in_lab_dir = os.path.join(EXP_DIR, "%s_vad_cuda" % datasetname)
        out_xvec_dir = os.path.join(EXP_DIR, "%s_xvec_cuda" % datasetname)
        if base == "Aish":
            print("Configs set for AMI mixed headset and Aishwarya with denoiser disabled")
            out_rttm_dir = r"A:\myworkspace\%s_spectral_testset_cuda" % datasetname  #QUANG WANG PATH
            out_rttm_dir = r"C:\Users\lenovo\PycharmProjects\pythonProject\%s_rttm_aish" % datasetname
        else:
            print("Configs set for AMI mixed headset and Desraj with denoiser disabled")
            out_rttm_dir = r"C:\Users\lenovo\PycharmProjects\pythonProject\%s_rttm_desraj" % datasetname
    print("AMI Variables: --------->")
    print("out_dir_OL: ", out_dir_OL)
    print("raw_in_wav_dir: ", raw_in_wav_dir)
    print("in_wav_dir: ", in_wav_dir)
    print("vad_out_dir: ", vad_out_dir)
    print("in_lab_dir: ", in_lab_dir)
    print("out_xvec_dir: ", out_xvec_dir)
    print("out_rttm_dir: ", out_rttm_dir)
elif datasetname == "displace2024":
    EXP_DIR = r"C:\Users\lenovo\PycharmProjects\pythonProject"
    if overlap_enable:
        out_dir_OL = os.path.join(EXP_DIR, "%s_ol"%datasetname)
        raw_in_wav_dir = os.path.join(EXP_DIR, "%s_wav"%datasetname)
        in_wav_dir = os.path.join(EXP_DIR, "%s_wav"%datasetname)
        vad_out_dir = in_lab_dir = os.path.join(EXP_DIR, "%s_vad_cuda"%datasetname)
        out_xvec_dir = os.path.join(EXP_DIR, "%s_xvec_cuda"%datasetname)
        if base == "Aish":
            print("Configs set for displace2024 mixed headset and Aishwarya with denoiser disabled")
            out_rttm_dir = r"C:\Users\lenovo\PycharmProjects\pythonProject\%s_rttm_aish_OL"%datasetname
        else:
            print("Configs set for displace2024 mixed headset and Desraj with denoiser disabled")
            out_rttm_dir = r"C:\Users\lenovo\PycharmProjects\pythonProject\%s_rttm_desraj_OL"%datasetname
    else:
        out_dir_OL = os.path.join(EXP_DIR, "%s_ol" % datasetname)
        raw_in_wav_dir = os.path.join(EXP_DIR, "%s_wav" % datasetname)
        in_wav_dir = os.path.join(EXP_DIR, "%s_wav" % datasetname)
        vad_out_dir = in_lab_dir = os.path.join(EXP_DIR, "%s_vad_cuda" % datasetname)
        out_xvec_dir = os.path.join(EXP_DIR, "%s_xvec_cuda" % datasetname)
        if base == "Aish":
            print("Configs set for displace2024 mixed headset and Aishwarya with denoiser disabled")
            out_rttm_dir = r"C:\Users\lenovo\PycharmProjects\pythonProject\%s_rttm_aish" % datasetname
        else:
            print("Configs set for displace2024 mixed headset and Desraj with denoiser disabled")
            out_rttm_dir = r"C:\Users\lenovo\PycharmProjects\pythonProject\%s_rttm_desraj" % datasetname
    print("displace2024 Variables: --------->")
    print("out_dir_OL: ", out_dir_OL)
    print("raw_in_wav_dir: ", raw_in_wav_dir)
    print("in_wav_dir: ", in_wav_dir)
    print("vad_out_dir: ", vad_out_dir)
    print("in_lab_dir: ", in_lab_dir)
    print("out_xvec_dir: ", out_xvec_dir)
    print("out_rttm_dir: ", out_rttm_dir)

xvec_transform = "diarizer_master/diarizer/models/ResNet101_16kHz/transform.h5"
time.sleep(10)
# Creating directories
# print(type(vad_out_dir), vad_out_dir, type(out_xvec_dir), out_xvec_dir)
os.makedirs(vad_out_dir, exist_ok=True)
os.makedirs(out_xvec_dir, exist_ok=True)
os.makedirs(out_dir_OL, exist_ok=True)
os.makedirs(out_rttm_dir, exist_ok=True)
# vad_out_dir.mkdir(exist_ok=True, parents=True)
# Path(out_xvec_dir).mkdir(exist_ok=True, parents=True)
# Path(out_dir_OL).mkdir(exist_ok=True, parents=True)
# Path(out_rttm_dir).mkdir(exist_ok=True, parents=True)


# extracting filenames in the list
_, _, filenames_ = next(walk(in_wav_dir))

files = [file.replace(".wav", "") for file in filenames_ ]
# files = [file.replace(".wav", "") for file in filenames_ if "hqyok" in file]
files = [file.replace(".wav", "") for file in filenames_ if "rtvuw" in file]

print(files)


in_dir2 = Path(in_wav_dir)

## VAD file creation
def vad_main(in_dir, files, out_dir, HYPER_PARAMETERS):
    # out_dir.mkdir(exist_ok=True, parents=True)
    os.makedirs(out_dir, exist_ok=True)

    model = Model.from_pretrained("pyannote/segmentation",
                                  use_auth_token="<secrets>")
    vad_pipeline = VoiceActivityDetection(segmentation=model, device="cuda")
    vad_pipeline.instantiate(HYPER_PARAMETERS)

    for file in in_dir.rglob("*.wav"):
        file_id = file.stem
        if file_id not in files:
            continue
        vad_out = vad_pipeline({"audio": file})
        align_time = None
        with open(f"{out_dir}/{file_id}.lab", "w") as f:
            for start, end in vad_out.get_timeline():
                if align_time is not None:
                    start = round(start / align_time) * align_time
                    end = round(end / align_time) * align_time
                f.write(f"{start:.3f} {end:.3f} sp\n")

# vad_main(Path(in_wav_dir), files,vad_out_dir, HYPER_PARAMETERS)
# if overlap_enable:
#     pyannote_overlap.main(Path(raw_in_wav_dir), files, Path(out_dir_OL))


for elem in files:
    filn = os.path.join(EXP_DIR, "list", "list_{}.txt".format(elem))
    os.environ['FILE_NAME'] = elem
    print(filn)
    f = open(filn, "w+")
    f.write(elem)
    f.close()
    # xvectors creation, Creation of seg and ark file
    out_ark_fn = os.path.join(out_xvec_dir, "{}.ark".format(elem))
    out_seg_fn = os.path.join(out_xvec_dir, "{}.seg".format(elem))
    if os.path.exists(out_ark_fn) and os.path.exists(out_seg_fn) and \
            os.path.getsize(out_ark_fn)>0 and os.path.getsize(out_seg_fn) >0:
        print("Skip generation of ark and seg file...%s"%elem)
    else:
        predict.prediction(filn , out_ark_fn, out_seg_fn, in_wav_dir, in_lab_dir)

    if datasetname == "vox":
        if overlap_enable:
            if denoiser_mode:
                # using overlap rttm generated from main file
                if data_type != "test":
                    out_dir_OL_tmp = os.path.join(r"C:\Users\lenovo\PycharmProjects\pythonProject", "vox_dev_ol")
                else:
                    out_dir_OL_tmp = out_dir_OL
                sclust.sclustering(out_seg_fn, out_ark_fn, xvec_transform, out_rttm_dir=out_rttm_dir,
                                   overlap_rttm=os.path.join(out_dir_OL_tmp, elem + ".rttm"), base = base, data_set_type=data_type)
            else:
                sclust.sclustering(out_seg_fn, out_ark_fn, xvec_transform, out_rttm_dir=out_rttm_dir,
                                   overlap_rttm=os.path.join(out_dir_OL, elem + ".rttm"),base = base, data_set_type=data_type)
        else:
            sclust.sclustering(out_seg_fn, out_ark_fn, xvec_transform,
                               out_rttm_dir = out_rttm_dir, data_set_type=data_type, base=base, datasetname=datasetname)
    elif datasetname == "ami" or datasetname == "displace2024":
        print("Inside %s block"%datasetname)
        if overlap_enable :
            sclust.sclustering(out_seg_fn, out_ark_fn, xvec_transform, out_rttm_dir=out_rttm_dir,
                               overlap_rttm=os.path.join(out_dir_OL, elem + ".rttm"), base=base,
                               data_set_type=data_type, datasetname=datasetname)
        else:
            sclust.sclustering(out_seg_fn, out_ark_fn, xvec_transform, out_rttm_dir=out_rttm_dir,
                               base=base, data_set_type=data_type, datasetname=datasetname)


sys.exit()
#DER Calculation
# to remove below line
#out_rttm_dir = r"C:\Users\lenovo\PycharmProjects\pythonProject\vox_dev_rttm_desraj_OL"
hyp_dict = {}
ground_truth_rttm_path = ""
if datasetname == "vox":
    if data_type == "test":
        ground_truth_rttm_path = r"C:\Users\lenovo\AppData\Roaming\JetBrains\PyCharmCE2020.2\scratches\rttm_git_repo\test"
    else:
        ground_truth_rttm_path = r'C:\Users\lenovo\AppData\Roaming\JetBrains\PyCharmCE2020.2\scratches\rttm_git_repo\dev'
elif datasetname == "ami":
    ground_truth_rttm_path = r"C:\Users\lenovo\PycharmProjects\pythonProject\ami_rttm"
elif datasetname == "displace2024":
    ground_truth_rttm_path = r"C:\Users\lenovo\PycharmProjects\pythonProject\displace2024_groundtruth_rttm"
_, _, filenames_ = next(walk(out_rttm_dir))
# filenames_=['eqttu.rttm']
ref_dict = {}
for rttmfile in filenames_:
    rttmfile1 = rttmfile
    if datasetname=="displace2024":
        rttmfile1=rttmfile.split(".")[0]+"_SPEAKER.rttm"
    with open(ground_truth_rttm_path+"\\"+rttmfile1, 'r') as fh:
        lines = fh.readlines()
        for line_str in lines:
            line_list = line_str.split()
            wavfilename = rttmfile.replace('rttm', 'wav')
            if wavfilename not in ref_dict.keys():
                ref_dict[wavfilename] = [(line_list[7],float(line_list[3]), float(line_list[3])+float(line_list[4]))]
            else:
                ref_dict[wavfilename].append((line_list[7],float(line_list[3]), float(line_list[3])+float(line_list[4])))
    with open(out_rttm_dir+"\\"+rttmfile, 'r') as fh:
        lines = fh.readlines()
        for line_str in lines:
            line_list = line_str.split()
            wavfilename = rttmfile.replace('rttm', 'wav')
            if wavfilename not in hyp_dict.keys():
                hyp_dict[wavfilename] = [(line_list[7],float(line_list[3]), float(line_list[3])+float(line_list[4]))]
            else:
                hyp_dict[wavfilename].append((line_list[7],float(line_list[3]), float(line_list[3])+float(line_list[4])))
res = {}

for k,v in hyp_dict.items():
    try:
        metric = DiarizationErrorRate()
        reference = Annotation(uri=k)
        hypothesis = Annotation(uri=k)
        for elem in v:
            hypothesis[Segment(elem[1], elem[2])] = elem[0]
        for elem in ref_dict[k]:
            reference[Segment(elem[1], elem[2])] = elem[0]
        error_dict = metric(reference, hypothesis, detailed=True)
        purity_score = DiarizationPurity()(reference, hypothesis)
        coverage_score = DiarizationCoverage()(reference, hypothesis)
        error_dict['purity'] = purity_score
        error_dict['coverage'] = coverage_score
        # error_dict['diarization error rate'] = error_dict['false alarm']+ error_dict['missed detection'] + error_dict['confusion']
        # print("filename: {} DER:{} {} {} {}".format(k, error_dict['diarization error rate'],
        #                                                   error_dict['false alarm'], error_dict['missed detection'],
        #                                                   error_dict['confusion']))
        print("filename: {} DER:{} and total: {} ".format(k, error_dict['diarization error rate'], error_dict['total']))
        res[k]=error_dict
        # print(error_dict)
    except Exception as e:
        print("ERROR:",e)

x = list(res.values())
# print("X: ",[i['diarization error rate'] for i in x])
# print("X: ",[i for i in x])


sum = [0.0]*8
for elem in x:
    sum[0] += elem['missed detection']
    sum[1] += elem['false alarm']
    sum[2] += elem['correct']
    sum[3] += elem['confusion']
    sum[4] += elem['total']
    sum[5] += elem['diarization error rate']
    sum[6] += elem['purity']
    sum[7] += elem['coverage']
# print("AVG missed detection={:.3f}".format(sum[0]/len(x)))
# print("AVG false alarm={:.3f}".format(sum[1]/len(x)))
# print("AVG correct={:.3f}".format(sum[2]/len(x)))
# print("AVG confusion={:.3f}".format(sum[3]/len(x)))
# print("AVG total={:.3f}".format(sum[4]/len(x)))
# print("AVG DER={:.3f}".format(sum[5]/len(x)))
print("sum of all files MD ", sum[0])
print("sum of all files FA ", sum[1])
print("sum of all files CONFUSION ", sum[3])
print("sum of all files TOTAL ", sum[4])
print("Overall DER={:.3f}".format((sum[0]+sum[1]+sum[3])/sum[4]))
print("AVG purity={:.3f}".format(sum[6]/len(x)))
print("AVG Coverage={:.3f}".format(sum[7]/len(x)))




#To find the number of speakers in rttm file
# mypath = r'C:\Users\lenovo\AppData\Roaming\JetBrains\PyCharmCE2020.2\scratches\rttm_git_repo\dev'
# mypath = r'C:\Users\lenovo\AppData\Roaming\JetBrains\PyCharmCE2020.2\scratches\rttm_git_repo\test'
# out_rttm_dir = ground_truth_rttm_path = r"C:\Users\lenovo\PycharmProjects\pythonProject\ami_rttm"
# _, _, filenames_ = next(walk(out_rttm_dir))
# filename = {}
# for rttmfile in filenames_:
#     with open(out_rttm_dir+"\\"+rttmfile, 'r') as fh:
#         lines = fh.readlines()
#         spk = set()
#         for line_str in lines:
#             line_list = line_str.split()
#             spk.add(line_list[7])
#         filename[rttmfile.replace('.rttm', '')] = len(spk)
#         # print(rttmfile.replace('.rttm', ''), len(spk))
#         print( len(spk))
# print(filename, end='\n')