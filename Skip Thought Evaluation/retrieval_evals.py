import pickle
import numpy as np
import os
from scipy import spatial
def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

video_data = load_obj("video_pred_data")
video_gt_data = load_obj("video_gt_data")
def apk(actual, predicted, k=10):
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

	return score / min(len(actual), k)

def retrieve_pred_topK(vec, k=5):
	videos = []
	vec = np.array(vec)
	for item in video_data:
		videos.append((item, spatial.distance.cosine(np.array(video_data[item][1]), vec)))
	videos.sort(key=lambda x:x[1])
	return np.array(videos)[:k, 0]

def retrieve_gt_topK(vec, k=5):
	videos = []
	vec = np.array(vec)
	for item in video_gt_data:
		val = [0.0, 0.0]
		for cap in video_gt_data[item][1]:
			val[1] += np.linalg.norm(np.array(cap) - vec)
			val[0] += 1
		val[1] /= val[0]
		videos.append((item, val[1]))
	videos.sort(key=lambda x:x[1])
	return np.array(videos)[:k, 0]

def precision(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    result = float(len(act_set & pred_set)) / float(k)
    return result

def recall(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    result = float(len(act_set & pred_set)) / float(len(act_set))
    return result

mAP = 0.0
pat1 = 0.0
rat1 = 0.0
pat10 = 0.0
rat10 = 0.0
count = 0.0
for item in video_gt_data:
	for cap in video_gt_data[item][1]:
		list_pred = retrieve_pred_topK(cap, 10)
		# print list_pred
		list_gt = retrieve_gt_topK(cap, 40)
		list_pred = map(lambda x:os.path.splitext(x)[0], list_pred)
		pat1 += precision(list_gt, list_pred, 1)
		pat10 += precision(list_gt, list_pred, 10)
		rat1 += recall(list_gt, list_pred, 1)
		rat10 += recall(list_gt, list_pred, 10)
		mAP += apk(list_gt, list_pred, 10)
		count += 1
	print count
pat1 /= float(count)
pat10 /= float(count)
rat1 /= float(count)
rat10 /= float(count)
mAP /= float(count)
with open("final_result_unk_some_cosine.txt", 'w') as f:
	f.write("Precision@1: " + str(pat1) + "\n")
	f.write("Precision@10: " + str(pat10) + "\n")
	f.write("Recall@1: " + str(rat1) + "\n")
	f.write("Recall@10: " + str(rat10) + "\n")
	f.write("meanAvgPrec@10: " + str(mAP) + "\n")


