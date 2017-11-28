import pickle
import skipthoughts
import numpy as np
def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

video_data = load_obj("video_pred_data")
model = skipthoughts.load_model()
encoder = skipthoughts.Encoder(model)

def retrieve_topK(vec, k=5):
	videos = []
	vec = np.array(vec)
	for item in video_data:
		videos.append((item, np.linalg.norm(np.array(video_data[item][1]) - vec)))
	videos.sort(key=lambda x:x[1])
	return videos[:k]
while True:
	input_query = raw_input("Enter a query to search:")
	vec = encoder.encode([input_query])[0]
	result = retrieve_topK(vec)
	print result

