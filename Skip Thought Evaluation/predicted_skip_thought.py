import json
import pickle
import skipthoughts
predicted_labels = 'S2VT_prediction.json'

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

with open(predicted_labels) as data_file:
	test_labels = json.load(data_file)

# print len(test_labels)
video_data = {}
model = skipthoughts.load_model()
encoder = skipthoughts.Encoder(model)
for item in test_labels:
	caption = item['caption'].replace('<unk>', 'something')
	caption = caption.replace('.', '')
	caption = caption.lower()
	vec = encoder.encode([caption])[0]
	print len(vec)
	video_data[item['id']] = (item['caption'], vec)
print len(video_data)
save_obj(video_data, "video_pred_data")
