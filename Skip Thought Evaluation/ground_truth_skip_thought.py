import json
import pickle
import skipthoughts
import sys  

reload(sys)
sys.setdefaultencoding('utf8')
predicted_labels = 'testing_public_label.json'

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

with open(predicted_labels) as data_file:
	test_labels = json.load(data_file)

video_data = {}
model = skipthoughts.load_model()
encoder = skipthoughts.Encoder(model)
for item in test_labels:
	captions = map(lambda x: x.encode('utf-8').strip(), item['caption'])
	# print captions
	# exit()
	captions = map(lambda x: x.replace('.', ''), captions)
	captions = map(lambda x: x.replace(',', ''), captions)
	captions = map(lambda x: x.replace('"', ''), captions)
	captions = map(lambda x: x.replace('\n', ''), captions)
	captions = map(lambda x: x.replace('?', ''), captions)
	captions = map(lambda x: x.replace('!', ''), captions)
	captions = map(lambda x: x.replace('\\', ''), captions)
	captions = map(lambda x: x.replace('/', ''), captions) 
	captions = map(lambda x: x.lower(), captions)
	vec = encoder.encode(captions)
	video_data[item['id']] = (item['caption'], vec)
print len(video_data)
save_obj(video_data, "video_gt_data")
