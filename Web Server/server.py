import socket
import pickle
import skipthoughts
import numpy as np
from scipy import spatial

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

video_data = load_obj("video_data")
model = skipthoughts.load_model()
encoder = skipthoughts.Encoder(model)

def retrieve_topK(vec, k=10):
	videos = []
	vec = np.array(vec)
	for item in video_data:
		videos.append((item, spatial.distance.cosine(np.array(video_data[item][1]), vec)))
        #videos.append((item, np.linalg.norm(np.array(video_data[item][1]) - vec)))
	videos.sort(key=lambda x:x[1])
	return np.array(videos)[:k, 0]

host = ''        # Symbolic name meaning all available interfaces
port = 12345     # Arbitrary non-privileged port
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((host, port))

print host , port
while True:

    try:
    	print "While loop"
    	s.listen(1)
    	conn, addr = s.accept()
    	print('Connected by', addr)
    	data = conn.recv(1024)
    	if data:
            query = str(data)
            query = query.replace('.', '')
            query = query.lower()
            vec = encoder.encode([query])[0]
            result = list(retrieve_topK(vec))
            print result
            conn.sendall('$$$'.join(result))

    except socket.error:
        print "Error Occured."
        break

conn.close()