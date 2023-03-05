import numpy as np

def normalize(v):
	norm = np.linalg.norm(v)
	if norm == 0:
		return v
	return v / norm
def hom(x):
	print(x)
	x = np.array(x)
	if len(x) != 3:
		print("error")
		return x
	return np.array([x[0],x[1],x[2],1])
def vec3(x):
	if len(x) != 4:
		print("error")
		return x
	return np.array([x[0],x[1],x[2]])

def translateM(x,y,z):
	return np.array([[1, 0, 0, x],
			  [0, 1, 0, y],
			  [0, 0, 1, z],
			  [0, 0, 0, 1]])
def scaleM(x):
	return np.array([[x, 0, 0, 0],
			  [0, x, 0, 0],
			  [0, 0, x, 0],
			  [0, 0, 0, 1]])
def rotateM(x):
	return np.array([[1, 0, 0, 0],
			  [0, 1, 0, 0],
			  [0, 0, 1, 0],
			  [0, 0, 0, 1]])


