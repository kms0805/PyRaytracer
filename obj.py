from OpenGL.GLUT import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np

class OBJ(object):
    def __init__(self, filename):
        vertexes = []
        normals = []
        faces_vertexes = []
        faces_normals = []
        numfaces = 0
        fileobj = open(filename)
        for line in fileobj:
            if line.isspace():
                continue
            vals = line.split()
            if vals[0] == "v":
                v = [float(x) for x in vals[1:4]]
                vertexes.append(v)
            if vals[0] == "vn":
                v = [float(float(x)/np.linalg.norm(vals[1:4])) for x in vals[1:4]]
                normals.append(v)
            if vals[0] == 'f':
                if len(vals) > 4:
                    temp_vs = []
                    temp_ns = []
                    for v in vals[1:]:
                        v = v.split('/')
                        temp_vs.append(v[0])
                        temp_ns.append(v[2])
                    for i in range(1, len(temp_vs) - 1):
                        numfaces += 1
                        faces_vertexes.extend(vertexes[int(temp_vs[0]) - 1])
                        faces_vertexes.extend(vertexes[int(temp_vs[i]) - 1])
                        faces_vertexes.extend(vertexes[int(temp_vs[i+1]) - 1])
                        faces_normals.extend(normals[int(temp_ns[0]) - 1])
                        faces_normals.extend(normals[int(temp_ns[i]) - 1])
                        faces_normals.extend(normals[int(temp_ns[i + 1]) - 1])
                else:
                    numfaces += 1
                    for v in vals[1:]:
                        v = v.split('/')
                        faces_vertexes.extend(vertexes[int(v[0]) - 1])
                        faces_normals.extend(normals[int(v[2]) - 1])
        self.faces_vertexes = faces_vertexes
        faces_normals = (GLfloat * len(faces_normals))(*faces_normals)
        self.faces_normals = faces_normals
        self.npoligons = numfaces
        self.faces_colors = [1.0 for i in range(len(faces_vertexes))]
