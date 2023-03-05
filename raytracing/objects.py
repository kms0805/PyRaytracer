import numpy as np
from .utils import *
class Material:
    def __init__(self, ambient, diffuse, specular, ref_n = 0, texture = None, is_texture = False):
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.ref_n = ref_n # for refraction
        self.is_texture = is_texture
        self.texture = texture

class Light:
     def __init__(self, ambient, diffuse, specular, position):
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.position = position

class Sphere:
    def __init__(self, material, center, radius):
        self.material = material
        self.center = center
        self.radius = radius

    def get_distance(self, O, D):
        # O -> origin of ray
        # D -> d vec of ray
        Q = O - self.center
        a = np.dot(D, D)
        b = 2 * np.dot(D, Q)
        c = np.dot(Q, Q) - self.radius ** 2
        d = b ** 2 - 4 * a * c
        if d < 0:
            return np.inf
        if d >= 0:
            t1 = (-b + np.sqrt(d)) / (2 * a)
            t2 = (-b - np.sqrt(d)) / (2 * a)
            distance = np.minimum(t1, t2)
            if distance < 0:
                return np.inf
            return distance

    def get_normal(self, intersect_point):
        return normalize(intersect_point - self.center)

    def get_texture(self, intersect_point):
        pattern = self.material.texture
        v = normalize(intersect_point - self.center)
        x= abs(v[0])
        y = abs(v[1])
        z = abs(v[2])
        new_x = 0.5 + np.arctan2(x,z)/(np.pi*2)
        new_y = 0.5 - np.arcsin(y)/np.pi
        t = pattern[np.int(np.floor(new_x*150)), np.int(np.floor(new_y*150)), :]
        return t


class Triangle:
    def __init__(self, material, p1, p2, p3):
        self.material = material
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.normal = normalize(np.cross((self.p2 - self.p1),( self.p3 - self.p1)))
        self.plane_constant = -1 * np.dot(self.normal,self.p1)
        self.edge21 = self.p2 - self.p1
        self.edge32 = self.p3 - self.p2
        self.edge13 = self.p1 - self.p3

    def get_distance(self, O, D):
        DdotN = np.dot(D, self.normal)
        if np.abs(DdotN) < 1e-5:
            return np.inf
        distance = -1 * np.real(np.divide(self.plane_constant+ np.dot(O, self.normal), DdotN))
        if distance < 0:
            return np.inf
        P = O + D * distance # P is intersection point
        vp1 = P - self.p1
        vp2 = P - self.p2
        vp3 = P - self.p3
        is_inside = np.dot(self.normal,np.cross(self.edge21, vp1)) > 0 and \
                    np.dot(self.normal,np.cross(self.edge32, vp2)) > 0 and \
                    np.dot(self.normal,np.cross(self.edge13, vp3)) > 0
        if not is_inside:
            return np.inf
        return distance

    def get_normal(self, intersect_point):
        return self.normal

class TriangleMesh:
    def __init__(self, material, obj_model, translate, rotate, scale):
        self.material = material
        self.triangles = []
        
        for i in range(int(len(obj_model.faces_vertexes) / 9)):
            j = i*3
            v1 = np.array([obj_model.faces_vertexes[3 * j], obj_model.faces_vertexes[3 * j + 1],
                            obj_model.faces_vertexes[3 * j + 2], 1])
            j += 1
            v2 = np.array([obj_model.faces_vertexes[3 * j], obj_model.faces_vertexes[3 * j + 1],
                            obj_model.faces_vertexes[3 * j + 2], 1])
            j += 1
            v3 = np.array([obj_model.faces_vertexes[3 * j], obj_model.faces_vertexes[3 * j + 1],
                            obj_model.faces_vertexes[3 * j + 2], 1])
            v1 = vec3(translate @ scale @ rotate @ v1)
            v2 = vec3(translate @ scale @ rotate @ v2)
            v3 = vec3(translate @ scale @ rotate @ v3)
            self.triangles.append(Triangle(material, v1, v2, v3))
        self.min_idx = -1


    def get_distance(self, O, D):
        distances = np.zeros(len(self.triangles))
        for i,tr in enumerate(self.triangles):
            distances[i] = tr.get_distance(O,D)
        if np.min(distances) > 4444444:
            self.min_idx = -1
            return np.inf
        self.min_idx = np.argmin(distances)
        distance = distances[self.min_idx]
        return distance

    def get_normal(self, intersect_point):
        return self.triangles[self.min_idx].get_normal(intersect_point)

class Square:
    def __init__(self, material, p1, p2, p3, p4):
        self.material = material
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.normal = normalize(np.cross((self.p2 - self.p1),( self.p3 - self.p1)))
        self.plane_constant = -1 * np.dot(self.normal,self.p1)
        self.edge21 = self.p2 - self.p1
        self.edge32 = self.p3 - self.p2
        self.edge43 = self.p4 - self.p3
        self.edge14 = self.p1 - self.p4

    def get_distance(self, O, D):
        DdotN = np.dot(D, self.normal)
        if np.abs(DdotN) < 1e-5:
            return np.inf
        distance = -1 * np.real(np.divide(self.plane_constant+ np.dot(O, self.normal), DdotN))
        if distance < 0:
            return np.inf
        P = O + D * distance # P is intersection point
        vp1 = P - self.p1
        vp2 = P - self.p2
        vp3 = P - self.p3
        vp4 = P - self.p4
        is_inside = np.dot(self.normal,np.cross(self.edge21, vp1)) > 0 and \
                    np.dot(self.normal,np.cross(self.edge32, vp2)) > 0 and \
                    np.dot(self.normal,np.cross(self.edge43, vp3)) > 0 and \
                    np.dot(self.normal,np.cross(self.edge14, vp4)) > 0
        if not is_inside:
            return np.inf
        return distance

    def get_normal(self, intersect_point):
        return self.normal

    def get_texture(self, intersect_point):
        if intersect_point[0] > 99 - 50 or intersect_point[2] > 49 + 25 or intersect_point[0] < 2 - 50 or intersect_point[2] < 2 + 25:
            return self.material.diffuse
        pattern = self.material.texture
        v = intersect_point
        t = pattern[np.int(np.floor(v[2]) - 25), np.int(np.floor(v[0]) + 50) ,:]
        
        return t

class Cube:
    def __init__(self, material, translate, rotate, scale):
        self.material = material
        self.squares = []
        v1 = vec3(translate @ scale @ rotate @ np.array([-0.5, -0.5, -0.5, 1])) # - - -1
        v2 = vec3(translate @ scale @ rotate @ np.array([ 0.5, -0.5, -0.5, 1 ])) # + - -2
        v3 = vec3(translate @ scale @ rotate @ np.array([ 0.5,  0.5, -0.5, 1])) # + + -3
        v4 = vec3(translate @ scale @ rotate @ np.array([-0.5,  0.5, -0.5, 1])) # - + -4
        v5 = vec3(translate @ scale @ rotate @ np.array([-0.5, -0.5,  0.5, 1])) # - - +5
        v6 = vec3(translate @ scale @ rotate @ np.array([ 0.5, -0.5,  0.5, 1])) # + - +6
        v7 = vec3(translate @ scale @ rotate @ np.array([ 0.5,  0.5,  0.5, 1])) # + + +7
        v8 = vec3(translate @ scale @ rotate @ np.array([-0.5,  0.5,  0.5, 1])) # - + +8
        self.squares.append(Square(self.material, v3, v4, v8, v7))
        self.squares.append(Square(self.material,v6, v5, v1, v2))
        self.squares.append(Square(self.material,v7, v8, v5, v6))
        self.squares.append(Square(self.material,v2, v1, v4, v3))
        self.squares.append(Square(self.material,v8, v4, v1, v5))
        self.squares.append(Square(self.material,v3, v7, v6, v2))
        self.min_idx = -1


    def get_distance(self, O, D):
        distances = np.zeros(len(self.squares))
        for i,sq in enumerate(self.squares):
            distances[i] = sq.get_distance(O,D)
        if np.min(distances) > 4444444:
            self.min_idx = -1
            return np.inf
        self.min_idx = np.argmin(distances)
        distance = distances[self.min_idx]
        return distance

    def get_normal(self, intersect_point):
        return self.squares[self.min_idx].get_normal(intersect_point)





