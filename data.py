from raytracing.objects import Material, Light
import cv2
import numpy as np
#material list

def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img

name = cv2.imread('myname.png')
name = cv2.resize(name,(100, 50))
pattern = cv2.imread('pattern.png')
pattern = cv2.cvtColor(pattern, cv2.COLOR_BGR2RGB)
pattern = cv2.resize(pattern,(150,150))
name = convert(name,0,1,np.float64)
pattern = convert(pattern,0,1,np.float64)
# print(pattern)
#chrome
mat_ambient = [0.25, 0.25, 0.25]
mat_diffuse = [0.4, 0.4, 0.4]
mat_specular = [0.774597, 0.774597, 0.774597]
chrome = Material(mat_ambient, mat_diffuse, mat_specular, 0, name, True)

#red plastic
mat_ambient = [0.1, 0.0, 0.0]
mat_diffuse = [0.5, 0.0, 0.0]
mat_specular = [0.05, 0.05, 0.05]
red_plastic = Material(mat_ambient, mat_diffuse, mat_specular, 0, pattern, True)

#bronze
mat_ambient = [0.2125, 	0.1275, 0.054]
mat_diffuse = [	0.714, 0.4284, 	0.18144]
mat_specular = [0.393548, 0.271906, 0.166721]
bronze = Material(mat_ambient, mat_diffuse, mat_specular)

# red rubber
mat_ambient = [0.05, 0.0, 0.0]
mat_diffuse = [0.5, 0.4, 0.4]
mat_specular = [0.7, 0.04, 0.04]
red_rubber = Material(mat_ambient, mat_diffuse, mat_specular)

# green plastic
mat_ambient = [0.0, 0.0, 0.0]
mat_diffuse = [0.1, 0.35, 0.1]
mat_specular = [0.45, 0.55, 0.45]
green_plastic = Material(mat_ambient, mat_diffuse, mat_specular)

# gold
mat_ambient = [	0.24725, 0.1995, 0.0745]
mat_diffuse = [0.75164, 0.60648, 0.22648]
mat_specular = [0.628281, 0.555802, 0.366065]
gold = Material(mat_ambient, mat_diffuse, mat_specular)

#ice
mat_ambient = [0.0, 0.0, 0.054]
mat_diffuse = [0.0, 0.1, 0.4]
mat_specular = [0.5, 0.5, 0.8]
ice = Material(mat_ambient, mat_diffuse, mat_specular, 0.9)

#Light1

ambientLight = [0.00, 0.00, 0.00]
diffuseLight = [0.3, 0.3, 0.3]
specularLight = [0.3, 0.3, 0.3]
position = [0, 100, 100.0,]
Light1 = Light(ambientLight, diffuseLight, specularLight, position)

ambientLight = [0, 0.001, 0.0]
diffuseLight = [0.0, 0.1, 0.0]
specularLight = [0.0, 0.2, 0.0,]
position = [-100, 100, -100.0,]
Light2 = Light(ambientLight, diffuseLight, specularLight, position)

ambientLight = [0.001, 0, 0.0]
diffuseLight = [0.1, 0.0, 0]
specularLight = [0.2, 0.0, 0.0]
position = [100, 100, 100]
Light3 = Light(ambientLight, diffuseLight, specularLight, position)





