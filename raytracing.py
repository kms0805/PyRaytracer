from raytracing.objects import *
from raytracing.utils import *
from data import *
from raytracing.camera import *
import numpy as np
import cv2
import obj

#load objects
### rubberduck
obj_model = obj.OBJ('Rubber_Duck_obj.obj')
T = translateM(50,-20,0)
R = np.eye(4)
S = scaleM(200)
duck = TriangleMesh(gold, obj_model,T,R,S)

### red cube
T = translateM(20,-10,20)
S = scaleM(20)
red_cube = Cube(red_rubber,T,R,S)

### bronze sphere
center = np.array([-20,-10,40])
radius = 10
bronze_sphere = Sphere(bronze,center,radius)

###chrome floor
fp1 = np.array([2000, -20, 2000])
fp2 = np.array([2000, -20, -2000])
fp3 = np.array([-2000, -20, -2000])
fp4 = np.array([-2000, -20, 2000])
floor = Square(chrome, fp1, fp2, fp3, fp4)

### textured sphere
center = np.array([0, -10, 0])
radius = 10
red_sphere = Sphere(red_plastic,center,radius)

### green cube
T = translateM(-50,-10,20)
S = scaleM(20)
green_cube = Cube(green_plastic,T,R,S)

### ice cube
T = translateM(0,0,60)
S = scaleM(20)
ice_cube = Cube(ice,T,R,S)

objects = [duck, bronze_sphere ,red_cube, floor, red_sphere, green_cube]
objects = [ice_cube, bronze_sphere , floor, red_sphere, green_cube, red_cube]
# objects = [red_sphere]

background_color = 0.0 * np.ones(3)

ambient = 0.1 * np.ones(3)

light_position = np.array([0, 10, 100])
light_color = 0.5*np.ones(3)

def get_intersected_object(O, D):
	distances = np.zeros(len(objects))
	obj_index = -1
	for i, object in enumerate(objects):
		distances[i] = object.get_distance(O, D)
	obj_index = np.argmin(distances)
	min_distance = distances[obj_index]
	return obj_index, min_distance


def is_shadowed(point, light):
	light_vector = normalize(light.position - point)
	obj_index, distance = get_intersected_object(point, light_vector)
	if objects[obj_index].material.ref_n != 0:
		return 0.5
	if distance > 444444:
		return 0
	else:
		return 1




def get_intersect_point(O, D, distance):
	return O + D * distance

def tracer(depth, O, D, light, is_refr = False):
	
	if depth < 0:
		return np.zeros(3)

	obj_index, distance = get_intersected_object(O, D)
	if distance > 444444:
		return background_color

	obj = objects[obj_index]

	intersect_point = get_intersect_point(O, D, distance)
	object_normal_vector = obj.get_normal(intersect_point)
	intersect_point = intersect_point + 0.0001 * object_normal_vector
	toL = normalize(light.position - intersect_point)
	toO = normalize(O - intersect_point)

	if obj.material.is_texture == True:
		diffuse = np.array(obj.get_texture(intersect_point))
	else:
		diffuse = np.array(obj.material.diffuse)
	specular = np.array(obj.material.specular)
	ambient = np.array(obj.material.specular)
	
	reflection_ray_vector = np.real(D - 2 * np.dot(D, object_normal_vector) * object_normal_vector)
	reflection_ray_color = tracer(depth - 1, intersect_point, reflection_ray_vector, light)

	traced_light_color = np.clip(reflection_ray_color, 0, 1)

	diffuse_color = diffuse * light.diffuse * np.dot(object_normal_vector, toL)
	specular_color = specular * light.specular * np.real(np.dot(object_normal_vector, normalize(toL + toO))) ** 50 
	ambient_color = ambient * light.ambient
	color = diffuse_color + specular_color + ambient_color
	if not is_refr:
		color = color + specular * traced_light_color

	if obj.material.ref_n != 0 :
		ref_n = obj.material.ref_n
		# print('D:' ,D)
		# print('N: ', object_normal_vector)
		if is_refr:
			ref_n = 1/ref_n
			cos_theta_i = np.dot(-1 * D, -1*object_normal_vector)
			if 1 - ref_n ** 2 * (1 - cos_theta_i ** 2) >= 0:
				# print('refr')
				cos_theta_r = np.sqrt(1 - ref_n ** 2 * (1 - cos_theta_i ** 2))
				refraction_ray_vector = (ref_n * cos_theta_i - cos_theta_r) * -1 * object_normal_vector - ref_n * -1 * D
			else:
				# print('refl')
				refraction_ray_vector = np.real(D - 2 * np.dot(D, -1*object_normal_vector) * -1*object_normal_vector)

			refraction_ray_vector = normalize(refraction_ray_vector)
			# print('out', refraction_ray_vector)

			refraction_ray_color = tracer(depth - 1, intersect_point, refraction_ray_vector, light, False)
		else:
			cos_theta_i = np.dot(-1 * D, object_normal_vector)
			cos_theta_r = np.sqrt(1 - ref_n ** 2 * (1 - cos_theta_i ** 2))
			refraction_ray_vector = (ref_n * cos_theta_i - cos_theta_r) * object_normal_vector - ref_n *-1 *  D

			refraction_ray_vector = normalize(refraction_ray_vector)
			# print('in', refraction_ray_vector)
			intersect_point = intersect_point - 0.0002 * object_normal_vector
			refraction_ray_color = tracer(depth - 1, intersect_point, refraction_ray_vector, light, True)

		color = color + refraction_ray_color

	if is_shadowed(intersect_point, light) == 1:
		return 0.1 * color
	elif is_shadowed(intersect_point, light) == 0.5:
		return 0.7 * color
	return color







def main():
	global objects
	h = 300
	w = 300
	objects = [duck, ice_cube, bronze_sphere, floor, red_sphere, green_cube, red_cube]
	image = np.zeros(shape=(h * w, 3))
	camera = Camera(np.array([50, 50, 200]), np.array([0, 0, 0]), h, w, 45)
	print('start rendering...')
	for i in range(len(image)):
		if len(image) // 4 == i:
			print('25%....')
		if len(image) // 2 == i:
			print('50%....')
		if len(image) // 4 * 3 == i:
			print('75%....')
		O = camera.ori
		D = camera.dir_list[i]
		pixel_color = tracer(4, O, D, Light1) + tracer(4, O, D, Light2) + tracer(4, O, D, Light3)
		image[i] = np.clip(pixel_color, 0, 1)
	print("finish~~~~~~~~~")
	image = image.reshape((h, w, 3))

	img_n = cv2.normalize(src=image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
	# img_n = cv2.resize(img_n, dsize=(500, 500), interpolation=cv2.INTER_AREA)
	RGBimage = cv2.cvtColor(img_n, cv2.COLOR_BGR2RGB)
	cv2.imwrite('output.png', RGBimage)
	cv2.imshow('output', RGBimage)
	cv2.waitKey(0)

if __name__ == "__main__":
	main()