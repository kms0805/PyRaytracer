import numpy as np
from .utils import *

class Camera:
    def __init__(self, ori, eye,  screen_w,screen_h,field_of_view = 90):
        self.ori = ori
        self.eye = eye
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.camera_width = np.linalg.norm(ori-eye)*np.tan(field_of_view * np.pi/180   /2.)*2.
        self.camera_height = self.camera_width
        self.cameraFwd = normalize(eye - ori)
        self.cameraRight = normalize(np.cross(self.cameraFwd,np.array([0,1,0])))
        self.cameraUp = np.cross(self.cameraRight,self.cameraFwd)

        self.x = np.linspace(-self.camera_width/2., self.camera_width/2., self.screen_w)
        self.y = np.linspace(self.camera_height/2., -self.camera_height/2., self.screen_h)
        xx,yy = np.meshgrid(self.x,self.y)
        self.x = xx.flatten()
        self.y = yy.flatten()

        self.dir_list = [normalize(self.eye - self.ori  +   self.cameraUp * self.y[i]  +  self.cameraRight * self.x[i] + self.cameraFwd) for i in range(len(self.x))]
