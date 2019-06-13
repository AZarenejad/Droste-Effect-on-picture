import numpy as np
import cv2
import cmath
import math
import copy


r1 = .2
r2 = .9
repeat = 6
pi = np.pi
teta = pi / 4
e = np.e




def func_log(z):
    length = abs(z)
    if r1 < length < r2 :
        return  np.log(z / r1)
    else:
        return 0

def func_exp(z):
    return np.exp(z)

def func_rotate(z):
    return z * np.exp(1j * teta)


def read_image_create_X_Y_Z(pic_name):
    image = cv2.imread(pic_name)
    image = np.array(image)
    row , col = image.shape[0],image.shape[1]
    vec1 = np.linspace(-1,1,num = row , endpoint =True)
    vec2 = np.linspace(-1,1,num = col , endpoint = True)
    X,Y = np.meshgrid(vec2 , vec1)
    Z  = X + Y*1j
    return image, X, Y , Z, row, col



def create_X_new_and_Y_new(Wx,Wy,row,col,image):
    Xnew = (Wx/np.max(np.abs(Wx)) + 1)*image.shape[1]/2
    Ynew = (Wy/np.max(np.abs(Wy)) + 1)*image.shape[0]/2
    Xnew = np.clip(Xnew, 0, image.shape[1]-1)
    Ynew = np.clip(Ynew, 0, image.shape[0]-1)
    Xnew = np.floor(Xnew).astype(int)
    Ynew = np.floor(Ynew).astype(int)
    return Xnew,Ynew
   

    
def create_total_image_after_mapping(Xnew,Ynew,image,row,col,repeat):
    new_img = np.zeros([row, col, 3], dtype=np.uint8)
    for i in range(repeat*row):
        for j in range(col):
            new_img[Ynew[i][j], Xnew[i][j]] = image[i % col][j]
    # cv2.imshow('image', new_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return new_img
  

def apply_func_log_get_new_matrix_Wx_Wy(Z,X,Y, row, col):
    Z = Z.flatten()
    Z = np.select([np.abs(Z) <= r2], [Z])
    Z = np.select([np.abs(Z) >= r1], [np.log(Z/r1)])
    Z = Z.reshape(X.shape)
    Wx, Wy = np.real(Z), np.imag(Z)
    return Z,Wx,Wy


def apply_func_rotate_get_new_matrix_Wx_Wy(Z,X,Y, row, col):
    Z = Z.flatten()
    Z = Z * (np.power(e, 1j*teta))
    Z = Z.reshape(X.shape)
    Wx, Wy = np.real(Z), np.imag(Z)
    return Z,Wx,Wy
   

def apply_func_exp_get_new_matrix_Wx_Wy(Z,X,Y, row, col):
    Z = Z.flatten()
    Z = np.power(e,Z)
    Z = Z.reshape(X.shape)
    Wx, Wy = np.real(Z), np.imag(Z)
    return Z,Wx,Wy

def log_mapping(pic_name):
    image, X,Y, Z, row, col =  read_image_create_X_Y_Z(pic_name)
    Z , Wx, Wy =  apply_func_log_get_new_matrix_Wx_Wy(Z,X,Y, row, col)
    Xnew, Ynew = create_X_new_and_Y_new(Wx,Wy,row,col,image)
    img = create_total_image_after_mapping(Xnew,Ynew,image,row,col,1)
    cv2.imwrite('log.jpg',img)

def rotate_mapping(pic_name):
    image,X,Y, Z, row, col =  read_image_create_X_Y_Z(pic_name)
    Z, Wx, Wy =   apply_func_rotate_get_new_matrix_Wx_Wy(Z,X,Y, row, col)
    Xnew, Ynew = create_X_new_and_Y_new(Wx,Wy,row,col,image)
    img = create_total_image_after_mapping(Xnew,Ynew,image,row,col,1)
    cv2.imwrite('rotate.jpg',img)



def exp_mapping(pic_name):
    image,X,Y,Z,row,col = read_image_create_X_Y_Z(pic_name)
    Z , Wx, Wy = apply_func_exp_get_new_matrix_Wx_Wy(Z,X,Y, row, col)
    Xnew, Ynew = create_X_new_and_Y_new(Wx,Wy,row,col,image)
    img = create_total_image_after_mapping(Xnew,Ynew,image ,row,col,1)
    cv2.imwrite('exp.jpg',img)


def func_droste(Xnew,Ynew,Wx,Wy,image,row,col):
    Y_ = Ynew
    for i in range(repeat-1):
        Y_ = np.concatenate((Y_, Ynew+(i+1)*image.shape[0]), axis=0)
    Ynew = Y_
    Xnew = np.tile(Xnew, (repeat, 1))

    Wxnew = (Xnew*2/row - 1)*np.max(np.abs(Wx))
    Wynew = (Ynew*2/col - 1)*np.max(np.abs(Wy))
    Znew = Wxnew + 1j*Wynew
    alpha = np.arctan(np.log(r2/r1)/(2*np.pi))
    Znew = Znew * (np.power(np.e, 1j*alpha)) * np.cos(alpha)
    Znew = np.power(np.e, Znew)
    Xnew = np.real(Znew)
    Ynew = np.imag(Znew)
    
    Xnew = (Xnew/np.max(np.abs(Xnew)) + 1)*col/2
    Ynew = (Ynew/np.max(np.abs(Ynew)) + 1)*row/2
    Xnew = np.clip(Xnew, 0, col-1)
    Ynew = np.clip(Ynew, 0, row-1)
    Xnew = np.floor(Xnew).astype(int)
    Ynew = np.floor(Ynew).astype(int)
    return Xnew,Ynew



def apply_droste_effect(pic_name):
    image, X ,Y , Z, row, col =  read_image_create_X_Y_Z(pic_name)
    Z , Wx, Wy =  apply_func_log_get_new_matrix_Wx_Wy(Z,X,Y, row, col)
    Xnew, Ynew = create_X_new_and_Y_new(Wx,Wy,row,col,image)
    Xnew,Ynew =  func_droste(Xnew,Ynew,Wx,Wy,image,row,col)
    new_img = create_total_image_after_mapping(Xnew,Ynew,image,row,col,repeat)
    cv2.imwrite('total.jpg',new_img)




if __name__ == "__main__":
    # log_mapping("clock.jpg")
    # rotate_mapping("clock.jpg")
    # exp_mapping("clock.jpg")
    apply_droste_effect('clock.jpg')