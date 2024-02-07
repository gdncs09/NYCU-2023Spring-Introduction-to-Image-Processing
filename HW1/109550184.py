import cv2
import numpy as np

def crop_image(img, h, w): #3x3 image
    image_list = []
    for r in range(0, 3):
        for c in range(0, 3):
            image = img[h*r:h*(r+1), w*c:w*(c+1)]
            image_list.append(image)
    return image_list
   
def BGR(image):
    B, G, R = image[:,:,0], image[:,:,1], image[:,:,2] #B, G, R = cv2.split(image)
    return B, G, R
   
def exchange_position_images(img1, img2, h, w):
    blank_image[0:h, 0:w] = img1
    blank_image[0:h, w*2:w*3] = img2

def gray_scale_image(image, h, w):
    B, G, R = BGR(image)
    gray_scale = np.uint8((R/255+G/255+B/255)/3*255)
    gray_scale = np.reshape(gray_scale, (h,w,1))
    return gray_scale

def intensity_resolution_image(image, h, w):
    gray_scale = gray_scale_image(image, h, w)
    intensity_level = 4
    compression = 256/intensity_level
    reduced_image = np.uint8(np.floor(gray_scale/256*intensity_level)*compression)
    reduced_image = np.reshape(reduced_image, (h,w,1))
    return reduced_image

def red_filter_image(image, h, w):
    gray_scale = gray_scale_image(image, h, w)
    B, G, R = BGR(image)
    red_condition = np.where(R > 150, 1, 0) + np.where(R*0.6 > G, 1, 0) + np.where(R*0.6 > B, 1, 0)
    red_condition = np.reshape(red_condition, (h, w, 1))
    red_filter = np.where(red_condition == 3, image, gray_scale)
    return red_filter

def yellow_filter_image(image, h, w):
    gray_scale = gray_scale_image(image, h, w)
    B, G, R = BGR(image)
    yellow_condition = np.where((G/255+R/255)*0.3*255 > B, 1, 0) + np.where(np.abs(G/255-R/255)*255 < 50, 1, 0)
    yellow_condition = np.reshape(yellow_condition, (h, w, 1))
    yellow_filter = np.where(yellow_condition == 2, image, gray_scale)
    return yellow_filter

def gchannel_operation_image(image):
    B, G, R = BGR(image)
    G = np.where(G*2 > G, G*2, 255) #G = cv2.add(G,G)
    merged_image = image
    merged_image[:,:,1] = G
    return merged_image

def bilinear_interpolation_image(image, h, w):
    new_h, new_w = h*2, w*2
    bilinear_image = np.zeros((new_h, new_w, 3), np.uint8) 
    for i in range(0, new_h): #y
        for j in range(0, new_w): #x
            x, y = j/2.0, i/2.0
            x1, y1 = int(x), int(y)
            x2, y2 = min(x1+1, w-1), min(y1+1, h-1)
            dx, dy = x-x1, y-y1
            w11 = (1-dx)*(1-dy)
            w21 = dx*(1-dy)
            w12 = (1-dx)*dy
            w22 = dx*dy
            bilinear_image[i,j] = w11*image[y1, x1] + w12*image[y1, x2] + w21*image[y2, x1] + w22*image[y2,x2]
    return bilinear_image[0:h, 0:w]

def bicubic_interpolation_image(image, h, w):
    new_h, new_w = h*2, w*2
    bicubic_image = np.zeros((new_h, new_w, 3), np.uint8) 
    
    def f(t): #Bicubic convolution algorithm
        a = -0.5
        if abs(t) <= 1:
            return ((a + 2) * abs(t**3) - (a + 3) * abs(t**2) + 1)
        elif 1 < abs(t) < 2:
            return (a * abs(t**3) - 5 * a * abs(t**2) + 8 * a * abs(t) - 4 * a)
        else:
            return 0
    
    for i in range(0, new_h):
        for j in range(0, new_w):
            x = j/2
            y = i/2
            x1 = int(x)+1
            y1 = int(y)+1
            
            pixel = np.zeros((3,))
            for ii in range(-1, 3):
                for jj in range(-1, 3):
                    if 0 <= y1+ii < h and 0 <= x1+jj < w:
                        weight = f(x - (x1 + jj)) * f(y - (y1 + ii))
                        pixel += weight * image[y1+ii, x1+jj]            
            pixel = np.clip(pixel, 0, 255)
            bicubic_image[i, j] = pixel.astype(np.uint8)
    return bicubic_image[0:h, 0:w]

if __name__ == "__main__":
    img = cv2.imread('test.jpg')
    
    #Init
    (height, width, channel) = img.shape #row, column
    h = int(height/3) #crop image row
    w = int(width/3) #crop image column
    image_list = crop_image(img, h, w)
    blank_image = np.zeros((height,width,channel), np.uint8)
    
    #--------------------------------------#
    exchange_position_images(image_list[2], image_list[0], h, w) #Exchange Position
    blank_image[h*2:h*3, 0:w] = gray_scale_image(image_list[6], h, w) #Gray Scale
    blank_image[h*2:h*3, w*2:w*3] = intensity_resolution_image(image_list[8], h, w) #Intensity Resolution
    blank_image[h:h*2, 0:w] = red_filter_image(image_list[3], h, w) #Color Filter - Red
    blank_image[h:h*2, w*2:w*3] = yellow_filter_image(image_list[5], h, w) #Color Filter - Yellow
    blank_image[h*2:h*3, w:w*2] = gchannel_operation_image(image_list[7]) #Channel Operation      
    blank_image[0:h, w:w*2] = bilinear_interpolation_image(image_list[1], h, w) #Bilinear Interpolation
    blank_image[h:h*2, w:w*2] = bicubic_interpolation_image(image_list[4],h ,w) #Bicubic Interpolation
    
    #--------------------------------------#
    cv2.imshow("109550184_Result", blank_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()