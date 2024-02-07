import cv2
import numpy as np

def histogram(img):
    M, N = img.shape
    L = 256
    hist = np.zeros(L)
    for i in range(M):
        for j in range(N):
            hist[img[i,j]] += 1 
    return hist
            
def EqualHist(img):
    L = 256
    M, N = img.shape
    hist = histogram(img)
    sum = 0.0
    T = np.zeros(L, np.uint8)
    for k in range(L):
        sum += hist[k]
        T[k] = round((L-1)/(M*N)*sum)
    blank_img = T[img].astype(np.uint8)
    return blank_img

def SpecHist(inImg, refImg):
    L = 256
    M, N = inImg.shape
    m, n = refImg.shape
    inHist = histogram(inImg)
    sum = 0.0
    T = np.zeros(L, np.uint8)
    for k in range(L):
        sum += inHist[k]
        T[k] = round((L-1)/(M*N)*sum)
    
    refHist = histogram(refImg)
    sum = 0.0
    G = np.zeros(L, np.uint8)
    for q in range(L):
        sum += refHist[q]
        G[q] = round((L-1)/(m*n)*sum)
        
    new = np.zeros(L, np.uint8)
    for i in range(L):
        j = 0
        while j < L-1 and T[i] > G[j]:
            j += 1
        new[i] = j

    blank_img = new[inImg].astype(np.uint8)
    return blank_img

def GaussianBlur(img):
    M, N = img.shape
    K = 1
    ksize = 5
    sigma = 25
    p = ksize // 2
    blank_img = np.zeros((M + 2*p, N + 2*p), np.float32)
    blank_img[p:p+M, p:p+N] = img.astype(np.float32)
    G = np.zeros((ksize, ksize), np.float32)
    for x in range(-p, -p + ksize):
        for y in range(-p, -p + ksize):
            G[x+p, y+p] = K*np.exp(-(x**2+y**2)/(2*(sigma**2)))
    G /= np.sum(G) #normalization
    tmp = blank_img
    for i in range(M):
        for j in range(N):
            blank_img[i+p, j+p] = np.sum(G*tmp[i:i+ksize, j:j+ksize])
    blank_img = blank_img[p:p+M, p:p+N]
    return blank_img.astype(np.uint8)

if __name__ == "__main__":
    img1 = cv2.imread('Q1.jpg', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('Q2.jpg', cv2.IMREAD_GRAYSCALE)
    img3 = cv2.imread('Q3.jpg', cv2.IMREAD_GRAYSCALE)
    EqualImg = EqualHist(img1)
    SpecImg = SpecHist(img2, img1)
    FiltImg = GaussianBlur(img3)
    print(FiltImg)
    print(cv2.GaussianBlur(img3, (5,5), 25))
    cv2.imshow('Output 1', EqualImg) 
    cv2.imshow('Output 2', SpecImg) 
    cv2.imshow('Output 3', FiltImg)
    cv2.imwrite('histogram_equalization.jpg', EqualImg)
    cv2.imwrite('histogram_specification.jpg', SpecImg)
    cv2.imwrite('gaussian_filter.jpg', FiltImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
