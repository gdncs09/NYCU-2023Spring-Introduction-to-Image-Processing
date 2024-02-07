#HW4
import cv2
import numpy as np
import matplotlib.pyplot as plt

def NotchRejectFilter(M, N, D0, u0, v0, service):
    H = np.zeros((M, N))
    u, v = np.repeat(np.arange(M), N).reshape((M, N)), np.repeat(np.arange(N), M).reshape((N, M)).transpose()
    D1 = np.sqrt((u-M/2-u0)**2 + (v-N/2-v0)**2)
    D2 = np.sqrt((u-M/2+u0)**2 + (v-N/2+v0)**2)
    if service == "Ideal":
        H[(D1<=D0) | (D2<=D0)] = 0
        H[(D1>D0) & (D2>D0)] = 1
    elif service == "Butterworth":
        n = 2
        D1D2 = D1*D2
        mask = D1D2!=0
        H[mask] = 1/(1+(D0**2/D1D2[mask])**n)
    elif service == "Gaussian":
        H = 1-np.exp(-1/2*(D1*D2/D0**2)**2)
    return H

def DoImage1(img):
    #Step 1: Get the Spectrum
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    c = 255 / np.log(1 + np.max(np.abs(fshift))) # c = (L-1)/log(1+max_val) 
    spectrum = c * np.log(np.abs(fshift) + 1) #s = c*log(1+r)
    #Step 2: Apply filter on Spectrum to reduce noise. 
    M, N = spectrum.shape
    center_M = M//2
    center_N = N//2
    notch_filter = np.ones((M, N))
    for u in range(center_M-50):
        notch_filter *= NotchRejectFilter(M, N, 10, center_M-u, center_N-center_N, "Ideal")
    filtered_spectrum = spectrum*notch_filter
    #Step 3: Convert the new Spectrum to spatial domain using Inverse Fourier Transform
    notch_reject = np.fft.ifftshift(fshift * notch_filter)
    inverse_notch_reject = np.fft.ifft2(notch_reject)
    final_img = np.abs(inverse_notch_reject)
    #Output
    scaled_img = cv2.convertScaleAbs(final_img / final_img.max() * 255)
    cv2.imwrite('Image1.jpg', scaled_img)  
    plt.subplot(131)
    plt.title('Step 1')
    plt.axis('off')
    plt.imshow(spectrum, "gray")
    plt.subplot(132)
    plt.title('Step 2')
    plt.axis('off')
    plt.imshow(filtered_spectrum, "gray")
    plt.subplot(133)
    plt.title('Step 3')
    plt.axis('off')
    plt.imshow(final_img, "gray")
    plt.show()
    plt.close()
    
def DoImage2(img):
    #Step 1: Get the Spectrum
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    c = 255 / np.log(1 + np.max(np.abs(fshift))) # c = (L-1)/log(1+max_val) 
    spectrum = c * np.log(np.abs(fshift) + 1) #s = c*log(1+r)
    #Step 2: Apply filter on Spectrum to reduce noise. 
    M, N = spectrum.shape
    center_M = M//2
    center_N = N//2
    notch_filter = np.ones((M, N))
    for u in range(20,M-20):
        for v in range(20,center_N-20):
            if spectrum[u,v] > 160 and np.abs(center_M-u) > 5 and np.abs(center_N-v) > 5: 
                map = spectrum[u-20:u+20, v-20:v+20]  
                if np.max(map) == spectrum[u,v]:
                    u0 = center_M-u
                    v0 = center_N-v
                    notch_filter *= NotchRejectFilter(M, N, 30, u0, v0, "Butterworth")
    filtered_spectrum = spectrum*notch_filter
    #Step 3: Convert the new Spectrum to spatial domain using Inverse Fourier Transform
    notch_reject = np.fft.ifftshift(fshift * notch_filter)
    inverse_notch_reject = np.fft.ifft2(notch_reject)
    final_img = np.abs(inverse_notch_reject)
    #Output
    scaled_img = cv2.convertScaleAbs(final_img / final_img.max() * 255)
    cv2.imwrite('Image2.jpg', scaled_img)
    plt.subplot(131)
    plt.title('Step 1')
    plt.axis('off')
    plt.imshow(spectrum, "gray")
    plt.subplot(132)
    plt.title('Step 2')
    plt.axis('off')
    plt.imshow(filtered_spectrum, "gray")
    plt.subplot(133)
    plt.title('Step 3')
    plt.axis('off')
    plt.imshow(final_img, "gray")
    plt.show()
    plt.close()
    
if __name__ == '__main__':
    img1 = cv2.imread('7.jpg', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('test2.tif', cv2.IMREAD_GRAYSCALE)
    DoImage1(img1)
    DoImage2(img2)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()