# NYCU-2023Spring-Introduction-to-Image-Processing
## Homework 1 
1. Exchange Position <br>
2. Gray Scale <br>
3. Intensity Resolution (Gray scale with 4 intensity levels) <br>
4. Color Filter – Red <br>
5. Color Filter – Yellow <br>
6. Channel Operation (Double the value of green channel) <br>
7. Bilinear Interpolation – 2x <br>
8. Bicubic Interpolation – 2x

## Homework 2
1. Histogram Equalization - Q1.jpg <br>
2. Histogram Specification - Transform the histogram of Q1.jpg to the histogram of Q2.jpg <br>
3. Gaussian Filter (K=1, size=5x5, σ=25) - Q3.jpg 

## Homework 4
Denoising <br>
&ensp; Step1. Get the Spectrum using Fourier Transform. (ex. np.fft.fft2) <br>
&ensp; Step2. Apply filter on Spectrum to reduce noise. <br> 
&ensp; Step3. Convert the new Spectrum to spatial domain using Inverse Fourier Transform.