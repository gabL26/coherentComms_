from IPython import get_ipython;
import numpy as np
from math import sin, pi
from matplotlib import pyplot as plt
get_ipython().magic('reset -sf')

# Function definitions
def channel(x, channelNoiseStd, alpha, linkLength):
    x = np.exp(-(alpha/2)*linkLength)*x
    x = x + channelNoiseStd*(np.random.randn(len(x)) + 1j*np.random.randn(len(x)))
    return x

def retrieveCode(comparison, initialBit):
    retrievedCode = np.ones(len(comparison)) 
    for i in range(len(comparison)-1):  #probably there is a better way to do this with matrices
        if comparison[i]==1:
            retrievedCode[i+1] = retrievedCode[i];
        else:
            retrievedCode[i+1] = not retrievedCode[i];
    return retrievedCode

## Constants
laserWavelength = 1550e-9; #[m]
c = 3e8; #[m/s] speed of light  
nu = c/laserWavelength;
q = 1.602e-19 #[C] elementary charge
h = 6.63e-34 #[J-s] Planck's constant
eta = 0.7 # quantum efficiency

noiseSteps = 200

numberOfErrors = np.zeros(noiseSteps)
estimatedBER = np.zeros(noiseSteps)

for k in range(noiseSteps):
    #Transmitter part
    # generate a sequence of binary numbers
    code = np.random.randint(0,2, 100)
    code[0] = 1 #this enforces a trivial protocol in which every communication starts with a '1'

    power = 1e-2 #10 mW of input power

    # This step codes the binary numbers in two symbols, representing the phase that
    # the electro-optic modulator applies to the carrier electric field. 
    symbols = np.exp(1j*pi*code)
    A = np.sqrt(power)*symbols #[W^0.5] optical field
    
    #Add gaussian noise noise_std*(a+ib), where a and b are two random number
    transmitterNoiseStd  = k*1e-3*np.sqrt(power)
    A = A + transmitterNoiseStd*(np.random.randn(len(A)) + 1j*np.random.randn(len(A))) 
    
    #Channel part 
    channelNoiseStd = 0
    alpha_dB = 0.5/1e3 #[dB/m] losses, e.g. 0.5 dB/km
    alpha = alpha_dB/4.343 #[1/m] losses
    linkLenght = 10e3; #[m]
    A = channel(A, channelNoiseStd, alpha, linkLenght)
    
    # A consequence of high losses is that the signal is more sensitive to phase noise sources
    # inside the fiber, but also that the '1' seen by the DD receiver would be closer to the 
    # '0' (SNR degradation) or to the receiver noise floor.
    # Other kind of noises in the fibers are PMD or nonlinear dispersion, in case of high laser fields. 
    
    # Receiver
    # The Differential BPSK (DBPSK) receiver uses a Mach-Zehnder interferometer with a delay line on an arm
    # equal to the bit time.
    # This allows to avoid an absolute phase reference like a local laser oscillator, which is needed for coherent
    # detection. Also, in case of coherent detection one should have used three more detectors and a pi/2 phase shift.
    # A coherent receiver however would have led to a lower BER for the same SNR. 
    
    responsivity = eta*q/(h*nu)
    
    # To simulate the DD receiver, it is possible to shift the p array by one position. 
    A_shifted = np.roll(A,-1)
    symbols_shifted = np.roll(symbols, -1)
    
    i_ph = responsivity*np.abs(A+A_shifted)**2 #[A] photocurrent
    #To make a more realistic system, I should add thermal noise and shot noise. 
    #Eventually if a bitrate is provided, a detector with a suitable bandwidth should be chosen,
    #and the shot noise and thermal noise contribution can be added. 
    #Until now, the only noise contribution is given by the transmitter.
    
    # Threshold detection
    i_thrs = max(i_ph)/2
    #The comparison array is 1 if two consequent symbols are identical (both 0 or both 1),
    #0 otherwise.
    
    comparison = i_ph>i_thrs
    #retrieved code allows to reconstruct the original string --> it should be equal to code
    retrievedCode = retrieveCode(comparison, 1) 
    
    numberOfErrors[k] = np.sum(np.abs(code-retrievedCode))
    estimatedBER[k] = numberOfErrors[k]/len(code)

plt.plot(range(0, noiseSteps)*transmitterNoiseStd*1e-3, numberOfErrors)
plt.xlabel('STD transmitter noise [$W^{1/2}$]')
plt.ylabel('Number of errors')
