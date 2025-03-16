#!/usr/bin/env python
# -*- coding: utf-8 -*-
# TP reconstruction TDM (CT)
# Prof: Philippe Després
# programme: Dmitri Matenine (dmitri.matenine.1@ulaval.ca)


# libs
import numpy as np

## filtrer le sinogramme
## ligne par ligne
def filterSinogram(sinogram):
    for i in range(sinogram.shape[0]):
        sinogram[i] = filterLine(sinogram[i])

## filter une ligne (projection) via FFT
## filter une ligne (projection) via FFT
def filterLine(projection):
    # Obtenir la taille de la projection
    n = len(projection)
    
    # Calculer la FFT de la projection
    proj_fft = np.fft.fft(projection)
    
    # Créer un filtre rampe dans le domaine fréquentiel
    # Le filtre rampe est |f| où f est la fréquence
    freq = np.fft.fftfreq(n)
    ramp_filter = np.abs(freq)
    
    # Appliquer le filtre rampe
    filtered_proj_fft = proj_fft * ramp_filter
    
    # Calculer la transformée de Fourier inverse pour revenir au domaine spatial
    filtered_projection = np.real(np.fft.ifft(filtered_proj_fft))
    
    return filtered_projection