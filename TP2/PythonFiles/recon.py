#!/usr/bin/env python
# -*- coding: utf-8 -*-
# TP reconstruction TDM (CT)
# Prof: Philippe Després
# programme: Dmitri Matenine (dmitri.matenine.1@ulaval.ca)


# libs
import numpy as np
import time
import matplotlib.pyplot as plt
import scipy as sp
# local files
import geo as geo
import util as util
import CTfilter as CTfilter

## créer l'ensemble de données d'entrée à partir des fichiers
def readInput():
    # lire les angles
    [nbprj, angles] = util.readAngles(geo.dataDir+geo.anglesFile)

    print("nbprj:",nbprj)
    print("angles min and max (rad):")
    print("["+str(np.min(angles))+", "+str(np.max(angles))+"]")

    # lire le sinogramme
    [nbprj2, nbpix2, sinogram] = util.readSinogram(geo.dataDir+geo.sinogramFile)

    if nbprj != nbprj2:
        print("angles file and sinogram file conflict, aborting!")
        exit(0)

    if geo.nbpix != nbpix2:
        print("geo description and sinogram file conflict, aborting!")
        exit(0)

    return [nbprj, angles, sinogram]


## reconstruire une image TDM en mode rétroprojection
def laminogram():
    
    [nbprj, angles, sinogram] = readInput()

    # initialiser une image reconstruite
    image = np.zeros((geo.nbvox, geo.nbvox))

    center_voxel = (geo.nbvox - 1) / 2
    center_pixel = (geo.nbpix - 1) / 2
    scale = geo.voxsize / geo.pixsize

    # "etaler" les projections sur l'image
    # ceci sera fait de façon "voxel-driven"
    # pour chaque voxel, trouver la contribution du signal reçu
    for j in range(geo.nbvox): # colonnes de l'image
        print("working on image column: "+str(j+1)+"/"+str(geo.nbvox))
        for i in range(geo.nbvox): # lignes de l'image
            x = j - center_voxel
            y = i - center_voxel
            for a in range(len(angles)):
                u = -x*np.cos(angles[a]) + y*np.sin(angles[a])
                u *= scale

                pixel_index = round(u + center_pixel)

                image[i, j] += sinogram[a, pixel_index]

                #le défi est simplement géométrique;
                #pour chaque voxel, trouver la position par rapport au centre de la
                #grille de reconstruction et déterminer la position d'arrivée
                #sur le détecteur d'un rayon partant de ce point et atteignant
                #le détecteur avec un angle de 90 degrés. Vous pouvez utiliser
                #le pixel le plus proche ou interpoler linéairement...Rappel, le centre
                #du détecteur est toujours aligné avec le centre de la grille de
                #reconstruction peu importe l'angle.

    util.saveImage(image, "lam")


## reconstruire une image TDM en mode retroprojection filtrée
def backproject():
    
    [nbprj, angles, sinogram] = readInput()
    
    # initialiser une image reconstruite
    image = np.zeros((geo.nbvox, geo.nbvox))
    
    ### option filtrer ###
    CTfilter.filterSinogram(sinogram)
    ######
    center_voxel = (geo.nbvox - 1) / 2
    center_pixel = (geo.nbpix - 1) / 2
    scale = geo.voxsize / geo.pixsize
    # "etaler" les projections sur l'image
    # ceci sera fait de façon "voxel-driven"
    # pour chaque voxel, trouver la contribution du signal reçu
    for j in range(geo.nbvox): # colonnes de l'image
        print("working on image column: "+str(j+1)+"/"+str(geo.nbvox))
        for i in range(geo.nbvox): # lignes de l'image
            x = j - center_voxel
            y = i - center_voxel
            for a in range(len(angles)):
                u = -x*np.cos(angles[a]) + y*np.sin(angles[a])
                u *= scale

                pixel_index = round(u + center_pixel)

                #if 0 <= pixel_index < sinogram.shape[1]:
                image[i, j] += sinogram[a, pixel_index]
                #votre code ici...
                #le défi est simplement géométrique;
                #pour chaque voxel, trouver la position par rapport au centre de la
                #grille de reconstruction et déterminer la position d'arrivée
                #sur le détecteur d'un rayon partant de ce point et atteignant
                #le détecteur avec un angle de 90 degrés. Vous pouvez utiliser
                #le pixel le plus proche ou interpoler linéairement...Rappel, le centre
                #du détecteur est toujours aligné avec le centre de la grille de
                #reconstruction peu importe l'angle.
                
    util.saveImage(image, "fbp")


## reconstruire une image TDM en mode retroprojection
def reconFourierSlice():
    
    [nbprj, angles, sinogram] = readInput()
    dim = sinogram.shape[1] # nombre de projections et dimension de l'image finale
    P = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(sinogram, axes=1), axis=1), axes=1) # On fait la TF de
    # chaque projections et on centre le DC (Freq=0)
    P = P.ravel()
    
    # grille polaire dans l'espace de Fourier
    r = np.arange(dim) - dim / 2
    r, a = np.meshgrid(r, angles)
    r = r.ravel()
    a = a.ravel()
    srcx = (dim / 2) + r*np.cos(a)
    srcy = (dim / 2) + r*np.sin(a)
    
    # On construit une grille cartésienne qui servira pour l'interpolation
    dstx, dsty = np.meshgrid(np.arange(dim), np.arange(dim))
    dstx = dstx.ravel()
    dsty = dsty.ravel()
    
    # Interpolation par la méthode des "voisins proches" de la grille polaire vers la grille cartésienne 
    fft2 = sp.interpolate.griddata((srcx, srcy), P, (dstx, dsty), method="nearest").reshape((dim, dim))
    recon = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(fft2)))) # Retour dans l'espace temporel,
    # on fait la TF inverse 2D et on shift pour que le résultat s'affiche comme il faut
    image = np.rot90(recon.T, -1) # On réoriente l'image reconstruite
    util.saveImage(image, "fft")

def compareSinogram():
    [nbprj, angles, sinogram] = readInput()

    # S'assurer que le sinogramme est un tableau NumPy
    sinogram = np.array(sinogram, dtype=float)
    
    # S'assurer que le sinogramme est un tableau NumPy et créer une copie pour le filtrage
    sinogram = np.array(sinogram, dtype=float)
    filtered_sinogram = sinogram.copy()
    
    # Filtrer le sinogramme (la fonction modifie filtered_sinogram en place)
    CTfilter.filterSinogram(filtered_sinogram)
    
    # Afficher et comparer les sinogrammes
    plt.figure(figsize=(14, 6))
    
    # Sinogramme original
    plt.subplot(1, 2, 1)
    plt.imshow(sinogram, cmap='gray', aspect='auto')
    plt.title('Sinogramme original')
    plt.colorbar()
    
    # Sinogramme filtré
    plt.subplot(1, 2, 2)
    plt.imshow(filtered_sinogram, cmap='gray', aspect='auto')
    plt.title('Sinogramme filtré')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('comparison_sinograms.png')
    plt.show()
    
    print("Dimensions du sinogramme:", sinogram.shape)
    print("Comparaison des valeurs - Original: min =", np.min(sinogram), "max =", np.max(sinogram))
    print("Comparaison des valeurs - Filtré: min =", np.min(filtered_sinogram), "max =", np.max(filtered_sinogram))
start_time = time.time()

# laminogram()
# backproject()
# reconFourierSlice()
# compareSinogram()
# print("--- %s seconds ---" % (time.time() - start_time))

