#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import os, sys, os.path
import numpy as np

#Blue
image_lower_rgb_blue = np.array([85, 122, 122]) 
image_upper_rgb_blue = np.array([90, 255, 255])

#Red
image_lower_rgb_red = np.array([0, 90, 90])  
image_upper_rgb_red = np.array([15, 255, 255])

def filtro_de_cor(img_bgr, low_rgb, high_rgb):
    img = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2HSV)
    filtro = cv2.inRange(img, low_rgb, high_rgb)
    return filtro 

def mascara_or(filtro1, filtro2):
    filtro = cv2.bitwise_or(filtro1, filtro2)
    return filtro

def mascara_and(filtro1, filtro2):
     filtro = cv2.bitwise_and(filtro1, filtro2)
     return filtro

def desenha_cruz(img, cordX,cordY, tamanho, color):
     cv2.line(img,(cordX - tamanho,cordY),(cordX + tamanho,cordY),color,5)
     cv2.line(img,(cordX,cordY - tamanho),(cordX, cordY + tamanho),color,5)

def desenha_cruz2(img, cordX2 , cordY2, tamanho, color):
     cv2.line(img,(cordX2 - tamanho,cordY2),(cordX2 + tamanho,cordY2),color,5)
     cv2.line(img,(cordX2,cordY2 - tamanho),(cordX2, cordY2 + tamanho),color,5)    

def escreve_texto(img, text, origem, color):
     font = cv2.FONT_HERSHEY_SIMPLEX
     origem = (0,50)
     cv2.putText(img, str(text), origem, font,1,color,2,cv2.LINE_AA)

def desenha_linha(img,cordX,cordY,cordX2,cordY2):
    color = (128,128,0)
    cv2.line(img,(cordX , cordY),(cordX2 , cordY2),color,5)

def calculo(img,cordX2,cordY2,cordX,cordY,color):
    p1=(cordX2,cordY2)
    p2=(cordX,cordY)
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    angulo_retas = np.rad2deg((ang1 - ang2) % (2 * np.pi))
    text=int(angulo_retas)
    font = cv2.FONT_HERSHEY_SIMPLEX
    origem = (50,100)
    cv2.putText(img, str(text), origem, font,1,color,2,cv2.LINE_AA)
    
def webcam(img):
    filtro_rgb1 = filtro_de_cor(img, image_lower_rgb_red, image_upper_rgb_red)
    filtro_rgb2 = filtro_de_cor(img, image_lower_rgb_blue, image_upper_rgb_blue)
    
    filtro_rgb = mascara_or(filtro_rgb1, filtro_rgb2)
    
    contornos, _ = cv2.findContours(filtro_rgb, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

    filtro_rgb = cv2.cvtColor(filtro_rgb, cv2.COLOR_GRAY2RGB) 
    contornos_img = filtro_rgb.copy()
    
    maior = None
    segmentomaior = None
    maior_area = 0
    segunda_maior_area = 0
    for c in contornos:
        area = cv2.contourArea(c)
        if area > maior_area and area > 500:
            segunda_maior_area = maior_area
            maior_area = area
            segmentomaior = maior 
            maior = c
            
        elif area > segunda_maior_area and area > 500:
            segunda_maior_area = area
            segmentomaior = c
    
    M = cv2.moments(maior)
    M2 = cv2.moments(segmentomaior)

    if M["m00"] != 0:
        cordX = int(M["m10"] / M["m00"])
        cordY = int(M["m01"] / M["m00"])
        
        cv2.drawContours(contornos_img, [maior], -1, [255, 0, 0], 5)
        desenha_cruz(contornos_img, cordX,cordY, 20, (0,0,255))
        
        texto = cordY , cordX
        origem = (0,50)
 
        escreve_texto(contornos_img, texto, (0,200), (0,255,0)) 
    if M2["m00"] != 0:
        cordX2 = int(M2["m10"] / M2["m00"])
        cordY2 = int(M2["m01"] / M2["m00"])
        
        cv2.drawContours(contornos_img, [segmentomaior], -1, [255, 0, 0], 5)
       
        desenha_cruz2(contornos_img, cordX2,cordY2, 20, (0,0,255))

        texto = cordY2 , cordX2
        origem = (500,50)
        
        escreve_texto(contornos_img, texto, (50,150), (0,255,0))        
        desenha_linha(contornos_img,cordX,cordY,cordX2,cordY2)
        calculo(contornos_img,cordX2,cordY2,cordX,cordY,(0,255,0))
    else:
        cordX, cordY, cordX2 , cordY2 = 0, 0 , 0 , 0
        texto = 'nao tem nada'
        origem = (0,50)
        escreve_texto(contornos_img, texto, origem, (0,0,255))
    


    return contornos_img

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

vc.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False

while rval:
    
    img = webcam(frame)

    cv2.imshow("preview", img)
    cv2.imshow("original", frame)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27:
        break

cv2.destroyWindow("preview")
vc.release()