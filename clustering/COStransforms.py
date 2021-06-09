# -*- coding: utf-8 -*-
import numpy as np
from math import sin, cos, pi, sqrt, atan2

## Function converts from WGS84 to ECEF
# @param lon Longitude in degrees (np.array, 1xN)
# @param lat Longitude in degrees (np.array, 1xN)
# @param h Height in meters (optional, default = 0)
# @return Matrix (3xN) containing ECEF values
def WGS84toECEF(lon,lat,h=0.):
    
    lon = np.multiply(lon, pi) / 180.0    
    lat = np.multiply(lat, pi) / 180.0
    
    a = 6378137.0
    eSquared = 6.69437999014*1e-3
    def N(phi):
        return np.multiply(a ,1 / np.sqrt(np.subtract(1.,np.multiply(eSquared,np.power(np.sin(phi),2)))))
    
    ECEF_X = np.multiply(np.add(N(lat),h) , np.multiply(np.cos(lat),np.cos(lon)))
    ECEF_Y = np.multiply(np.add(N(lat),h) , np.multiply(np.cos(lat),np.sin(lon)))
    ECEF_Z = np.multiply(np.add(np.multiply(N(lat) , (1. - eSquared)), h) , np.sin(lat))
    
    return np.array([ECEF_X, ECEF_Y, ECEF_Z]).reshape([3, lon.size])
    
## Function converts from ECEF to ENU
# @param tECEF Matrix (3xN) with ECEF coordinates
# @param tREF Local reference for origin of tangential plane (structured array)
# @param h Height in meters (optional, default = 0)    
# @return Matrix (3xN) containing ENU values
def ECEFtoENU(tECEF,tREF):
    
    ECEF_X = np.squeeze(np.asarray(tECEF[0,:]))
    ECEF_Y = np.squeeze(np.asarray(tECEF[1,:]))
    ECEF_Z = np.squeeze(np.asarray(tECEF[2,:]))
    
    lon = float(tREF['lon']) * pi / 180
    lat = float(tREF['lat']) * pi / 180   
    
    tRot = np.array([[-sin(lon), cos(lon), 0.],
            [-sin(lat)*cos(lon), -sin(lat)*sin(lon), cos(lat)], 
            [cos(lat)*cos(lon), cos(lat)*sin(lon), sin(lat)]])    
    
    tENU = np.dot(tRot, np.array([np.subtract(ECEF_X, tREF['ECEF'][0]), 
                                  np.subtract(ECEF_Y, tREF['ECEF'][1]), 
                                  np.subtract(ECEF_Z, tREF['ECEF'][2])]))
            
    return tENU
    
## Function converts from ENU to ECEF
# @param tENU Matrix (3xN) with ENU coordinates
# @param tREF Local reference for origin of tangential plane (structured array) 
# @return Matrix (3xN) containing tECEF values
def ENUtoECEF(tENU,tREF):
        
    lon = float(tREF['lon']) * pi / 180
    lat = float(tREF['lat']) * pi / 180
    
    tRot = np.array([[-sin(lon), -cos(lon)*sin(lat), cos(lon)*cos(lat)],
            [cos(lon), -sin(lon)*sin(lat), sin(lon)*cos(lat)],
            [0., cos(lat), sin(lat)]])
    
    dECEF = np.squeeze(np.asarray(np.dot(tRot,tENU)))

    if len(dECEF.shape) == 2:
        retECEF = np.add(dECEF,np.tile(tREF['ECEF'],(1,dECEF.shape[1])))
    else:
        retECEF = np.add(dECEF,tREF['ECEF'])
    
    return retECEF

## Function converts from ECEF to WGS84
# @param tECEF Matrix (3xN) with ECEF coordinates
# @return Matrix (3xN) containing WGS84 values
def ECEFtoWGS84(tECEF):
    a = 6378137.0
    eSquared = 6.69437999014*1e-3

    WGS84_lon = np.arctan2(tECEF[1,:],tECEF[0,:])
    WGS84_lat = np.zeros(tECEF.shape[1])
    WGS84_height = np.zeros(tECEF.shape[1])
    # iterative procedure    
    thsld = 1e-8
    
    for idx in np.arange(tECEF.shape[1]):
        p = np.sqrt(tECEF[0,idx]**2 + tECEF[1,idx]**2)
        z = tECEF[2,idx]
        phi_old = np.arctan2(z/p, 1 - eSquared)
        phi = np.Inf
        bContinue = True
        while bContinue:
            N = a / np.sqrt(1 - eSquared*np.sin(phi_old)**2)
            h = p / np.cos(phi_old) - N
            phi = np.arctan2(z/p, 1 - N/(N+h)*eSquared)
            
            if abs(phi_old - phi) < thsld:
                bContinue = False
            
            phi_old = phi
        
        WGS84_lat[idx] = phi
        WGS84_height[idx] = h
        
        if (np.remainder(idx,1000) == 0.) and (idx > 0):
            print('{0:1d} of {1:1d}'.format(idx,tECEF.shape[1]))
        
    
    
    WGS84 = np.array([np.multiply(WGS84_lon,180.0/pi), np.multiply(WGS84_lat,180.0/pi), WGS84_height])
    
    return WGS84
    
## Wrapper script to compute from ENU to WGS84 representation
# @param tENU Matrix (3xN) containing ENU values
# @param tREF Structured array containing geodetic and ECEF coordinates of 
#        referene point
# @return Matrix (3xN) with WGS84 coordinates
def ENUtoWGS84(tENU,tREF):
    tECEF = ENUtoECEF(tENU,tREF)
    return ECEFtoWGS84(np.asarray(tECEF).reshape((3, tECEF.shape[1])))
## Wrapper script to transform from WGS84 to ENU representation
# @param lon Longitude in degrees (np.array, 1xN)
# @param lat Latitude in degrees (np.array, 1xN)
# @param h Height in meters (np.array, 1xN; default = 0 )
# @param tREF Structured array containing geodetic and ECEF coordinates of 
#        referene point
# @return Matrix (3xN) with ENU coordinates
def WGS84toENU(lon,lat,tREF,h=0.):
    tECEF = WGS84toECEF(lon,lat,h)
    return ECEFtoENU(tECEF,tREF)