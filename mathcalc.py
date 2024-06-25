# Setup a bunch of global variables
# Eventually these can and should be moved to a config file or input arg for CLI style function
DragCoefficient = 0.205 # default Cd
MagnCoefficient = 0.001 # default lift coefficienet
AirDensity = 1.225      # kg/m^3
AirDynDens = 1.825e-5   # kg/m-s
SurfaceArea = 0.001330  # m^2 effective surface area (dimpled) 0.00133
Radius = 0.042672       # meters
Vx0 = 40                # m/s
Vy0 = 25                # m/s
Vz0 = 0                 # m/s
BackSpin = 104.719755   # rad/s Z axis spin
SideSpin = 14.719755    # rad/s Y axis spin this needs to be *-1 in the equations for convention
BarrelSp = 0            # rad/s X axis spin, basically not realistic but included for completeness
gravity = 9.8                 # m/s^2
m = 0.0453              # kg weight of golf ball
spindecay = 0.05        # 5% per second
SpDragRt = 0.0001       # Spin drag rate

import math
import numpy as np

def calcX(dt,vx,vy,vz):
    xresist = calcAirResist(BackSpin,vx)
    ymagnus = calcMagnus(BackSpin,-vy)  # Vy increases X drag on the ball
    zmagnus = calcMagnus(SideSpin,vz)
    dx = vx*dt # need to solve for dx
    vx = vx - xresist*dt/m + ymagnus*dt/m + zmagnus*dt/m
    return dx,vx

def calcY(dt,vy,vx,vz):
    yresist = calcAirResist(BackSpin,vy)
    xmagnus = calcMagnus(BackSpin,vx)
    zmagnus = calcMagnus(BarrelSp,-vz)
    
    dy = vy*dt  # need to solve for dy
    vy = vy - gravity*dt - yresist*dt/m + xmagnus*dt/m + zmagnus*dt/m
    return dy,vy

def calcZ(dt,vz,vx,vy):
    zresist = calcAirResist(BackSpin,vz)
    xmagnus = calcMagnus(SideSpin,vx)
    ymagnus = calcMagnus(BarrelSp,vy)
    dz = vz*dt
    vz = vz - zresist*dt/m - xmagnus*dt/m - ymagnus*dt/m
    return dz,vz

def calcAirResist(sp1,Velocity):
    calcDragCoef(sp1,Velocity)
    airResistance = 0.5 * DragCoefficient * AirDensity * SurfaceArea * Velocity**2
    return airResistance

def calcDragCoef(sp1,Velocity):
    global DragCoefficient
    reynolds = AirDensity * abs(Velocity) * (2 * Radius) / AirDynDens
    # At High velocities there will be less drag
    InitialDrag = math.exp(-0.0000054*reynolds)
    SpinDrag =  SpDragRt * sp1
    DragCoefficient = InitialDrag + SpinDrag
    # print(DragCoefficient)

def calcMagnus(sp1,Velocity):
    gmagnus = math.pi * Radius**2 * sp1 * 1.7 # Turbulent air factor
    Mforce = AirDensity * Velocity * gmagnus * Radius**2 
    return Mforce

def calcSpinDecay(dt):
    global BackSpin
    global SideSpin

    BackSpin = BackSpin - BackSpin*spindecay*dt
    SideSpin = SideSpin - SideSpin*spindecay*dt

def updateGlobals(ballspeed,launchangle,sprate,sangle,sspinr):
    global Vy0
    global Vx0
    global Vz0
    global BackSpin
    global SideSpin

    Vx0 = ballspeed * 0.44704 * math.cos(math.radians(launchangle)) # Convert mph to m/s
    Vy0 = ballspeed * 0.44704 * math.sin(math.radians(launchangle)) # Convert mph to m/s
    Vz0 = ballspeed * 0.44704 * math.sin(math.radians(sangle))
    BackSpin = sprate * 2 * math.pi / 60 # Convert RPM to rad/sec
    SideSpin = sspinr * 2 * math.pi / 60 # Convert RPM to rad/sec

def getInitialV():
    return Vx0,Vy0,Vz0
# def getInitialV(bspeed, langle, sprate, sangle, sspinr):
#     # Example calculation - adjust according to the actual physics equations you need
#     InitVx = bspeed * np.cos(np.radians(langle))
#     InitVy = bspeed * np.sin(np.radians(langle))
#     InitVz = sspinr  # This is simplistic; adjust as necessary
#     return InitVx, InitVy, InitVz


def meterstoyards(meter_array):
    yard_array = [element * 1.09361 for element in meter_array] # Convert meters to yards
    return yard_array

def yardstofeet(yard_array):
    ft_array = [element * 3 for element in yard_array] # Convert yards to meters
    return ft_array

def simulate_trajectory(bspeed, langle, sprate, sangle, sspinr):
    v = bspeed * 0.44704  # Convert mph to m/s
    theta = np.radians(langle)  # Convert degrees to radians
    phi = np.radians(sangle)  # Shape angle in radians

    g = 9.81  # Gravity in m/s^2
    t = np.linspace(0, 8, num=500)  # Simulate for 8 seconds
    dt = t[1] - t[0]

    # Initial velocities
    vx = v * np.cos(theta) * np.cos(phi)
    vy = v * np.sin(theta)
    vz = v * np.cos(theta) * np.sin(phi)

    # Magnus force simplification (not realistic, illustrative only)
    magnus_factor = 0.0001
    mx = magnus_factor * sprate * vz  # Effect of back spin on lateral deflection
    my = magnus_factor * sspinr * vx  # Effect of side spin on height
    mz = magnus_factor * sprate * vx  # Effect of back spin on height

    # Positions
    x = vx * t + mx * t**2
    y = vy * t - 0.5 * g * t**2 + my * t**2
    z = vz * t + mz * t**2

    return x, y, z

if __name__ == '__main__':
    print("Test")