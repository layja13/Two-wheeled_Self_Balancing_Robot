//This program stabilizes the position (angle) of a self-balancing robot through a standard PID controller (using Arduino libraries).
//It contains much less lines of code that the ones I had found on internet. 
//The code is organized and commented.

#include <PID_v1.h>
#include <LMotorController.h>
#include "I2Cdev.h"
#include "MPU6050_6Axis_MotionApps20.h"

//-----------MPU6050---------
// MPU object creation
MPU6050 mpu;

// Set up arrays where MPU data is stored
uint8_t fifoBuffer[64]; 
Quaternion q;           
VectorFloat gravity;    
float ypr[3];  

//-----------PID---------------
//PID gains
double Kp = 75, Ki = 650 , Kd = 2.1;   // Kp = 65, Ki = 300 , Kd = 1;  //Kp = 85, Ki = 300 , Kd = 0.5;  
//Input, Output and Setpoint 
double Input, Output, Setpoint = 88.4;
//Pid object with its parameters
PID pid(&Input, &Output, &Setpoint, Kp, Ki, Kd, DIRECT);

//-----------L298N------------
//Motor control parameters and hardware connection
int ena = 6;
int enb = 5;
int in1 = 8;
int in2 = 9;
int in3 = 10;
int in4 = 11;    
double SpeedRight = 1;
double SpeedLeft = 1;
//Motor object set up and constructor
LMotorController motores(ena , in1, in2, enb, in3, in4, SpeedLeft, SpeedRight);

void setup() {
  Serial.begin(115200);

  //-----------MPU6050---------
  //Testing MPU connection and initialization for general usage
   if(mpu.testConnection()){
     Serial.println("Successful connection");
     mpu.dmpInitialize();
     mpu.setDMPEnabled(true);
    }
   else {
     Serial.println("Failed connection");
    }

    //-----------PID---------------
  //Set up of PID digital parameters
  pid.SetMode(AUTOMATIC);
  pid.SetSampleTime(5);
  pid.SetOutputLimits(-255,255);
}

void loop() {
  //-----------MPU6050---------
  //MPU data obtained stored in fifoBuffer vector
  mpu.dmpGetCurrentFIFOPacket(fifoBuffer);
  //Sensing of Yaw, Pitch, Roll // error and Pitch shown in serial monitor
  mpu.dmpGetQuaternion(&q, fifoBuffer);
  mpu.dmpGetGravity(&gravity, &q);
  mpu.dmpGetYawPitchRoll(ypr, &q, &gravity);
  Serial.print("Pitch:\t");
  Serial.print(Input);
  Serial.print("          ");
  Serial.print("Error: ");
  Serial.print(Setpoint-Input);
  Serial.print("          ");
  Serial.print("Output: ");
  Serial.println(Output);
  
  //-----------PID---------------
  //The input is the first value of ypr array (PITCH) // Also, the input is scaled
  Input = ypr[1] * 90/M_PI;
  if (Input <0){
    Input = Input+180;
  }
  // PID library algorithm function (It determines the "Output" variable).
  Output = pid.Compute(); 
  
  //-----------L298N------------
  //Output PID controller signal (control signal) is equal to the signal the motors receive
  motores.move(Output, 70);
}
