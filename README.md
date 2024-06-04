# Two Wheeled Self Balancing Robot
This repo contains three independent (but connected) projects. The first one, is a simulation of the Two Wheeled Self Balancing Robot (TWSBR) using a Continuous Time Recurrent Neural Network (CTRNN) as a controller. Its perfomance is contrasted with that of a classical PID controller, both tuned by means of a Genetic Algorithm.

In the second project, an exploration of the project one is made from a different perspective; an Adaptive PID controller is carried out. This Adaptive PID controller was proposed in such a way that it combines two different Adaptive strategies in itself: the well-known Gain Schedulling strategy and an Adaptive Gain strategy. The Adaptive Gain strategy is based on a mechanism of mathematical rules that allows to modify the PID parameters online when environment is changing or the stabilization of the robot face difficulties. The Adaptive PID Controller was tested in three different scenarios.

The third project is the code for a real implementation of the TWSBR with a standard PID controller using an Arduino.

Project one and two uses the same genetic algorithm, and the same simulation environment, which is Pybullet. Basically, everything (with the exception of the environment) is made from scratch.
