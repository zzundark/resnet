import os
import argparse
import datetime
import config as cfg
from inputs import input_pipeline 
from solver import solver 
from cnn1 import cnn1
from resnet import resnet


def main():
	input=input_pipeline()
	print("ready input pipeline")
	net=resnet()
	_solver=solver(net,input,'./log')
	_solver.train_and_test()
	
	


if __name__ == '__main__':
	main()