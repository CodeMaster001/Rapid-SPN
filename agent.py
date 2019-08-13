# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 14:44:54 2019

@author: pjs37741
"""

import argparse

import boto3

import  model_loader





def build_arguments():

    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group()

    group.add_argument('-upload', '--upload',action='store_true')

    group.add_argument('-bucket', '--bucket_name')
     
    group.add_argument('-bucket_file', '--bucket_file')
    
    group.add_argument('-file_path', '--file_path', action='store_true')


    args = parser.parse_args()


if __name__ == '__main__':

    build_arguments()
    
    if args.upload:
        model_loader.upload_file(args.bucket,args.bucket_file,args.file_path)
        