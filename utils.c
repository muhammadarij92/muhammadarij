#include <iostream>
#include <stdlib.h>
#include <stdio.h>


float *read_image(int size, const char* img_file)
{
	float *img;
	int channels = 3;

	img = new float [size*size*channels];

	FILE *fp = fopen(img_file, "r");
	
	if (fp == NULL)
	{
		std::cout<<"test image open failed!";
		exit(-1);
	}

	for(int channels=0; channels<3; channels++)
	{
		for(int i=0; i<size; i++)
		{
			for(int j=0; j<size; j++)
				fscanf(fp, "%f\n", img + i*size + j + channels*(size*size));
		}
	}

	return img;
}

float* read_bias(int out_channels, const char* bias_file)
{
	float *bias;

	bias = new float [out_channels];

	FILE *fp = fopen(bias_file, "r");

	if (fp == NULL)
	{
		std::cout<<"bias file open failed!";
		exit(-1);
	}

	for(int out_channel=0; out_channel<out_channels; out_channel++)
	{
		fscanf(fp, "%f\n", bias + out_channel);
	}

	return bias;
}

float *read_weights(int out_channels, int in_channels, int kernel_size, const char* weight_file)
{
	float *weights;
	weights = new float [out_channels*in_channels*kernel_size*kernel_size];

	int out_kernel_size = in_channels * kernel_size *kernel_size;

	FILE *fp = fopen(weight_file, "r");

	if (fp == NULL)
	{
		std::cout<<"weight file open failed!";
		exit(-1);
	}
	
	for(int out_channel=0; out_channel<out_channels; out_channel++)
	{
		for(int in_channel=0; in_channel<in_channels; in_channel++)
		{
			for(int i=0; i<kernel_size; i++)
			{
				for(int j=0; j<kernel_size; j++)
				{
					fscanf(fp, "%f\n", weights + kernel_size*i + j + in_channel*(kernel_size*kernel_size) + out_channel*out_kernel_size);
					//std::cout<<*(weights + kernel_size*i + j + in_channel*(kernel_size*kernel_size))<<"\n";
				}
			}
		}
	}

	return weights;
}


void print_weights(int out_channels, int in_channels, int kernel_size, float* weights)
{
	int out_kernel_size = in_channels * kernel_size * kernel_size;

	for(int out_channel=0; out_channel<out_channels; out_channel++)
	{
		std::cout<<"Output Channel: "<<out_channel+1<<"\n";

		for(int in_channel=0; in_channel<in_channels; in_channel++)
		{
			std::cout<<"Input Channel: "<<in_channel+1<<"\n";
			for(int i=0; i<kernel_size; i++)
			{
				for(int j=0; j<kernel_size; j++)
				{
					std::cout<<*(weights + kernel_size*i + j + in_channel*(kernel_size*kernel_size) + out_channel*out_kernel_size)<<" ";
				}
			std::cout<<"\n";
			}
		}
	}
}


void print_image(int channels, int rows, int cols, float* image)
{

	for(int channel=0; channel<channels; channel++)
	{
		std::cout<<"Channel: "<<channel+1<<"\n";

		for(int i=0; i<rows; i++)
		{
			for(int j=0; j<cols; j++)
			{
				std::cout<<*(image + i*rows + j + channel*(rows*cols))<<" ";
			}
		std::cout<<"\n";
		}
	}

	std::cout<<std::endl;
}

void print_bias(int out_channels, float* bias)
{
	for(int out_channel=0; out_channel<out_channels; out_channel++)
	{
		std::cout<<"Channel : "<<out_channel+1<<" , bias : "<<*(bias + out_channel)<<" \n";
	}
	std::cout<<"\n";

}

int get_post_maxPool_size(int pool_size, int input_size)
{
	return input_size/pool_size;
}

float* read_weights_fc(int rows, int cols, const char* weight_file)
{
	float* weights;
	weights = new float [rows*cols];

	FILE *fp = fopen(weight_file, "r");

	if (fp == NULL)
	{
		std::cout<<"Fully connected : weights file read failed!";
		exit(-1);
	}

	for(int i=0; i<rows; i++)
	{
		for(int j=0; j<cols; j++)
		{
			fscanf(fp, "%f\n",weights + i*cols + j);
		}
	}

	return weights;

}


void print_weights_fc(int rows, int cols, float* weights)
{
	for(int i=0; i<rows; i++)
	{
		std::cout<<"Row : "<<i+1<<"\n";
		for(int j=0; j<cols; j++)
		{
			std::cout<<*(weights + i*cols + j)<<" ";
		}
		std::cout<<std::endl;
	}
}

float* read_bias_fc(int rows, const char* bias_file)
{
	float* bias;
	bias = new float [rows];

	FILE *fp = fopen(bias_file, "r");

	if(fp == NULL)
	{
		std::cout<<"Fully connected : bias file read failed!";
		exit(-1);
	}

	for(int i=0; i<rows; i++)
	{
		fscanf(fp, "%f\n", bias + i);
	}

	return bias;
}

void print_bias_fc(int rows, float* bias)
{

	for(int i=0; i<rows; i++)
	{
		std::cout<<"Feature : "<<i+1<<" , Bias : "<<*(bias + i)<<"\n";
	}
}

void print_linear(int rows, float* vec)
{
	for(int i=0; i<rows;i++)
	{
		std::cout<<"Row: "<<i+1<<" , Value : "<<*(vec +i)<<"\n";
	}
}