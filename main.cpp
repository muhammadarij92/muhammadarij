#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include "model.h"

#include "Utils/utils.h"
#include "weight_access.cpp"
#include "weighhts_access_fully_connected.cpp"



static const char* input_imagePath = "images/ship.txt";

std::vector<std::string> split(const std::string& s, char c) {
  std::vector<std::string> v;
  unsigned int i = 0;
  unsigned int j = s.find(c);
  while (j < s.size() ){
    v.push_back(s.substr(i, j - i));
    i = ++j;
    j = s.find(c, j);
    if (j >= s.size() ) {
      v.push_back(s.substr(i,s.size() ));
      break;
    }
  }
  return v;
}
int main()
{
	float* hInputImage;
    float *w[NUMBER_WEIGHTS];
    for(int i=0;i<NUMBER_WEIGHTS;i++){
      w[i] = new(nothrow) float [weights_number[i]];
    if (!w)
    {
	    cout<<"allocaion failed"<<"";
	    cout<<"\n i="<<i;
  }} 
  // int  l = sizeof(w[0][0]);
   // cout<<"\n"<<l;
    
  std::cout<<"Loading weights into DDR memory"<<std::endl;
 // std::ifstream in("weights_vgg16_cifar10.dat",std::ios_base::binary);
  std::cout<<"Initializing weight buffers for  each layers"<<std::endl;

 /*  for (int i=0;i<NUMBER_WEIGHTS;i++)
    {
      in.read((char *)w[i],sizeof(float)*weights[i]);
    }
*/


	/* ------------------------------------- OpenCL Setup Code ------------------------------------- */
	
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	std::vector<cl::Device> devices;
	platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

	cl::Context context(devices);

	cl::CommandQueue queue = cl::CommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

	float *output_buffer = new float [OUTPUT_BUFFER_SIZE];
	float *input_buffer = new float [INPUT_BUFFER_SIZE];
    for (int i =0;i<OUTPUT_BUFFER_SIZE;i++)
    {
    	output_buffer[i] = 0;
    }
    
   std::fstream inputi;
    std::fstream input_label;
    inputi.open("test_images_data.dat");
    input_label.open("test_images_label.dat");

    std::string lincA;
    std::string lincB;
     int count = 0;
     int testcases = 5;
    for (int i = 0;i<testcases;i++)
    {
    getline(inputi, lincA);
    getline(input_label,lincB);
    std::vector<std::string> listfilemax = split(lincA,',');
    int test_label = atof(lincB.c_str());
    for (int l=0; l<32*32*3; l++)
        {
            input_buffer[l]= atof(listfilemax.at(l).c_str());

        }

	int c = 0;
       int zz = 0;
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////       
 for(int layer=0; layer< 9; layer++)
 { c=0;	 
   if(layer==0){	 
       string weightFilePath("conv2d_1.txt");
        Conv2D layer1(weightFilePath);
        layer1.layerSummary();
        for(int a=0; a < layer1.numChannels; a++)
        {
        for(int i=0; i<layer1.kernelDepth; i++)
        {
                for(int j=0; j<layer1.kernelWidth; j++)
                
		{
	
                        for(int k=0; k<layer1.kernelHeight; k++)
                                {
                                        w[0][c]= layer1.weights[a][i][j][k];
        //                             	cout<<w[0][c]<<" ";
                                        c++;}
				    // cout<<w[0][zz]<<" ";
				zz++;
                     //   cout<<layer1.weights[1][i][j][k]<<" ";
              }
      }
      }
         cout<< "\nc at w[0]= "<<c;
         c=0;
        for(int i=0; i< layer1.numChannels; i++)
        {
                w[1][c]= layer1.biases[i];
		c++;
	}
	         cout<< "\nc at w[1]= "<<c;

   }

   else if(layer==1){
       string weightFilePath("conv2d_2.txt");
        Conv2D layer1(weightFilePath);
        layer1.layerSummary();
        for(int a=0; a < layer1.numChannels; a++)
        {
        for(int i=0; i<layer1.kernelDepth; i++)
        {
                for(int j=0; j<layer1.kernelWidth; j++)

                {

                        for(int k=0; k<layer1.kernelHeight; k++)
                                {
                                        w[2][c]= layer1.weights[a][i][j][k];
          //                              cout<<w[2][c]<<" ";
                                        c++;}
                                    // cout<<w[0][zz]<<" ";
                                zz++;
                     //   cout<<layer1.weights[1][i][j][k]<<" ";
              }
      }
      }
	cout<< "\nw[2]="<<c;

         c=0;
        for(int i=0; i< layer1.numChannels; i++)
        {
                w[3][c]= layer1.biases[i];
                c++;
        }
	cout << "\n c at w[2]="<< c;
   }



   else if(layer==2){
       string weightFilePath("conv2d_3.txt");
        Conv2D layer1(weightFilePath);
        layer1.layerSummary();
        for(int a=0; a < layer1.numChannels; a++)
        {
        for(int i=0; i<layer1.kernelWidth; i++)
        {
                for(int j=0; j<layer1.kernelHeight; j++)

                {

                        for(int k=0; k<layer1.kernelDepth; k++)
                                {
                                        w[4][c]= layer1.weights[a][i][j][k];
            //                            cout<<w[4][c]<<" ";
                                        c++;}
                                    // cout<<w[0][zz]<<" ";
                                zz++;
                     //   cout<<layer1.weights[1][i][j][k]<<" ";
              }
      }
      }
	cout<<"\nw[4]="<<c;

         c=0;
        for(int i=0; i< layer1.numChannels; i++)
        {
                w[5][c]= layer1.biases[i];
                c++;
        }
           cout<<"\nw[5]="<<c;

   }


  
    

   else if(layer==3){
       string weightFilePath("conv2d_4.txt");
        Conv2D layer1(weightFilePath);
        layer1.layerSummary();
        for(int a=0; a < layer1.numChannels; a++)
        {
        for(int i=0; i<layer1.kernelWidth; i++)
        {
                for(int j=0; j<layer1.kernelHeight; j++)

                {

                        for(int k=0; k<layer1.kernelDepth; k++)
                                {
                                        w[6][c]= layer1.weights[a][i][j][k];
              //                          cout<<w[6][c]<<" ";
                                        c++;}
                                    // cout<<w[0][zz]<<" ";
                                zz++;
                     //   cout<<layer1.weights[1][i][j][k]<<" ";
              }
      }
      }
	cout<<"\n w[6]="<<c;
         c=0;
        for(int i=0; i< layer1.numChannels; i++)
        {
                w[7][c]= layer1.biases[i];
                c++;
        }
	        cout<<"\n w[7]="<<c;

   }



    



   else if(layer==4){
       string weightFilePath("conv2d_5.txt");
        Conv2D layer1(weightFilePath);
        layer1.layerSummary();
        for(int a=0; a < layer1.numChannels; a++)
        {
        for(int i=0; i<layer1.kernelWidth; i++)
        {
                for(int j=0; j<layer1.kernelHeight; j++)

                {

                        for(int k=0; k<layer1.kernelDepth; k++)
                                {
                                        w[8][c]= layer1.weights[a][i][j][k];
                //                        cout<<w[8][c]<<" ";
                                        c++;}
                                    // cout<<w[0][zz]<<" ";
                                zz++;
                     //   cout<<layer1.weights[1][i][j][k]<<" ";
              }
      }
      }

         c=0;
        for(int i=0; i< layer1.numChannels; i++)
        {
                w[9][c]= layer1.biases[i];
                c++;
        }}


     



   else if(layer==5){
       string weightFilePath("conv2d_6.txt");
        Conv2D layer1(weightFilePath);
        layer1.layerSummary();
        for(int a=0; a < layer1.numChannels; a++)
        {
        for(int i=0; i<layer1.kernelWidth; i++)
        {
                for(int j=0; j<layer1.kernelHeight; j++)

                {

                        for(int k=0; k<layer1.kernelDepth; k++)
                                {
                                        w[10][c]= layer1.weights[a][i][j][k];
                  //                      cout<<w[10][c]<<" ";
                                        c++;}
                                    // cout<<w[0][zz]<<" ";
                                zz++;
                     //   cout<<layer1.weights[1][i][j][k]<<" ";
              }
      }
      }

         c=0;
        for(int i=0; i< layer1.numChannels; i++)
        {
                w[11][c]= layer1.biases[i];
                c++;
        }}


    


/*   else if(j==6){
       string weightFilePath("conv2d_7.txt");
        Conv2D layer1(weightFilePath);
        layer1.layerSummary();
        for(int a=0; a < layer1.numChannels; a++)
        {
        for(int i=0; i<layer1.kernelWidth; i++)
        {
                for(int j=0; j<layer1.kernelHeight; j++)

                {

                        for(int k=0; k<layer1.kernelDepth; k++)
                                {
                                        w[12][c]= layer1.weights[a][i][j][k];
                    //                    cout<<w[12][c]<<" ";
                                        c++;}
                                    // cout<<w[0][zz]<<" ";
                                zz++;
                     //   cout<<layer1.weights[1][i][j][k]<<" ";
              }
      }
      }

         c=0;
        for(int i=0; i< layer1.numChannels; i++)
        {
                w[13][c]= layer1.biases[i];
                c++;
        }}


     



   else if(j==7){
       string weightFilePath("conv2d_8.txt");
        Conv2D layer1(weightFilePath);
        layer1.layerSummary();
        for(int a=0; a < layer1.numChannels; a++)
        {
        for(int i=0; i<layer1.kernelWidth; i++)
        {
                for(int j=0; j<layer1.kernelHeight; j++)

                {

                        for(int k=0; k<layer1.kernelDepth; k++)
                                {
                                        w[14][c]= layer1.weights[a][i][j][k];
                      //                  cout<<w[14][c]<<" ";
                                        c++;}
                                    // cout<<w[0][zz]<<" ";
                                zz++;
                     //   cout<<layer1.weights[1][i][j][k]<<" ";
              }
      }
      }

         c=0;
        for(int i=0; i< layer1.numChannels; i++)
        {
                w[15][c]= layer1.biases[i];
                c++;
        }}


     


   else if(j==8){
       string weightFilePath("conv2d_9.txt");
        Conv2D layer1(weightFilePath);
        layer1.layerSummary();
        for(int a=0; a < layer1.numChannels; a++)
        {
        for(int i=0; i<layer1.kernelWidth; i++)
        {
                for(int j=0; j<layer1.kernelHeight; j++)

                {

                        for(int k=0; k<layer1.kernelDepth; k++)
                                {
                                        w[16][c]= layer1.weights[a][i][j][k];
                        //                cout<<w[16][c]<<" ";
                                        c++;}
                                    // cout<<w[0][zz]<<" ";
                            //    zz++;
                     //   cout<<layer1.weights[1][i][j][k]<<" ";
              }
      }
      }

         c=0;
        for(int i=0; i< layer1.numChannels; i++)
        {
                w[17][c]= layer1.biases[i];
                c++;
        }}


  


   else if(j==9){
       string weightFilePath("conv2d_10.txt");
        Conv2D layer1(weightFilePath);
        layer1.layerSummary();
        for(int a=0; a < layer1.numChannels; a++)
        {
        for(int i=0; i<layer1.kernelWidth; i++)
        {
                for(int j=0; j<layer1.kernelHeight; j++)

                {

                        for(int k=0; k<layer1.kernelDepth; k++)
                                {
                                        w[18][c]= layer1.weights[a][i][j][k];
                                       // cout<<w[18][c]<<" ";
                                        c++;}
                                    // cout<<w[0][zz]<<" ";
                                zz++;
                     //   cout<<layer1.weights[1][i][j][k]<<" ";
              }
      }
      }

         c=0;
        for(int i=0; i< layer1.numChannels; i++)
        {
                w[19][c]= layer1.biases[i];
                c++;
        }}



   else if(j==10){
       string weightFilePath("conv2d_9.txt");
        Conv2D layer1(weightFilePath);
        layer1.layerSummary();
        for(int a=0; a < layer1.numChannels; a++)
        {
        for(int i=0; i<layer1.kernelWidth; i++)
        {
                for(int j=0; j<layer1.kernelHeight; j++)

                {

                        for(int k=0; k<layer1.kernelDepth; k++)
                                {
                                        w[20][c]= layer1.weights[a][i][j][k];
                                      //  cout<<w[20][c]<<" ";
                                        c++;}
                                    // cout<<w[0][zz]<<" ";
                                zz++;
                     //   cout<<layer1.weights[1][i][j][k]<<" ";
              }
      }
      }

         c=0;
        for(int i=0; i< layer1.numChannels; i++)
        {
                w[21][c]= layer1.biases[i];
                c++;
        }}


    

   else if(j==11){
       string weightFilePath("conv2d_9.txt");
        Conv2D layer1(weightFilePath);
        layer1.layerSummary();
        for(int a=0; a < layer1.numChannels; a++)
        {
        for(int i=0; i<layer1.kernelWidth; i++)
        {
                for(int j=0; j<layer1.kernelHeight; j++)

                {

                        for(int k=0; k<layer1.kernelDepth; k++)
                                {
                                        w[22][c]= layer1.weights[a][i][j][k];
                                       // cout<<w[22][c]<<" ";
                                        c++;}
                                    // cout<<w[0][zz]<<" ";
                                zz++;
                     //   cout<<layer1.weights[1][i][j][k]<<" ";
              }
      }
      }

         c=0;
        for(int i=0; i< layer1.numChannels; i++)
        {
                w[23][c]= layer1.biases[i];
                c++;
        }}




   else if(j==12){
       string weightFilePath("conv2d_9.txt");
        Conv2D layer1(weightFilePath);
        layer1.layerSummary();
        for(int a=0; a < layer1.numChannels; a++)
        {
        for(int i=0; i<layer1.kernelWidth; i++)
        {
                for(int j=0; j<layer1.kernelHeight; j++)

                {

                        for(int k=0; k<layer1.kernelDepth; k++)
                                {
                                        w[24][c]= layer1.weights[a][i][j][k];
                                      //  cout<<w[24][c]<<" ";
                                        c++;}
                                    // cout<<w[0][zz]<<" ";
                                zz++;
                     //   cout<<layer1.weights[1][i][j][k]<<" ";
              }
      }
      }


	
         c=0;
        for(int i=0; i< layer1.numChannels; i++)
        {
                w[25][c]= layer1.biases[i];
                c++;
        }}


 
*/


else  if(layer == 6){
  string weightFilePath("dense_1.txt");
        FullyConnect layer2(weightFilePath);
        layer2.layerSummary();


        for(int i=0; i<layer2.out_features; i++)
        {
                for (int j =0; j<layer2.in_features; j++)
                        w[12][c]= layer2.weights_fully[i][j];
                //        cout<<layer2.weights_fully[i][j]<<" ";
		c++;
        }


        c = 0;
        for(int i=0; i<layer2.out_features; i++){
                w[13][c] = layer2.biases_fully[i];c++;
     //           cout<<layer2.biases_fully[i]<<" ";
	}
                //cout<<layer1.weights[i]<<"\n ";
        cout<<endl;
}



else if (layer == 7){
  string weightFilePath("dense_2.txt");
        FullyConnect layer2(weightFilePath);
        layer2.layerSummary();


        for(int i=0; i<layer2.out_features; i++)
        {
                for (int j =0; j<layer2.in_features; j++){
                        w[14][c]= layer2.weights_fully[i][j];
                      //        cout<<"\nc="<<c ;
                        c++;
                        //  cout<<layer2.weights_fully[i][j]<<"\n\n\n ";
                }
        }
         cout<<"\nc at w[14]="<<c;

        c = 0;
        for(int i=0; i<layer2.out_features; i++){
                w[15][c] = layer2.biases_fully[i];c++;
   //             cout<<layer2.biases_fully[i]<<" ";
                //cout<<layer1.weights[i]<<"\n ";
        }
        cout<<"\n c at w[15]="<<c;

        cout<<endl;
}

//cout<<"\n c at w[17]="<<c;
else if (layer == 8){
  string weightFilePath("dense_3.txt");
        FullyConnect layer2(weightFilePath);
        layer2.layerSummary();


        for(int i=0; i<layer2.out_features; i++)
        {
                for (int j =0; j<layer2.in_features; j++){
			w[16][c]= layer2.weights_fully[i][j];
                     //	cout<<"\nc="<<c ;
		      	c++; 
			//  cout<<layer2.weights_fully[i][j]<<"\n\n\n ";
		}
        }
        cout<<"\nc at w[16]="<<c;

	c = 0;
        for(int i=0; i<layer2.out_features; i++){
		w[17][c] = layer2.biases_fully[i];c++;
   //             cout<<layer2.biases_fully[i]<<" ";
                //cout<<layer1.weights[i]<<"\n ";
	}
	//cout<<"\n c at w[17]="<<c;

        cout<<endl;
}


/*
else if (layer == 8){
  string weightFilePath("dense_3.txt");
        FullyConnect layer2(weightFilePath);
        layer2.layerSummary();
     cout<<"loop 1\n"<<"";
           int d=0;
	   
	  // cout<<"see the blunder"<<w[16][678];
       for(int i=0; i<layer2.in_features; i++)
        {
                for (int j =0; j<layer2.out_features; j++){
                        w[16][d]= layer2.weights_fully[i][j];
		//	cout<<"\n"<<c;
                //       cout<<w[16][d]<<"\n ";
			d++;
			cout<<"\nd="<<d;
        }
	}
     cout<<"loop2\n"<<"";
        c = 0;
        for(int i=0; i<layer2.out_features; i++){
                w[17][c] = layer2.biases_fully[i];
               // cout<<layer2.biases_fully[i]<<" ";
                //cout<<layer1.weights[i]<<"\n ";
	}
        cout<<endl;
}*/






}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	cout<<"layers loop started"<<" ";   
 int weight_count = 0;
 for (int j =0;j<NUMBER_LAYERS;j++)
 {
   if (layer[j][0]==0)
   
   {
      	
	/* ------------------------------------ Layer 1 Starts ------------------------------------ */

	int in_channels, out_channels, kernel_size;
	in_channels = layer[j][1];
	out_channels = layer[j][2];
	kernel_size = layer[j][5];
	imgRows = layer[j][4];
	imgCols = layer[j][4];		
	std::cout<<"Performing Convolution  "<<j+1<<" "<<std::endl;
	//std::cout<<in_channels<<" "<<out_channels<<" "<<kernel_size<<" "<<imgRows;
	
  
	try
	{
		  cl::Buffer inputBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, in_channels*imgRows*imgCols*sizeof(float));
    cl::Buffer filterBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, in_channels*out_channels*kernel_size*kernel_size*sizeof(float));
    cl::Buffer biasBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, out_channels*sizeof(float));
    cl::Buffer outputBuffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, out_channels*imgRows*imgCols*sizeof(float));
    cl::Buffer in_channelsBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));
    cl::Buffer out_channelsBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));
    cl::Buffer kernelSizeBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));
    cl::Buffer imgRowsBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));
    cl::Buffer imgColsBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));

		queue.enqueueWriteBuffer(inputBuffer, CL_TRUE, 0, in_channels*imgRows*imgCols*sizeof(float), input_buffer);
		queue.enqueueWriteBuffer(filterBuffer, CL_TRUE, 0, in_channels*out_channels*kernel_size*kernel_size*sizeof(float), w[weight_count]);
		queue.enqueueWriteBuffer(biasBuffer, CL_TRUE, 0, out_channels*sizeof(float), w[weight_count + 1]);
		queue.enqueueWriteBuffer(outputBuffer, CL_TRUE, 0, out_channels*imgRows*imgCols*sizeof(float), output_buffer);
		queue.enqueueWriteBuffer(in_channelsBuffer, CL_TRUE, 0, sizeof(int), &in_channels);
		queue.enqueueWriteBuffer(out_channelsBuffer, CL_TRUE, 0, sizeof(int), &out_channels);
		queue.enqueueWriteBuffer(kernelSizeBuffer, CL_TRUE, 0, sizeof(int), &kernel_size);
		queue.enqueueWriteBuffer(imgRowsBuffer, CL_TRUE, 0, sizeof(int), &imgRows);
		queue.enqueueWriteBuffer(imgColsBuffer, CL_TRUE, 0, sizeof(int), &imgCols);

		std::ifstream sourceFile("cl_kernels/conv.cl");
        std::string sourceCode(
         std::istreambuf_iterator<char>(sourceFile),
         (std::istreambuf_iterator<char>()));
         cl::Program::Sources source(1,
         std::make_pair(sourceCode.c_str(),
         sourceCode.length() + 1));

     	cl::Program program = cl::Program(context, source);

     	program.build(devices);
     	
     	cl::Kernel kernel(program, "convolution");

     	kernel.setArg(0, out_channelsBuffer);
     	kernel.setArg(1, in_channelsBuffer);
     	kernel.setArg(2, kernelSizeBuffer);
     	kernel.setArg(3, inputBuffer);
     	kernel.setArg(4, filterBuffer);
     	kernel.setArg(5, biasBuffer);
     	kernel.setArg(6, outputBuffer);
     	kernel.setArg(7, imgRowsBuffer);
     	kernel.setArg(8, imgColsBuffer);

     	cl::NDRange global(imgCols, imgRows);
     	cl::NDRange local(2, 2);
      cl::Event event;
     	queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local,NULL,&event);
      queue.finish();
     	// Read data back
     	queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, out_channels*imgRows*imgCols*sizeof(float), output_buffer);
     cl_ulong time_start;
     cl_ulong time_end;
     
     event.wait();
    double total_time;
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &time_end); 
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &time_start);
    total_time = time_end - time_start;

/* Results */
std::cout << "Execution time in milliseconds for convolution layer " << total_time*1.0e-6f << std::endl;   
	}
	catch(cl::Error error)
	{
		std::cout << error.what() << "(" << error.err() << ")" <<std::endl;
	}
    weight_count = weight_count+2;



    
  
if (j==0 )  // or 1 or 3 or 4 or 6 or 7
{

    for (int p = 0;p<(out_channels*imgRows*imgCols);p++)   // Code to print output of the every convolution layer
          { 
               
           //  cout << output_buffer [p] <<"  ";
      }    
    
}



    
    for (int p = 0;p<(out_channels*imgRows*imgCols);p++)
          {

              input_buffer[p] = output_buffer[p];
      }

    }
   
    else if (layer[j][0]==1)
      {
  
	/* ------------------------------------ Layer 1 Ends ------------------------------------ */

	
	/* ------------------------------------ MaxPool 2D Starts ------------------------------------ */

	int channels, pool_size, outImgRows, outImgCols;
	channels = layer[j][1];
	imgRows = layer[j][3];
	imgCols = layer[j][3];
	pool_size = 2;

	outImgRows = get_post_maxPool_size(pool_size, imgRows);
	outImgCols = get_post_maxPool_size(pool_size, imgCols);
	for (int i =0;i<channels*outImgCols*outImgCols;i++)
 {
  output_buffer[i] = 0;
 }


	try
	{
		cl::Buffer inputBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, channels*imgRows*imgCols*sizeof(float));
		cl::Buffer outputBuffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, channels*outImgRows*outImgCols*sizeof(float));
		cl::Buffer channelsBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));
		cl::Buffer poolSizeBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));
		cl::Buffer inDimBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));
		cl::Buffer outDimBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));

		queue.enqueueWriteBuffer(inputBuffer, CL_TRUE, 0, channels*imgRows*imgCols*sizeof(float), input_buffer);
		queue.enqueueWriteBuffer(outputBuffer, CL_TRUE, 0, channels*outImgRows*outImgCols*sizeof(float), output_buffer);
		queue.enqueueWriteBuffer(channelsBuffer, CL_TRUE, 0, sizeof(int), &channels);
		queue.enqueueWriteBuffer(poolSizeBuffer, CL_TRUE, 0, sizeof(int), &pool_size);
		queue.enqueueWriteBuffer(inDimBuffer, CL_TRUE, 0, sizeof(int), &imgRows);
		queue.enqueueWriteBuffer(outDimBuffer, CL_TRUE, 0, sizeof(int), &outImgRows);

		std::ifstream sourceFile("cl_kernels/max_pool2d.cl");
      std::string sourceCode(
         std::istreambuf_iterator<char>(sourceFile),
         (std::istreambuf_iterator<char>()));
      cl::Program::Sources source(1,
         std::make_pair(sourceCode.c_str(),
         sourceCode.length() + 1));

     	cl::Program program = cl::Program(context, source);

     	program.build(devices);
     	
     	cl::Kernel kernel(program, "max_pool2d");

     	kernel.setArg(0, channelsBuffer);
     	kernel.setArg(1, inDimBuffer);
     	kernel.setArg(2, poolSizeBuffer);
     	kernel.setArg(3, outDimBuffer);
     	kernel.setArg(4, inputBuffer);
     	kernel.setArg(5, outputBuffer);

     	cl::NDRange global(outImgRows, outImgCols);
     	cl::NDRange local(1, 1);
     cl::Event event;
      queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local,NULL,&event);
      queue.finish();

     	queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, channels*outImgRows*outImgCols*sizeof(float), output_buffer);
         cl_ulong time_start;
     cl_ulong time_end;
     
     event.wait();
    double total_time;
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &time_end); 
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &time_start);
    total_time = time_end - time_start;

/* Results */
std::cout << "Execution time in milliseconds for maxpool layer " << total_time*1.0e-6f << std::endl;   

	}
	catch(cl::Error error)
	{
		std::cout << error.what() << "(" << error.err() << ")" <<std::endl;
	}
 for (int p = 0;p<(channels*outImgRows*outImgCols);p++)
          { 
               
              input_buffer[p] = output_buffer[p]; 
      }
   
      
    }

     else
     {

	/* ------------------------------------ Fully Connected 1 Starts ------------------------------------ */
	
	int in_features, out_features;
	in_features = layer[j][1];
	out_features = layer[j][2];

	int weight_count_full = 12;
	//std::cout<<"Performing Fully Connected "<<j<<" "<<std::endl;
	//
	cout << in_features<< "  ";

	try
	{
		cl::Buffer inputBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, in_features*sizeof(float));
		cl::Buffer outputBuffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, out_features*sizeof(float));
		cl::Buffer weightsBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, in_features*out_features*sizeof(float));
		cl::Buffer biasesBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, out_features*sizeof(float));
		cl::Buffer inFeaturesBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));
		cl::Buffer outFeaturesBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));

		queue.enqueueWriteBuffer(inputBuffer, CL_TRUE, 0, in_features*sizeof(float), input_buffer);
		queue.enqueueWriteBuffer(outputBuffer, CL_TRUE, 0, out_features*sizeof(float), output_buffer);
		queue.enqueueWriteBuffer(weightsBuffer, CL_TRUE, 0, in_features*out_features*sizeof(float), w[weight_count_full]);
		queue.enqueueWriteBuffer(biasesBuffer, CL_TRUE, 0, out_features*sizeof(float), w[weight_count_full + 1]);
		queue.enqueueWriteBuffer(inFeaturesBuffer, CL_TRUE, 0, sizeof(int), &in_features);
		queue.enqueueWriteBuffer(outFeaturesBuffer, CL_TRUE, 0, sizeof(int), &out_features);

		std::ifstream sourceFile("cl_kernels/relu_linear.cl");
      std::string sourceCode(
         std::istreambuf_iterator<char>(sourceFile),
         (std::istreambuf_iterator<char>()));
      cl::Program::Sources source(1,
         std::make_pair(sourceCode.c_str(),
         sourceCode.length() + 1));

     	cl::Program program = cl::Program(context, source);

     	program.build(devices);
     	
     	cl::Kernel kernel(program, "relu_linear");

     	kernel.setArg(0, inFeaturesBuffer);
     	kernel.setArg(1, outFeaturesBuffer);
     	kernel.setArg(2, inputBuffer);
     	kernel.setArg(3, weightsBuffer);
     	kernel.setArg(4, biasesBuffer);
     	kernel.setArg(5, outputBuffer);

     	cl::NDRange global(out_features, 1);
     	cl::NDRange local(1, 1);
     cl::Event event;
      queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local,NULL,&event);
      queue.finish();

     	queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, out_features*sizeof(float), output_buffer);
               cl_ulong time_start;
     cl_ulong time_end;
     
     event.wait();
    double total_time;
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &time_end); 
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &time_start);
    total_time = time_end - time_start;

/* Results */
std::cout << "Execution time in milliseconds for Fully Connected/Dense layer " << total_time*1.0e-6f << std::endl;   


	}
	catch(cl::Error error)
	{
		std::cout << error.what() << "(" << error.err() << ")" <<std::endl;
	
	}

if (j== 8 || j == 9 || j== 10)
{

    for (int p = 0;p<(out_features);p++)   // Code to print output of the every convolution layer
          {

        //     cout << input_buffer [p] <<"  ";
      }

}





     for (int n = 0; n < layer[j][1]; n++) {
              input_buffer[n] =  output_buffer[n];
       //       std::cout<<input_buffer[n]<<std::endl;
      }
    weight_count_full = weight_count_full + 2;  
    } 
}


  float max = input_buffer[0];
      int outputs = 0;
      for (int j = 0; j < 10; j++)
      {

          if (input_buffer[j] > max)
         {
            outputs = j;
            max = input_buffer[j];
        }
      }

      std::cout<<i<<"   Predicted "<<outputs<<" "<<"Expected" << test_label;
      if ( outputs != test_label )
      {
          std ::cout<<"      "<<"Mismatched";

      }
         else
       {
          
              count = count+1;
        
      }
     std::cout<<" Done"<<std::endl;
    

}
std::cout<<std::endl<<"Accuracy : "<<(((count)*100)/testcases); 
	return 0;

}
