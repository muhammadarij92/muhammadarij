#include <iostream>
#include <string>
#include <fstream>
#include <stdexcept>
using namespace std;

class FullyConnect
{
	public:
		int out_features;
		int in_features;
		string *layerName;
		string *weightFilePath;

		FILE *filePtr;

	public:
		FullyConnect(string weightFilePath)
		{
			this->weightFilePath = new string(weightFilePath);
			this->layerName = NULL;

			bool status = readFile();
			
			if(status == 0)
				throw std::invalid_argument("read failed, please make sure you are provding correct file path...");
			else
				cout<<"Pointer to file "<<*(this->weightFilePath)<<" opened successfully..."<<endl;
			
			// Get layer name
			parseLayerName();

			// Get kernel dimensions
			parseKernelDimensions();

			// Allocate space to hold weights
			allocateSpace();

			// Parse weights value into array
			parseWeights();

			// Parse biases value into array
			parseBiases();

		}

		~FullyConnect()
		{
			delete this->layerName;
			delete this->weightFilePath;

			deallocateSpace();
		}

		void layerSummary()
		{
			cout<<"Layer Name : "<<*(this->layerName)<<endl;
			cout<<"Input Features : "<<this->in_features<<endl;
			cout<<"Output Features : "<<this->out_features<<endl;
		}


		double **weights_fully;
		double *biases_fully;


	protected:

		void allocateSpace()
		{	
			/* Allocate Space for weights */

			// number of channels x width x height x depth
			this->weights_fully = new double*[this->out_features];
			for(int channel=0; channel<(this->out_features); channel++)
			{
				this->weights_fully[channel] = new double[this->in_features];
			}

			/* Allocate space for biases */
			this->biases_fully = new double[this->out_features];
		}

		void deallocateSpace()
		{
			/* Deallocate space of weights */
			for(int channel=0; channel<(this->out_features); channel++)
			{

				delete[] this->weights_fully[channel];
			}

			delete[] this->weights_fully;

			/* Deallocates space of biases */
			delete[] this->biases_fully;
		}

		bool readFile()
		{	

			this->filePtr = fopen(this->weightFilePath->c_str(), "r");
			if(this->filePtr == 0)
				return false;
			else
				return true;
		}

		void parseLayerName()
		{
			char tmp[100];
			fscanf(this->filePtr, "%s\n", tmp);
			this->layerName = new string(tmp);
		}

		void parseKernelDimensions()
		{
			fscanf(this->filePtr, "%d %d\n", &this->in_features,  &this->out_features);
		}

		void parseWeights()
		{
			for(int channel=0; channel<(this->out_features); channel++)
			{
				for(int width=0; width<(this->in_features); width++)
				
					fscanf(this->filePtr, "%lf ", &weights_fully[channel][width]);
			}
			fscanf(this->filePtr, "\n");
		}

		void parseBiases()
		{
			for(int channel=0; channel<(this->out_features); channel++)
				fscanf(this->filePtr,"%lf ", &biases_fully[channel]);
		}
};
/*
int main()
{
	string weightFilePath("dense_16.txt");
	FullyConnect layer2(weightFilePath);
	layer2.layerSummary();

	
	for(int i=0; i<layer2.out_features; i++)
	{
		for (int j =0; j<layer2.in_features; j++)
			cout<<layer2.weights_fully[i][j]<<" ";
	}


	
	for(int i=0; i<layer2.out_features; i++)
		cout<<layer2.biases_fully[i]<<" ";
		//cout<<layer1.weights[i]<<"\n ";
	cout<<endl;
	

	return 0;
}*/
