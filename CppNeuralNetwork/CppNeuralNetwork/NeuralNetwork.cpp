#include "NeuralNetwork.h"

namespace NNet {

	NeuralNetwork::NeuralNetwork(const NeuralNetworkParameters& params)
	{
		generateNeurons(params);
		generateWeights(params);
	}




	void NeuralNetwork::generateNeurons(const NeuralNetworkParameters& params) {
		unsigned nLayers = params.layers.size();
		assert(nLayers >= 2);

		neuronVectors.clear();
		biasVectors.clear();

		for (size_t i = 0; i < nLayers; i++)
		{
			unsigned nNeuron = params.layers.at(i).nNeuron;
			std::vector<float> neurons;
			for (size_t j = 0; j < nNeuron; j++)
			{
				neurons.push_back(0);
			}
			neuronVectors.push_back(neurons);
			biasVectors.push_back(neurons);
		}

		inputNeurons = &(neuronVectors.at(0));
		outputNeurons = &(neuronVectors.at(nLayers - 1));
	}

	void NeuralNetwork::generateWeights(const NeuralNetworkParameters& params) {
		unsigned nLayers = params.layers.size();
		assert(nLayers >= 2);

		weightMatricies.clear();

		for (size_t i = 0; i < nLayers - 1; ++i)
		{
			unsigned inputLayerNo = i;
			unsigned outputLayerNo = i + 1;

			unsigned nInputNeurons = params.layers.at(inputLayerNo).nNeuron;
			unsigned nOutputNeurons = params.layers.at(outputLayerNo).nNeuron;

			unsigned nWeight = nInputNeurons * nOutputNeurons;

			std::vector<float> weights;
			for (size_t j = 0; j < nWeight; j++)
			{
				weights.push_back(1);
			}
			weightMatricies.push_back(weights);
		}
	}






	void NeuralNetwork::calculateValue(const size_t weightLayerNo, const size_t outputNeuronNo) {
		WeightMatrix& weightMatrix = weightMatricies[weightLayerNo];
		NeuronVector& inputVector = neuronVectors[weightLayerNo];
		NeuronVector& outputVector = neuronVectors[weightLayerNo + 1];

		size_t nInputs = inputVector.size();

		float& outNeuron = outputVector[outputNeuronNo];

		outNeuron = biasVectors[weightLayerNo + 1][outputNeuronNo];

		size_t weightBeginPos = nInputs * outputNeuronNo;
		for (size_t i = 0; i < nInputs; i++)
		{
			outNeuron += weightMatrix[weightBeginPos + i] * inputVector[i];
		}

		outNeuron = tanh(outNeuron);
	}

	void NeuralNetwork::calculateLayer(const size_t outputLayerNo) {
		size_t nNeuronsOut = neuronVectors[outputLayerNo].size();
		for (size_t i = 0; i < nNeuronsOut; i++)
		{
			calculateValue(outputLayerNo - 1, i);
		}
	}

	void NeuralNetwork::calculateOutputs() {
		for (size_t i = 1; i < neuronVectors.size(); i++)
		{
			calculateLayer(i);
		}
	}





	void NeuralNetwork::setWeight(const size_t weightLayerNo, const size_t weightNo, const float weight) {
		weightMatricies[weightLayerNo][weightNo] = weight;
	}

	void NeuralNetwork::setBias(const size_t layerNo, const size_t neuronNo, const float bias) {
		biasVectors[layerNo][neuronNo] = bias;
	}


	NeuralNetwork::NeuronVector& NeuralNetwork::getInputs() {
		return *inputNeurons;
	}

	NeuralNetwork::NeuronVector& NeuralNetwork::getOutputs() {
		return *outputNeurons;
	}

	float NeuralNetwork::getNeuronValue(size_t layerNo, size_t neuronNo) const {
		return neuronVectors[layerNo][neuronNo];
	}

	size_t NeuralNetwork::getLayerNumber() const {
		return neuronVectors.size();
	}

	size_t NeuralNetwork::getNeuronCount(size_t layerNo) const {
		return neuronVectors.at(layerNo).size();
	}

	size_t NeuralNetwork::getWeightCount(size_t beforeLayerNo) const {
		return weightMatricies.at(beforeLayerNo).size();
	}
};
