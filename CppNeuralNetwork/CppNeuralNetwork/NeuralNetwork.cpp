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

		for (size_t i = 0; i < nLayers; i++)
		{
			unsigned nNeuron = params.layers.at(i).nNeuron;
			std::vector<float> neurons;
			for (size_t j = 0; j < nNeuron; j++)
			{
				neurons.push_back(0);
			}
			neuronVectors.push_back(neurons);
		}
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
