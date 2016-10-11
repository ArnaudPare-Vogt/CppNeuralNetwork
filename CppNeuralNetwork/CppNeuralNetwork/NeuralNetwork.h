#pragma once
#include <cassert>
#include <vector>

namespace NNet {

	struct LayerDescription {
		unsigned nNeuron;
	};

	struct NeuralNetworkParameters {
		std::vector<LayerDescription> layers;
	};

	class NeuralNetwork
	{
	private:
		typedef std::vector<std::vector<float>> WeightMaticies;
		typedef std::vector<std::vector<float>> NeuronVectors;

		WeightMaticies weightMatricies;
		NeuronVectors neuronVectors;

	public:
		NeuralNetwork(const NeuralNetworkParameters& params);
	private:
		void generateNeurons(const NeuralNetworkParameters& params);
		void generateWeights(const NeuralNetworkParameters& params);
	public:
		size_t getLayerNumber() const;
		size_t getNeuronCount(size_t layerNo) const;
		size_t getWeightCount(size_t beforeLayerNo) const;
	};

};
