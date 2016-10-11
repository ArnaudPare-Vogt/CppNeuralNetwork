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
	public:
		typedef std::vector<float> NeuronVector;
		typedef std::vector<std::vector<float>> WeightMaticies;
		typedef std::vector<NeuronVector> NeuronVectors;
	private:

		WeightMaticies weightMatricies;
		NeuronVectors neuronVectors;

		NeuronVector* inputNeurons;
		NeuronVector* outputNeurons;

	public:
		NeuralNetwork(const NeuralNetworkParameters& params);
	private:
		void generateNeurons(const NeuralNetworkParameters& params);
		void generateWeights(const NeuralNetworkParameters& params);
	public:
		NeuronVector& getInputs();
		NeuronVector& getOutputs();

		float getNeuronValue(size_t layerNo, size_t neuronNo) const;

		size_t getLayerNumber() const;
		size_t getNeuronCount(size_t layerNo) const;
		size_t getWeightCount(size_t beforeLayerNo) const;
	};

};
