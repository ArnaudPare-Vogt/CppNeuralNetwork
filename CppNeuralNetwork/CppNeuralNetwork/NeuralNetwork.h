#pragma once

#include <cassert>
#include <cmath>
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
		typedef std::vector<float> BiasVector;
		typedef std::vector<float> NeuronVector;
		typedef std::vector<float> WeightMatrix;
		typedef std::vector<BiasVector> BiasVectors;
		typedef std::vector<NeuronVector> NeuronVectors;
		typedef std::vector<WeightMatrix> WeightMaticies;
	private:

		WeightMaticies weightMatricies;
		NeuronVectors neuronVectors;
		BiasVectors biasVectors;

		NeuronVector* inputNeurons;
		NeuronVector* outputNeurons;

	public:
		NeuralNetwork(const NeuralNetworkParameters& params);
	private:
		void generateNeurons(const NeuralNetworkParameters& params);
		void generateWeights(const NeuralNetworkParameters& params);

		void calculateValue(const size_t weightLayerNo, const size_t outputNeuronNo);
		void calculateLayer(const size_t outputLayerNo);
	public:
		void calculateOutputs();


		void setWeight(const size_t weightLayerNo, const size_t weightNo, const float weight);
		void setBias(const size_t layerNo, const size_t neuronNo, const float bias);


		NeuronVector& getInputs();
		NeuronVector& getOutputs();

		float getNeuronValue(size_t layerNo, size_t neuronNo) const;

		size_t getLayerNumber() const;
		size_t getNeuronCount(size_t layerNo) const;
		size_t getWeightCount(size_t beforeLayerNo) const;
	};

};
