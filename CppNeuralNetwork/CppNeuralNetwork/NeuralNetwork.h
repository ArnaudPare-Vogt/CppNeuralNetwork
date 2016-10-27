#pragma once

#include <vector>

#include <Eigen/Dense>

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
		typedef Eigen::MatrixXf Matrix;
		typedef Eigen::VectorXf Vector;
		typedef std::vector<Vector> BiasVectors;
		typedef std::vector<Vector> NeuronVectors;
		typedef std::vector<Matrix> WeightMaticies;
	private:

		WeightMaticies weightMatricies;
		NeuronVectors neuronVectors;
		BiasVectors biasVectors;

		Vector* inputNeurons;
		Vector* outputNeurons;

		Vector* expectedOutputVector = 0;

		float(*activation)(float) = &tanh;

		NeuronVectors simpleSummationValues;
		NeuronVectors errorVectors;

		static float tanhDerivative(float x) { return 1 - pow(tanh(x), 2); };
		float(*activationDerivative)(float) = &tanhDerivative;

	public:
		NeuralNetwork(const NeuralNetworkParameters& params);
		~NeuralNetwork();
	private:
		void generateNeurons(const NeuralNetworkParameters& params);
		void generateWeights(const NeuralNetworkParameters& params);

		void calculateValue(const size_t weightLayerNo, const size_t outputNeuronNo);
		void calculateLayer(const size_t outputLayerNo);

		void calculateBackPropLayer(const size_t outputLayerNo);
		void calculateBackPropOutputs();

		void costDerivative(Vector& error);
		void backPropagateError();
	public:
		void calculateOutputs();
		void backPropagation(float changeRate);

		void setWeight(const size_t weightLayerNo, const size_t weightRow, const size_t weightCol, const float weight);
		void setBias(const size_t layerNo, const size_t neuronNo, const float bias);


		Vector& getInputs();
		Vector& getOutputs();

		float getNeuronValue(size_t layerNo, size_t neuronNo) const;

		size_t getLayerNumber() const;
		size_t getNeuronCount(size_t layerNo) const;
		size_t getWeightCount(size_t beforeLayerNo) const;
	};

};
