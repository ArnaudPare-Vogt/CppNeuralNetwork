#include "NeuralNetwork.h"

#include <cassert>
#include <cmath>

namespace NNet {

	NeuralNetwork::NeuralNetwork(const NeuralNetworkParameters& params)
	{
		generateNeurons(params);
		generateWeights(params);
	}

	NeuralNetwork::~NeuralNetwork() {
		if (expectedOutputVector != 0) {
			delete(expectedOutputVector);
		}
	}


	void NeuralNetwork::generateNeurons(const NeuralNetworkParameters& params) {
		unsigned nLayers = params.layers.size();
		assert(nLayers >= 2);

		neuronVectors.clear();
		biasVectors.clear();
		simpleSummationValues.clear();
		errorVectors.clear();
		if (expectedOutputVector != 0) {
			delete(expectedOutputVector);
		}

		for (size_t i = 0; i < nLayers; i++)
		{
			unsigned nNeuron = params.layers.at(i).nNeuron;
			Vector neurons(nNeuron);
			neuronVectors.push_back(neurons);
			biasVectors.push_back(neurons);
			simpleSummationValues.push_back(neurons);
			errorVectors.push_back(neurons);
		}

		inputNeurons = &(neuronVectors.at(0));
		outputNeurons = &(neuronVectors.at(nLayers - 1));
		expectedOutputVector = new Vector(outputNeurons->size());

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

			Matrix weights(nOutputNeurons, nInputNeurons);
			weights.setZero();
			weightMatricies.push_back(weights);
		}
	}






	void NeuralNetwork::calculateValue(const size_t weightLayerNo, const size_t outputNeuronNo) {
		Matrix& weightMatrix = weightMatricies[weightLayerNo];
		Vector& inputVector = neuronVectors[weightLayerNo];
		Vector& outputVector = neuronVectors[weightLayerNo + 1];

		size_t nInputs = inputVector.size();

		float& outNeuron = outputVector[outputNeuronNo];

		outNeuron = biasVectors[weightLayerNo + 1][outputNeuronNo];

		size_t weightBeginPos = nInputs * outputNeuronNo;
		for (size_t i = 0; i < nInputs; i++)
		{
			outNeuron += weightMatrix(outputNeuronNo, i) * inputVector[i];
		}

		outNeuron = tanh(outNeuron);
	}

	void NeuralNetwork::calculateLayer(const size_t outputLayerNo) {
		Vector& out = neuronVectors[outputLayerNo];
		Vector& in = neuronVectors[outputLayerNo - 1];
		Vector& bias = biasVectors[outputLayerNo];
		Matrix& weights = weightMatricies[outputLayerNo - 1];

		out = (weights*in + bias).unaryExpr(activation);
	}

	void NeuralNetwork::calculateOutputs() {
		for (size_t i = 1; i < neuronVectors.size(); i++)
		{
			calculateLayer(i);
		}
	}

	void NeuralNetwork::calculateBackPropLayer(const size_t outputLayerNo) {
		Vector& out = neuronVectors[outputLayerNo];
		Vector& in = neuronVectors[outputLayerNo - 1];
		Vector& bias = biasVectors[outputLayerNo];
		Matrix& weights = weightMatricies[outputLayerNo - 1];

		Vector& zValue = simpleSummationValues[outputLayerNo];
		zValue = weights*in + bias;
		out = zValue.unaryExpr(activation);

	}

	void NeuralNetwork::calculateBackPropOutputs() {
		for (size_t outputLayerNo = 1; outputLayerNo < neuronVectors.size(); outputLayerNo++)
		{
			calculateBackPropLayer(outputLayerNo);
		}
	}

	void NeuralNetwork::costDerivative(Vector& error) {
		for (size_t i = 0; i < outputNeurons->size(); i++)
		{
			error[i] = (*outputNeurons)[i] - (*expectedOutputVector)[i];
		}
	}

	void NeuralNetwork::backPropagateError() {
		Vector& last = errorVectors.back();
		last = ((*outputNeurons) - (*expectedOutputVector)).cwiseProduct(simpleSummationValues.back().unaryExpr(activationDerivative));
		for (size_t i = errorVectors.size() - 2; i > 0; i--)
		{
			Vector& error = errorVectors[i];
			Matrix& weights = weightMatricies[i];
			Vector& prevError = errorVectors[i + 1];
			Vector& simpleSummationVector = simpleSummationValues[i];

			error = (weights.transpose() * prevError).cwiseProduct(simpleSummationVector.unaryExpr(activationDerivative));
		}
	}

	void NeuralNetwork::backPropagation(float changeRate) {
		calculateBackPropOutputs();
		backPropagateError();
		
	}



	void NeuralNetwork::setWeight(const size_t weightLayerNo, const size_t weightRow, const size_t weightCol, const float weight) {
		weightMatricies[weightLayerNo](weightRow, weightCol) = weight;
	}

	void NeuralNetwork::setBias(const size_t layerNo, const size_t neuronNo, const float bias) {
		biasVectors[layerNo][neuronNo] = bias;
	}


	NeuralNetwork::Vector& NeuralNetwork::getInputs() {
		return *inputNeurons;
	}

	NeuralNetwork::Vector& NeuralNetwork::getOutputs() {
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
