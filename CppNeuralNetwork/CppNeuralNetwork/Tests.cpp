#include <cassert>
#include <iostream>
#include <string>

#define CATCH_CONFIG_RUNNER
#include "catch.hpp"

#include "NeuralNetwork.h"


void testCreationOnce(NNet::NeuralNetworkParameters params) {
	NNet::NeuralNetwork nn(params);

	REQUIRE(nn.getLayerNumber() == params.layers.size());
	for (size_t i = 0; i < params.layers.size(); i++)
	{
		REQUIRE(nn.getNeuronCount(i) == params.layers.at(i).nNeuron);
	}

	for (size_t i = 0; i < params.layers.size() - 1; i++)
	{
		REQUIRE(nn.getWeightCount(i) == params.layers.at(i).nNeuron * params.layers.at(i + 1).nNeuron);
	}
}

TEST_CASE("NeuralNets are created with correct values", "[NNet]") {
	std::string testName = "Creation & structure";
	using namespace NNet;
	NeuralNetworkParameters params;
	LayerDescription desc;


	desc.nNeuron = 2;
	params.layers.push_back(desc);
	desc.nNeuron = 3;
	params.layers.push_back(desc);
	desc.nNeuron = 1;
	params.layers.push_back(desc);
	testCreationOnce(params);
	params.layers.clear();

	desc.nNeuron = 1;
	params.layers.push_back(desc);
	desc.nNeuron = 1;
	params.layers.push_back(desc);
	testCreationOnce(params);
	params.layers.clear();

	desc.nNeuron = 100;
	params.layers.push_back(desc);
	desc.nNeuron = 50;
	params.layers.push_back(desc);
	desc.nNeuron = 30;
	params.layers.push_back(desc);
	desc.nNeuron = 10;
	params.layers.push_back(desc);
	testCreationOnce(params);
	params.layers.clear();

	desc.nNeuron = 10;
	params.layers.push_back(desc);
	desc.nNeuron = 20;
	params.layers.push_back(desc);
	desc.nNeuron = 30;
	params.layers.push_back(desc);
	desc.nNeuron = 20;
	params.layers.push_back(desc);
	desc.nNeuron = 10;
	params.layers.push_back(desc);
	desc.nNeuron = 5;
	params.layers.push_back(desc);
	testCreationOnce(params);
	params.layers.clear();

}

void testIOVectorsOnce(NNet::NeuralNetworkParameters params) {
	using namespace NNet;
	NeuralNetwork nn(params);

	NeuralNetwork::NeuronVector& in = nn.getInputs();
	NeuralNetwork::NeuronVector& out = nn.getOutputs();
	REQUIRE(in.size() == params.layers.front().nNeuron);
	REQUIRE(out.size() == params.layers.back().nNeuron);

	for (size_t i = 0; i < in.size(); i++)
	{
		in[i] = float(i);
		REQUIRE(float(i) == nn.getNeuronValue(0, i));
	}

	for (size_t i = 0; i < out.size(); i++)
	{
		out[i] = float(i);
		REQUIRE(float(i) == nn.getNeuronValue(params.layers.size() - 1, i));
	}

}

TEST_CASE("NeuralNets have correct input/output neurons", "[NNet]") {
	std::string testName = "Creation & structure";
	using namespace NNet;
	NeuralNetworkParameters params;
	LayerDescription desc;

	desc.nNeuron = 2;
	params.layers.push_back(desc);
	desc.nNeuron = 3;
	params.layers.push_back(desc);
	desc.nNeuron = 1;
	params.layers.push_back(desc);
	testIOVectorsOnce(params);
	params.layers.clear();

	desc.nNeuron = 1;
	params.layers.push_back(desc);
	desc.nNeuron = 1;
	params.layers.push_back(desc);
	testIOVectorsOnce(params);
	params.layers.clear();

	desc.nNeuron = 5;
	params.layers.push_back(desc);
	desc.nNeuron = 6;
	params.layers.push_back(desc);
	desc.nNeuron = 2;
	params.layers.push_back(desc);
	testIOVectorsOnce(params);
	params.layers.clear();
}


int main(int argc, char** argv) {
	int result = Catch::Session().run(argc, argv);
	std::getchar();

	return result;
}


