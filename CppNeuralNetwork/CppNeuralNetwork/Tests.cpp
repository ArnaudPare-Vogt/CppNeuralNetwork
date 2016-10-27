#include <cassert>
#include <iostream>
#include <string>
#include <cmath>

#define CATCH_CONFIG_RUNNER
#include <Catch/catch.hpp>

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

	NeuralNetwork::Vector& in = nn.getInputs();
	NeuralNetwork::Vector& out = nn.getOutputs();
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

TEST_CASE("Neural network weight multiplication", "[NNet]") {
	using namespace NNet;
	SECTION("Test 1"){
		NeuralNetworkParameters params;
		LayerDescription desc;
		desc.nNeuron = 1;
		params.layers.push_back(desc);
		desc.nNeuron = 1;
		params.layers.push_back(desc);
		NeuralNetwork nn(params);
		nn.getInputs()[0] = 1;
		nn.setWeight(0, 0, 0, 0.2f);
		nn.calculateOutputs();
		REQUIRE(nn.getOutputs()[0] - tanh(1 * 0.2) < 0.0001f);

		nn.getInputs()[0] = 0.3f;
		nn.setWeight(0, 0, 0, 4);
		REQUIRE(nn.getOutputs()[0] - tanh(0.3 * 4) < 0.0001f);

		nn.setBias(1, 0, -1);
		nn.calculateOutputs();
		REQUIRE(nn.getOutputs()[0] - tanh((0.3 * 4) - 1) < 0.0001f);
	}

	SECTION("Test 2") {
		NeuralNetworkParameters params;
		LayerDescription desc;
		desc.nNeuron = 2;
		params.layers.push_back(desc);
		desc.nNeuron = 3;
		params.layers.push_back(desc);
		desc.nNeuron = 1;
		params.layers.push_back(desc);
		NeuralNetwork nn(params);

		nn.setWeight(0, 0, 0, 1.3f);
		nn.setWeight(0, 0, 1, 0.5f);
		nn.setWeight(0, 1, 0, 1.4f);
		nn.setWeight(0, 1, 1, 3.1f);
		nn.setWeight(0, 2, 0, -0.7f);
		nn.setWeight(0, 2, 1, 0.1f);

		nn.setWeight(1, 0, 0, 0.1f);
		nn.setWeight(1, 0, 1, 0.3f);
		nn.setWeight(1, 0, 2, 0.4f);

		nn.setBias(1, 0, 0.01f);
		nn.setBias(1, 1, -0.2f);
		nn.setBias(1, 2, 0.3f);

		nn.setBias(2, 0, 0.1f);

		NeuralNetwork::Vector& in = nn.getInputs();
		in[0] = 0.2f;
		in[1] = 0.7f;
		NeuralNetwork::Vector& out = nn.getOutputs();

		nn.calculateOutputs();
		REQUIRE(out[0] - 0.4921 < 0.001);
	}
}


int main(int argc, char** argv) {
	int result = Catch::Session().run(argc, argv);
	std::getchar();

	return result;
}


