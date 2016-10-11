#include <cassert>
#include <iostream>
#include <string>

#include "NeuralNetwork.h"


void testCreationOnce(std::ostream& out, std::string testName, int numberOfTest, NNet::NeuralNetworkParameters params) {
	NNet::NeuralNetwork nn(params);

	if (nn.getLayerNumber() != params.layers.size()) {
		out << "Test failed at " << testName << " #" << numberOfTest << ".1" << std::endl;
	}
	for (size_t i = 0; i < params.layers.size(); i++)
	{
		if (nn.getNeuronCount(i) != params.layers.at(i).nNeuron) {
			out << "Test failed at " << testName << " #" << numberOfTest << ".2" << std::endl;
		}
	}

	for (size_t i = 0; i < params.layers.size() - 1; i++)
	{
		if (nn.getWeightCount(i) != params.layers.at(i).nNeuron * params.layers.at(i + 1).nNeuron) {
			out << "Test failed at " << testName << " #" << numberOfTest << ".3" << std::endl;
		}
	}
}

void testCreation(std::ostream& out) {
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
	testCreationOnce(out, testName, 0, params);
	params.layers.clear();

	desc.nNeuron = 1;
	params.layers.push_back(desc);
	desc.nNeuron = 1;
	params.layers.push_back(desc);
	testCreationOnce(out, testName, 1, params);
	params.layers.clear();

	desc.nNeuron = 100;
	params.layers.push_back(desc);
	desc.nNeuron = 50;
	params.layers.push_back(desc);
	desc.nNeuron = 30;
	params.layers.push_back(desc);
	desc.nNeuron = 10;
	params.layers.push_back(desc);
	testCreationOnce(out, testName, 2, params);
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
	testCreationOnce(out, testName, 2, params);
	params.layers.clear();

}

void performAllTests(std::ostream& out) {
	testCreation(out);




}

int main() {
	std::cout << "Tests are beginning for NeuralNetwork" << std::endl;
	performAllTests(std::cout);
	std::cout << "Tests have ended for NeuralNetwork" << std::endl;
	std::getchar();
}
