#pragma once

#include <vector>

namespace NNet {


	class Matrix {
	private:
		std::vector < std::vector<float> > values;
		int nRows, nCols;

	public:
		Matrix(size_t nRows, size_t nCols);
		
	public:
		size_t getReservedColumns() const;
		size_t getReservedRows() const;

		int getNRows() const;
		int getNCols() const;
	};


}
