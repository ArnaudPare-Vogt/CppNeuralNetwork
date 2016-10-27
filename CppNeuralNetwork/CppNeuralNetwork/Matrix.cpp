#include "Matrix.h"

#include <cassert>

namespace NNet {
	Matrix::Matrix(size_t nRows, size_t nCols) : nRows(nRows), nCols(nCols) {
		assert(nRows > 0);
		assert(nCols > 0);
		for (size_t i = 0; i < nRows; i++)
		{
			std::vector<float> vec;
			values.push_back(vec);
			values.back().reserve(nCols);
		}
	}

	size_t Matrix::getReservedRows() const {
		return values.capacity();
	}

	size_t Matrix::getReservedColumns() const {
		if (values.size() > 0) {
			return values[0].capacity();
		}
		else {
			return 0;
		}
	}

	int Matrix::getNCols() const {
		return nCols;
	}

	int Matrix::getNRows() const {
		return nRows;
	}
}