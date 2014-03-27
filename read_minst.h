#pragma once

#include <ml/bit_vec.h>

#include <string>
#include <utility>
#include <vector>

typedef ml::BitVec<28> MINSTImage;

//! Reads set of handwritten images and corresponding labels.
//! For information about IDX format used see http://yann.lecun.com/exdb/mnist/
std::pair<std::vector<MINSTImage>
         ,std::vector<unsigned>>
readMINSTData(const std::string& imagesFile, const std::string& labelsFile);

