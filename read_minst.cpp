#include "read_minst.h"

#include <ml/exception.h>
#include <fstream>

namespace {

const uint32_t IMAGE_WIDTH = 28;
const uint32_t IMAGE_HEIGHT = 28;
const uint32_t IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT;

uint32_t swapEndian(uint32_t u) {
    return (u << 24) |
           ((u << 8) & 0x00FF0000) |
           ((u >> 8) & 0x0000FF00) |
           (u >> 24);
}

template <typename IStream, typename T>
void readValue(IStream& inputStream, T& t) {
    inputStream.read(reinterpret_cast<char*>(&t), sizeof(t));
}

} // namespace

std::pair<std::vector<MINSTImage>
         ,std::vector<unsigned>>
readMINSTData(const std::string& imagesFile, const std::string& labelsFile) {
    std::ifstream imagesStream(imagesFile, std::ios_base::binary);
    std::ifstream labelsStream(labelsFile, std::ios_base::binary);

    uint16_t zero;
    uint8_t dataTypeCode;
    uint8_t numOfDimensions;
    uint32_t numOfImages;
    uint32_t imageWidth;
    uint32_t imageHeight;
    // Reading IDX header for images
    readValue(imagesStream, zero);
    readValue(imagesStream, dataTypeCode);
    readValue(imagesStream, numOfDimensions);
    REQUIRE(zero == 0, "Invalid IDX header.");
    REQUIRE(dataTypeCode == 0x08, "Invalid data type code for images.");
    REQUIRE(numOfDimensions == 3, "Invalid number of dimensions for images.");
    readValue(imagesStream, numOfImages);
    readValue(imagesStream, imageWidth);
    readValue(imagesStream, imageHeight);
    numOfImages = swapEndian(numOfImages);
    imageWidth = swapEndian(imageWidth);
    imageHeight = swapEndian(imageHeight);
    REQUIRE(imageWidth == IMAGE_WIDTH && imageHeight == IMAGE_HEIGHT,
            "Incorrect image size");

    // Reading IDX header for labels
    uint32_t numOfLabels;
    readValue(labelsStream, zero);
    readValue(labelsStream, dataTypeCode);
    readValue(labelsStream, numOfDimensions);
    REQUIRE(zero == 0, "Invalid IDX header.");
    REQUIRE(dataTypeCode == 0x08, "Invalid data type code for labels.");
    REQUIRE(numOfDimensions == 1, "Invalid number of dimensions for labels.");
    readValue(labelsStream, numOfLabels);
    numOfLabels = swapEndian(numOfLabels);
    REQUIRE(numOfImages == numOfLabels,
            "Number of images and labels should be the same.");

    std::pair<std::vector<MINSTImage>, std::vector<unsigned>> result;
    auto& images = result.first;
    images.reserve(numOfImages);
    auto& labels = result.second;
    labels.reserve(numOfImages);

    std::vector<uint8_t> image(IMAGE_SIZE);
    uint8_t label;
    for (uint32_t i = 0; i < numOfImages; ++i) {
        imagesStream.read(
                reinterpret_cast<char*>(image.data()), IMAGE_SIZE);
        MINSTImage minstImage;
        for (unsigned i = 0; i < IMAGE_SIZE; ++i) {
            if (image[i]) {
                minstImage.set(i);
            }
        }
        images.push_back(minstImage);
        readValue(labelsStream, label);
        labels.push_back(label);
    }
    return result;
}

