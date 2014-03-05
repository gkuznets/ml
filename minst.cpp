#include <ml/bit_vec.h>
#include <ml/dag_muticlass.h>
#include <ml/dataset/dataset.h>
#include <ml/exception.h>
#include <ml/kernels.h>
#include <ml/svm/svc.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <boost/program_options.hpp>

const uint32_t IMAGE_WIDTH = 28;
const uint32_t IMAGE_HEIGHT = 28;
const uint32_t IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT;

typedef ml::VecDataset<ml::BitVec<IMAGE_SIZE>, int> MINSTDataset;

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

//! Reads set of handwritten images and corresponding labels.
//! For information about IDX format used see http://yann.lecun.com/exdb/mnist/
MINSTDataset readMINSTDataset(
        const std::string& imagesFile,
        const std::string& labelsFile)
{
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

    MINSTDataset dataset(numOfImages);
    std::vector<uint8_t> image(IMAGE_SIZE);
    uint8_t label;
    for (uint32_t i = 0; i < numOfImages; ++i) {
        imagesStream.read(
                reinterpret_cast<char*>(image.data()), IMAGE_SIZE);
        ml::BitVec<IMAGE_SIZE> packedImage;
        for (unsigned i = 0; i < IMAGE_SIZE; ++i) {
            if (image[i]) {
                packedImage.set(i);
            }
        }
        readValue(labelsStream, label);
        ml::set(i, packedImage, static_cast<int>(label), dataset);
    }
    return dataset;
}

template <typename Classifier>
float test(Classifier&& classifier, const MINSTDataset& dataset) {
    float errorsCount = 0;
    for (uint64_t i = 0; i < size(dataset); ++i) {
        if (classifier(example(i ,dataset)) != label(i, dataset)) {
            errorsCount += 1.0;
        }
    }
    return errorsCount / static_cast<float>(size(dataset));
}

int main(int argc, char** argv) {
    namespace po = boost::program_options;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("training-images,I", po::value<std::string>(),
            "File with training images.")
        ("training-labels,L", po::value<std::string>(),
            "File with training labels.")
        ("test-images,i", po::value<std::string>(),
            "File with test images.")
        ("test-labels,l", po::value<std::string>(),
            "File with test labels.")
        ("kernel,k", po::value<std::string>()->default_value("poly"),
            "Kernel type: 'gaussian' for gaussian RBF or 'poly' for polynomial.")
    ;

    po::variables_map vars;
    po::store(po::parse_command_line(argc, argv, desc), vars);
    po::notify(vars);

    if (vars.count("help")) {
        std::cout << desc << "\n";
        return 0;
    }

    if (!vars.count("training-images") || !vars.count("training-labels") ||
        !vars.count("test-images") || !vars.count("test-labels")) {
        std::cout << "Specify all training/test set files\n";
        return 1;
    }

    auto minstTrainingSet = readMINSTDataset(
            vars["training-images"].as<std::string>(),
            vars["training-labels"].as<std::string>());

    auto minstTestSet = readMINSTDataset(
            vars["test-images"].as<std::string>(),
            vars["test-labels"].as<std::string>());

    const double REGULARIZATION_PARAM = 0.5;
    if (vars["kernel"].as<std::string>() == "poly") {
        auto classifier = ml::dag::train(minstTrainingSet,
            [REGULARIZATION_PARAM] (const MINSTDataset& ds) {
                return ml::svc::train(
                    ds, REGULARIZATION_PARAM, ml::PolynomialKernel<2>());
            });
        std::cout << "Error rate: "
                  << test(classifier, minstTestSet) << "\n";
    } else if (vars["kernel"].as<std::string>() == "gaussian") {
        auto classifier = ml::dag::train(minstTrainingSet,
            [REGULARIZATION_PARAM] (const MINSTDataset& ds) {
                return ml::svc::train(
                    ds, REGULARIZATION_PARAM, ml::RBFKernel(15.0));
            });
        std::cout << "Error rate: "
                  << test(classifier, minstTestSet) << "\n";
    } else {
        std::cout << "Unknown kernel type: "
                  << vars["kernel"].as<std::string>() << "\n";
        return 1;
    }
}

