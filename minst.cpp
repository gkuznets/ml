#include "read_minst.h"
#include <ml/bit_vec.h>
#include <ml/dag_muticlass.h>
#include <ml/dataset/dataset.h>
#include <ml/exception.h>
#include <ml/kernels.h>
#include <ml/svm/svc.h>

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include <boost/program_options.hpp>

typedef ml::VecDataset<MINSTImage, int> MINSTDataset;

MINSTDataset
readMINSTDataset(const std::string& imagesFile, const std::string& labelsFile) {
    auto data = readMINSTData(imagesFile, labelsFile);
    MINSTDataset result;
    result.examples = std::move(data.first);
    result.labels.resize(result.examples.size());
    std::copy(data.second.begin(), data.second.end(), result.labels.begin());
    return result;
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

