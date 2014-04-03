#include "read_minst.h"
#include <meta/logic.h>
#include <meta/params.h>
#include <ml/ann.h>
#include <ml/ann/stop_criterion.h>
#include <ml/exception.h>

#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <boost/program_options.hpp>
#include <Eigen/Dense>
#include <QtGui/QImage>
#include <QtGui/QPainter>

static const unsigned IMG_SIZE = 28;
static const unsigned IMG_AREA = IMG_SIZE * IMG_SIZE;

std::vector<QImage> generateImages(size_t numImages) {
    std::random_device gen;
    //std::mt19937 gen;
    std::uniform_int_distribution<> dis(1, IMG_SIZE - 1);
    std::vector<QImage> result;
    result.reserve(numImages);
    for (size_t i = 0; i < numImages; ++i) {
        result.emplace_back(IMG_SIZE, IMG_SIZE, QImage::Format_RGB32);
        QImage& img = result.back();
        img.fill(Qt::white);
        QPainter p(&img);
        p.setRenderHint(QPainter::Antialiasing, true);

        QPen pen;
        pen.setWidth(2);
        pen.setColor(Qt::black);
        p.setPen(pen);
        p.drawLine(dis(gen), dis(gen), dis(gen), dis(gen));
        p.end();
    }
    return result;
}

typedef Eigen::Matrix<double, IMG_AREA, 1> ImageVec;

ImageVec img2vec(const QImage& img) {
    ImageVec result;
    for (int y = 0; y < img.height(); ++y) {
        for (int x = 0; x < img.width(); ++x) {
            result(x + y * img.width()) =
                static_cast<double>(qGray(img.pixel(x, y))) / 128.0 - 1.0;
        }
    }
    return result;
}

void dump(const QImage& img) {
    for (int y = 0; y < img.height(); ++y) {
        for (int x = 0; x < img.width(); ++x) {
            std::cout << std::setfill('0') << std::setw(3);
            std::cout << qGray(img.pixel(x, y)) << " ";
        }
        std::cout << "\n";
    }
}

void dump(const ImageVec& img, unsigned width = IMG_SIZE, unsigned height = IMG_SIZE) {
    for (unsigned y = 0; y < height; ++y) {
        for (unsigned x = 0; x < width; ++x) {
            std::cout << std::setfill('0') << std::setw(3);
            std::cout << static_cast<int>((img(x + y * width) + 1.0) * 128.0)
                      << " ";
        }
        std::cout << "\n";
    }
}

struct ANNDataset {
    ANNDataset() = default;
    ANNDataset(ANNDataset&&) = default;
    explicit ANNDataset(uint64_t size)
        : examples(IMG_AREA, size), labels(IMG_AREA, size) {}

    Eigen::Matrix<double, IMG_AREA, Eigen::Dynamic> examples;
    Eigen::Matrix<double, IMG_AREA, Eigen::Dynamic> labels;
};

uint64_t size(const ANNDataset& ds) {
    return ds.examples.cols();
}


#if 0
int main() {
    const size_t datasetSize = 3000;
    auto images = generateImages(datasetSize);
    ANNDataset trainingSet(datasetSize);
    unsigned pos = 0;
    for (const auto& img: images) {
        auto vec = img2vec(img);
        trainingSet.examples.col(pos) = vec;
        trainingSet.labels.col(pos) = vec;
        pos++;
    }

    auto validationImages = generateImages(300);
    ANNDataset validationSet(300);
    pos = 0;
    for (const auto& img: validationImages) {
        auto vec = img2vec(img);
        validationSet.examples.col(pos) = vec;
        validationSet.labels.col(pos) = vec;
        pos++;
    }

    typedef ml::ann::NetworkConf<
        ml::ann::Input<IMG_AREA>,
        ml::ann::FullyConnected<100>,
        ml::ann::FullyConnected<IMG_AREA>> NetworkConf;

    time_t start = clock();
    auto clsfr = ml::ann::train(NetworkConf{}, trainingSet, validationSet);
    std::cout << " time: " << (clock() - start) / CLOCKS_PER_SEC << "\n";
    auto test = generateImages(1);
    dump(test[0]);
    std::cout << "\n";
    dump(clsfr(img2vec(test[0])));
    clsfr.dump("nodes.txt");
}
#endif

ImageVec img2vec(const MINSTImage& img) {
    ImageVec result;
    for (unsigned i = 0; i < img.size(); ++i) {
        result(i) = (img(i) ? 1.0 : -1.0);
    }
    return result;
}

ANNDataset
readMINSTDataset(const std::string& imgs, const std::string& lbls) {
    auto minstData = readMINSTData(imgs, lbls);
    const auto& images = minstData.first;
    ANNDataset result(images.size());
    for (unsigned pos = 0; pos < images.size(); ++pos) {
        auto vec = img2vec(images[pos]);
        result.examples.col(pos) = vec;
        result.labels.col(pos) = std::move(vec);
    }
    return result;
}

void addNoise(ANNDataset& ds, double level = 0.25) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (unsigned r = 0; r < ds.examples.rows(); ++r) {
        for (unsigned c = 0; c < ds.examples.cols(); ++c) {
            if (dis(gen) < level) {
                ds.examples(r, c) = 0.0;
            }
        }
    }
}

int main(int argc, char** argv) {
    try {
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

        typedef ml::ann::NetworkConf<
            ml::ann::Input<28 * 28>,
            ml::ann::FullyConnected<200>,
            ml::ann::FullyConnected<28 * 28>> NetworkConf;

        ml::ann::EarlyStopping<ANNDataset, NetworkConf>
            earlyStopping(minstTestSet, 1);
        ml::ann::EpochsNumber epochsNumber(100);

        auto optimizationParams = std::make_tuple(
                meta::param<ml::batchSizeP>(32),
                meta::param<ml::regularizationP>(ml::ann::L2Regularization(1.0)),
                meta::param<ml::stopCriterionP>(
                    meta::any(earlyStopping.ref(), epochsNumber)),
                meta::param<ml::optimizationMonitorP>(earlyStopping.ref()));
        time_t start = clock();
        std::cout << "training...\n";
        auto clsfr = ml::ann::train(minstTrainingSet,
                optimizationParams, NetworkConf{});
        std::cout << " time: " << (clock() - start) / CLOCKS_PER_SEC << "\n";
    } catch (ml::RuntimeException& e) {
        std::cout << e.what() << "\n";
        return -1;
    }
}

