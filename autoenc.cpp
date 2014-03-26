#include <ml/ann.h>

#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <Eigen/Dense>
#include <QtGui/QImage>
#include <QtGui/QPainter>

std::vector<QImage> generateImages(size_t numImages) {
    std::random_device gen;
    //std::mt19937 gen;
    std::uniform_int_distribution<> dis(1, 9);
    std::vector<QImage> result;
    result.reserve(numImages);
    for (size_t i = 0; i < numImages; ++i) {
        result.emplace_back(10, 10, QImage::Format_RGB32);
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

typedef Eigen::Matrix<double, 100, 1> ImageVec;

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

void dump(const ImageVec& img, unsigned width = 10, unsigned height = 10) {
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
        : examples(100, size), labels(100, size) {}

    Eigen::Matrix<double, 100, Eigen::Dynamic> examples;
    Eigen::Matrix<double, 100, Eigen::Dynamic> labels;
};

uint64_t size(const ANNDataset& ds) {
    return ds.examples.cols();
}


int main() {
    const size_t datasetSize = 2000;
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
        ml::ann::Input<100>,
        ml::ann::FullyConnected<60>,
        ml::ann::FullyConnected<100>> NetworkConf;

    time_t start = clock();
    auto clsfr = ml::ann::train(NetworkConf{}, trainingSet, validationSet);
    std::cout << " time: " << (clock() - start) / CLOCKS_PER_SEC << "\n";
    auto test = generateImages(1);
    dump(test[0]);
    std::cout << "\n";
    dump(clsfr(img2vec(test[0])));
}

