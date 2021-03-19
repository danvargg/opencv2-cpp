#include <stdint.h>
#include <torch/torch.h>
#include <iostream>
#include <string>
#include <vector>
#include "CSVReader.h"

// Reference: https://pytorch.org/tutorials/advanced/cpp_frontend.html

struct Options {
  size_t trainBatchSize = 4;
  size_t testBatchSize = 100;
  size_t epochs = 100;
  size_t logInterval = 20;
  // path must end in delimiter
  std::string datasetPath = "./BostonHousing.csv";
  // For CPU use torch::kCPU and for GPU use torch::kCUDA
  torch::DeviceType device = torch::kCPU;
};

static Options options;

// Convert string values from CSV to float and perform normalization
std::vector<std::vector<float>> normalize_feature(std::vector<std::vector<std::string> > feat, int rows, int cols) {
  std::vector<float> input(cols, 1);
  // Declare 2D array to store nSamples x nfeatures
  std::vector<std::vector<float>> data(rows, input);

  for (int i = 0; i < cols; i++) {   	// each column has one feature
      /* initialize the maximum element with 0 */
      // std::stof is used to convert string to float
      float maxm = std::stof(feat[1][i]);
      float minm = std::stof(feat[1][i]);

      // Run the inner loop over rows (all values of the feature) for given column (feature)
      for (int j = 1; j < rows; j++) {
          // check if any element is greater than the maximum element
          // of the column and replace it
          if (std::stof(feat[j][i]) > maxm)
              maxm = std::stof(feat[j][i]);

          if (std::stof(feat[j][i]) < minm)
              minm = std::stof(feat[j][i]);
      }


      // From above loop, we have min and max value of the feature
      // Will use min and max value to normalize values of the feature
      for (int j = 0; j < rows-1; j++) {
        // Normalize the feature values to lie between 0 and 1
        data[j][i] = (std::stof(feat[j+1][i]) - minm)/(maxm - minm);
      }
  }

  return data;
}

// Define Data to accomodate pairs of (feat, output)
using Data = std::vector<std::pair<std::vector<float>, float>>;

// Use CustomDataset class to load any type of dataset other than inbuilt datasets
// Reference: https://github.com/pytorch/examples/blob/master/cpp/custom-dataset/custom-dataset.cpp

class CustomDataset : public torch::data::datasets::Dataset<CustomDataset> {
  using Example = torch::data::Example<>;

  Data data;

 public:
  CustomDataset(const Data& data) : data(data) {}

  // Returns the Example at the given index, here we convert our data to tensors
  Example get(size_t index) {
    int fSize = data[index].first.size();
    // Convert feature vector into tensor of size fSize x 1
    auto tdata = torch::from_blob(&data[index].first, {fSize, 1});
    // Convert output value into tensor of size 1
    auto toutput = torch::from_blob(&data[index].second, {1});
    return {tdata, toutput};
  }

  // To get the size of the data
  torch::optional<size_t> size() const {
    return data.size();
  }
};


std::pair<Data, Data> readInfo() {
  Data train, test;

  // Reads data from CSV file.
  // CSVReader class is defined in CSVReader.h header file
  CSVReader reader(options.datasetPath);
  std::vector<std::vector<std::string> > dataList = reader.getData();


  int N = dataList.size();	// Total number of data points
  // As last column is output, feature size will be number of column minus one.
  int fSize = dataList[0].size() - 1;
  std::cout << "Total number of features: " << fSize << std::endl;
  std::cout << "Total number of data points: " << N << std::endl;
  int limit = 0.8*N;	// 80 percent data for training and rest 20 percent for validation
  std::vector<float> input(fSize, 1);
  std::vector<std::vector<float>> data(N, input);

  // Normalize data
  data = normalize_feature(dataList, N, fSize);


  for (int i=1; i < N; i++) {
    for (int j= 0; j < fSize; j++) {
        input[j] = data[i-1][j];
    }

    float output = std::stof(dataList[i][fSize]);

    // Split data data into train and test set
    if (i <= limit) {
      train.push_back({input, output});
    } else {
      test.push_back({input, output});
    }
  }

  std::cout << "Total number of training data: " << train.size() << std::endl;
  std::cout << "Total number of test data: " << test.size() << std::endl;

  // Shuffle training data
  std::random_shuffle(train.begin(), train.end());

  return std::make_pair(train, test);
}


// Linear Regression Model
struct Net : torch::nn::Module {
  /*
  Network for Linear Regression is just a single neuron (i.e. one Dense Layer)
  Usage: auto net = std::make_shared<Net>(num_features, num_outputs)
  */
  Net(int num_features, int num_outputs) {
    neuron = register_module("neuron", torch::nn::Linear(num_features, num_outputs));
    }

  torch::Tensor forward(torch::Tensor x) {
    /*Convert row tensor to column tensor*/
    x = x.reshape({x.size(0), -1});
    /*Pass the input tensor through linear function*/
    x = neuron->forward(x);
    return x;
  }

  /*Initilaize the constructor with null pointer. More details given in the reference*/
  torch::nn::Linear neuron{ nullptr };
};

template <typename DataLoader>
void train(std::shared_ptr<Net> network, DataLoader& loader, torch::optim::Optimizer& optimizer, size_t epoch, size_t data_size) {
  size_t index = 0;
  /*Set network in the training mode*/
  network->train();
  float Loss = 0;

  for (auto& batch : loader) {
    auto data = batch.data.to(options.device);
    auto targets = batch.target.to(options.device).view({-1, 1});
    // Execute the model on the input data
    auto output = network->forward(data);

    //Using mean square error loss function to compute loss
    auto loss = torch::mse_loss(output, targets);

    // Reset gradients
    optimizer.zero_grad();
    // Compute gradients
    loss.backward();
    //Update the parameters
    optimizer.step();

    Loss += loss.template item<float>();

    if (index++ % options.logInterval == 0) {
      auto end = std::min(data_size, (index + 1) * options.trainBatchSize);

      std::cout << "Train Epoch: " << epoch << " " << end << "/" << data_size
                << "\tLoss: " << Loss / end << std::endl;
    }
  }
}

template <typename DataLoader>
void test(std::shared_ptr<Net> network, DataLoader& loader, size_t data_size) {
  network->eval();

  for (const auto& batch : loader) {
    auto data = batch.data.to(options.device);
    auto targets = batch.target.to(options.device).view({-1, 1});

    auto output = network->forward(data);
    std::cout << "Predicted:"<< output[0].template item<float>() << "\t" << "Groundtruth: "
	    << targets[0].template item<float>() << std::endl;
    std::cout << "Predicted:"<< output[1].template item<float>() << "\t" << "Groundtruth: "
	    << targets[1].template item<float>() << std::endl;
    std::cout << "Predicted:"<< output[2].template item<float>() << "\t" << "Groundtruth: "
	    << targets[2].template item<float>() << std::endl;
    std::cout << "Predicted:"<< output[3].template item<float>() << "\t" << "Groundtruth: "
	    << targets[3].template item<float>() << std::endl;
    std::cout << "Predicted:"<< output[4].template item<float>() << "\t" << "Groundtruth: "
	    << targets[4].template item<float>() << std::endl;

    auto loss = torch::mse_loss(output, targets);

    break;
  }
}

int main() {
  /*Sets manual seed from libtorch random number generators*/
  torch::manual_seed(1);

  /*Use CUDA for computation if available*/
  if (torch::cuda::is_available())
    options.device = torch::kCUDA;
  std::cout << "Running on: "
            << (options.device == torch::kCUDA ? "CUDA" : "CPU") << std::endl;

  /*Read data and split data into train and test sets*/
  auto data = readInfo();

  /*Uses Custom Dataset Class to load train data. Apply stack collation which takes
  batch of tensors and stacks them into single tensor along the first dimension*/
  auto train_set =
      CustomDataset(data.first).map(torch::data::transforms::Stack<>());
  auto train_size = train_set.size().value();

  /*Data Loader provides options to speed up the data loading like batch size, number of workers*/
  auto train_loader =
      torch::data::make_data_loader(
          std::move(train_set), options.trainBatchSize);

  std::cout << train_size << std::endl;
  /*Uses Custom Dataset Class to load test data. Apply stack collation which takes
  batch of tensors and stacks them into single tensor along the first dimension*/
  auto test_set =
      CustomDataset(data.second).map(torch::data::transforms::Stack<>());
  auto test_size = test_set.size().value();

  /*Test data loader similar to train data loader*/
  auto test_loader =
      torch::data::make_data_loader(
          std::move(test_set), options.testBatchSize);
  /*Create Linear  Regression Network*/
  auto net = std::make_shared<Net>(13, 1);

  /*Moving model parameters to correct device*/
  net->to(options.device);

  /*Using stochastic gradient descent optimizer with learning rate 0.000001*/
  torch::optim::SGD optimizer(
       net->parameters(), torch::optim::SGDOptions(0.000001));

  std::cout << "Training..." << std::endl;
  for (size_t i = 0; i < options.epochs; ++i) {
    /*Run the training for all iterations*/
    train(net, *train_loader, optimizer, i + 1, train_size);
    std::cout << std::endl;

    if (i == options.epochs - 1) {
      std::cout << "Testing..." << std::endl;
      test(net, *test_loader, test_size);
    }
  }


  return 0;
}
