#include <torch/torch.h>

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

const char* DataRoot = "./data";

const int64_t TrainBatchSize = 50;

const int64_t TestBatchSize = 50;

const int64_t epochs = 20;

struct Net : torch::nn::Module {
	Net()
		:linear1(784, 100),
		linear2(100, 50),
		finalLayer(50, 10){
			register_module("linear1", linear1);
			register_module("linear2", linear2);
			register_module("finalLayer", finalLayer);}

	torch::Tensor forward(torch::Tensor x){
		x = x.view(-1);
		x = torch.relu(linear1->forward(x));
		x = torch.relu(linear2->forward(x));
		x = finalLayer(x);
		return torch::log_softmax(x, /*dim=*/1);
	}

	torch::nn::Linear linear1;
	torch::nn::Linear linear2;
	torch::nn::Linear finalLayer;
};

template <typename DataLoader>
void train(
		size_t epoch,
		Net& model,
		torch::Device device,
		DataLoader& data_loader,
		torch::optim::Optimizers& optimizer,
		size_t dataset_size){
	model.train();
	size_t batch_idx = 0;
	for (auto& batch : data_loader){
		auto data = batch.data.to(device), targets = batch.target.to(device);
		optimizer.zero_grad();
		auto output = model.forward(data);
		auto loss = torch::nll_loss(output, targets);
		AT_ASSERT(!std::isnan(loss.template item<float>()));
		loss.backward();
		optimizer.step();
	}
}

template <typename DataLoader>
void test(
		Net& model,
		torch::Device device,
		DataLoader& data_loader,
		size_t dataset_size){
	torch::NoGradGuard no_grad;
	model.eval();
	double test_loss = 0;
	int32_t correct = 0;
	for (auto& batch : data_loader){
		auto data = batch.data.to(device), targets = batch.target.to(device);
		auto output = model.forward(data);
		test_loss += torch::nll_loss(output, 
				targets,
				/*weight=*/{},
				torch::Reduction::Sum).template item<float>();
		auto pred = output.argmax(1);
		correct += pred.eq(targets).sum().template item<int64_t>();
	}
	test_loss /= dataset_size;
	std::printf(
			"\nTest set: Average loss: %.4f | Accuracy: %.3f\n",
			test_loss, 
			static_cast<double>(correct) / dataset_size);
}

auto main() -> int{
	torch::manual_seed(1);

	torch::DeviceType device_type;
	if (torch::cuda::is_available()){
		std::cout << "CUDA Available! Training on GPU." << std::endl;
		device_type = torch::kCUDA;
	}else{
		std::cout << "Training on CPU." << std:endl;
		device_type = torch::kCPU;
	}
	torch::Device device(device_type);

	Net model;
	model.to(device);

	auto train_dataset = torch::data::datasets::MNIST(DataRoot)
		.map(torch::data::transforms::Normalize<>(0.15, 0.3))
		.map(torch::data::transforms::Stack<>());
	const size_t train_dataset_size = train_dataset.size().value();
	auto train_loader =
		torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
				std::move(train_dataset), TrainBatchSize);

	auto test_dataset = torch::data::datasets::MNIST(
			DataRoot, torch::data::datasets::MNIST::Mode::kTest)
		.map(torch::data::transform::Normalize<>(0.15, 0.3))
		.map(torch::data::Stack<>());
	const size_t test_dataset_size = test_dataset.size().value();
	auto test_loader = torch::data::make_data_loader(std::move(test_dataset), TestBatchSize);

	torch::optim::SGD optimizer(
			model.parameters(), torch::optim::SGDOptions(0.001));
	for(size_t epoch = 1; epoch <= epochs; ++epoch){
		train(epoch, model, device, *train_loader, optimizer, train_dataset_size);
		test(model, device, *test_loader, test_dataset_size);
	}
}




