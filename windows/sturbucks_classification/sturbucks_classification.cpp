#include <caffe/caffe.hpp>
#include "boost/filesystem.hpp"
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;
using namespace boost::filesystem;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

#define BATCH_SIZE 32

class Classifier {
public:
	Classifier(const string& model_file,
		const string& trained_file,
		const string& label_file);

	std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);
	std::vector<std::vector<Prediction>> ClassifyBatch(const vector<cv::Mat> imgs, int num_classes = 2);
private:
	std::vector<float> Predict(const cv::Mat& img);
	std::vector< float> PredictBatch(const vector<cv::Mat> imgs);

	void WrapInputLayer(std::vector<cv::Mat>* input_channels);
	void WrapBatchInputLayer(std::vector<std::vector<cv::Mat>> * input_batch);

	void Preprocess(const cv::Mat& img,
		std::vector<cv::Mat>* input_channels);
	void PreprocessBatch(const vector<cv::Mat> imgs,
		std::vector< std::vector<cv::Mat> >* input_batch);

private:
	shared_ptr<Net<float> > net_;
	cv::Size input_geometry_;
	int num_channels_;
	std::vector<string> labels_;
};

Classifier::Classifier(const string& model_file,
	const string& trained_file,
	const string& label_file) {
#ifdef CPU_ONLY
	Caffe::set_mode(Caffe::CPU);
#else
	Caffe::set_mode(Caffe::GPU);
#endif

	/* Load the network. */
	net_.reset(new Net<float>(model_file, TEST));
	net_->CopyTrainedLayersFrom(trained_file);

	CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
//	CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

	Blob<float>* input_layer = net_->input_blobs()[0];
	num_channels_ = input_layer->channels();
	CHECK(num_channels_ == 3 || num_channels_ == 1)
		<< "Input layer should have 1 or 3 channels.";
	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

	/* Load labels. */
	std::ifstream labels(label_file.c_str());
	CHECK(labels) << "Unable to open labels file " << label_file;
	string line;
	while (std::getline(labels, line))
		labels_.push_back(string(line));

	Blob<float>* output_layer = net_->output_blobs().back();
	CHECK_EQ(labels_.size(), output_layer->channels())
		<< "Number of labels is different from the output layer dimension.";
}

static bool PairCompare(const std::pair<float, int>& lhs,
	const std::pair<float, int>& rhs) {
	return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
	std::vector<std::pair<float, int> > pairs;
	for (size_t i = 0; i < v.size(); ++i)
		pairs.push_back(std::make_pair(v[i], static_cast<int>(i)));
	std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

	std::vector<int> result;
	for (int i = 0; i < N; ++i)
		result.push_back(pairs[i].second);
	return result;
}

/* Return the top N predictions. */
std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N) {
	std::vector<float> output = Predict(img);

	N = std::min<int>(labels_.size(), N);
	std::vector<int> maxN = Argmax(output, N);
	std::vector<Prediction> predictions;
	for (int i = 0; i < N; ++i) {
		int idx = maxN[i];
		predictions.push_back(std::make_pair(labels_[idx], output[idx]));
	}

	return predictions;
}

std::vector< std::vector<Prediction> > Classifier::ClassifyBatch(const vector< cv::Mat > imgs, int num_classes){
	std::vector<float> output_batch = PredictBatch(imgs);
	std::vector< std::vector<Prediction> > predictions;
	for (int j = 0; j < imgs.size(); j++){
		std::vector<float> output(output_batch.begin() + j*num_classes, output_batch.begin() + (j + 1)*num_classes);
		std::vector<int> maxN = Argmax(output, num_classes);
		std::vector<Prediction> prediction_single;
		for (int i = 0; i < num_classes; ++i) {
			int idx = maxN[i];
			prediction_single.push_back(std::make_pair(labels_[idx], output[idx]));
		}
		predictions.push_back(std::vector<Prediction>(prediction_single));
	}
	return predictions;
}

std::vector<float> Classifier::Predict(const cv::Mat& img) {
	Blob<float>* input_layer = net_->input_blobs()[0];
	input_layer->Reshape(1, num_channels_,
		input_geometry_.height, input_geometry_.width);
	/* Forward dimension change to all layers. */
	net_->Reshape();

	std::vector<cv::Mat> input_channels;
	WrapInputLayer(&input_channels);

	Preprocess(img, &input_channels);

	net_->Forward();

	/* Copy the output layer to a std::vector */
	Blob<float>* output_layer = net_->output_blobs().back();
	const float* begin = output_layer->cpu_data();
	const float* end = begin + output_layer->channels();
	return std::vector<float>(begin, end);
}

std::vector< float >  Classifier::PredictBatch(const vector< cv::Mat > imgs) {
	Blob<float>* input_layer = net_->input_blobs()[0];

	input_layer->Reshape(BATCH_SIZE, num_channels_,
		input_geometry_.height,
		input_geometry_.width);

	/* Forward dimension change to all layers. */
	net_->Reshape();

	std::vector< std::vector<cv::Mat> > input_batch;
	WrapBatchInputLayer(&input_batch);

	PreprocessBatch(imgs, &input_batch);

	net_->ForwardPrefilled();

	/* Copy the output layer to a std::vector */
	Blob<float>* output_layer = net_->output_blobs().back();
	const float* begin = output_layer->cpu_data();
	const float* end = begin + output_layer->channels()*imgs.size();
	return std::vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
* (one per channel). This way we save one memcpy operation and we
* don't need to rely on cudaMemcpy2D. The last preprocessing
* operation will write the separate channels directly to the input
* layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
	Blob<float>* input_layer = net_->input_blobs()[0];

	int width = input_layer->width();
	int height = input_layer->height();
	float* input_data = input_layer->mutable_cpu_data();
	for (int i = 0; i < input_layer->channels(); ++i) {
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels->push_back(channel);
		input_data += width * height;
	}
}

void Classifier::WrapBatchInputLayer(std::vector<std::vector<cv::Mat> > *input_batch){
	Blob<float>* input_layer = net_->input_blobs()[0];

	int width = input_layer->width();
	int height = input_layer->height();
	int num = input_layer->num();
	float* input_data = input_layer->mutable_cpu_data();
	for (int j = 0; j < num; j++){
		vector<cv::Mat> input_channels;
		for (int i = 0; i < input_layer->channels(); ++i){
			cv::Mat channel(height, width, CV_32FC1, input_data);
			input_channels.push_back(channel);
			input_data += width * height;
		}
		input_batch->push_back(vector<cv::Mat>(input_channels));
	}
	cv::imshow("bla", input_batch->at(1).at(0));
	cv::waitKey(1);
}

void Classifier::Preprocess(const cv::Mat& img,
	std::vector<cv::Mat>* input_channels) {
	/* Convert the input image to the input image format of the network. */
	cv::Mat sample;
	if (img.channels() == 3 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
	else if (img.channels() == 4 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
	else if (img.channels() == 4 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
	else if (img.channels() == 1 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
	else
		sample = img;

	cv::Mat sample_resized;
	if (sample.size() != input_geometry_)
		cv::resize(sample, sample_resized, input_geometry_);
	else
		sample_resized = sample;

	cv::Mat sample_float;
	if (num_channels_ == 3)
		sample_resized.convertTo(sample_float, CV_32FC3);
	else
		sample_resized.convertTo(sample_float, CV_32FC1);

	/* This operation will write the separate BGR planes directly to the
	* input layer of the network because it is wrapped by the cv::Mat
	* objects in input_channels. */
	cv::split(sample_float, *input_channels);

	CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
		== net_->input_blobs()[0]->cpu_data())
		<< "Input channels are not wrapping the input layer of the network.";
}


void Classifier::PreprocessBatch(const vector<cv::Mat> imgs,
	std::vector< std::vector<cv::Mat> >* input_batch){
	for (int i = 0; i < imgs.size(); i++){
		cv::Mat img = imgs[i];
		std::vector<cv::Mat> *input_channels = &(input_batch->at(i));

		/* Convert the input image to the input image format of the network. */
		cv::Mat sample;
		if (img.channels() == 3 && num_channels_ == 1)
			cv::cvtColor(img, sample, CV_BGR2GRAY);
		else if (img.channels() == 4 && num_channels_ == 1)
			cv::cvtColor(img, sample, CV_BGRA2GRAY);
		else if (img.channels() == 4 && num_channels_ == 3)
			cv::cvtColor(img, sample, CV_BGRA2BGR);
		else if (img.channels() == 1 && num_channels_ == 3)
			cv::cvtColor(img, sample, CV_GRAY2BGR);
		else
			sample = img;

		cv::Mat sample_resized;
		if (sample.size() != input_geometry_)
			cv::resize(sample, sample_resized, input_geometry_);
		else
			sample_resized = sample;

		cv::Mat sample_float;
		if (num_channels_ == 3)
			sample_resized.convertTo(sample_float, CV_32FC3);
		else
			sample_resized.convertTo(sample_float, CV_32FC1);

		/* This operation will write the separate BGR planes directly to the
		* input layer of the network because it is wrapped by the cv::Mat
		* objects in input_channels. */
		cv::split(sample_float, *input_channels);

		//        CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
		//              == net_->input_blobs()[0]->cpu_data())
		//          << "Input channels are not wrapping the input layer of the network.";
	}
}

int main(int argc, char** argv) {
	if (argc != 5) {
		std::cerr << "Usage: " << argv[0]
			<< " deploy.prototxt network.caffemodel"
			<< " labels.txt img.jpg" << std::endl;
		return 1;
	}

	::google::InitGoogleLogging(argv[0]);

	string model_file = argv[1];
	string trained_file = argv[2];
	string label_file = argv[3];
	Classifier classifier(model_file, trained_file, label_file);

	string path = argv[4];

	if (is_directory(path)) {
		directory_iterator end_itr;
		int file_counter = 0;
		std::vector<cv::Mat> imgs;
		std::vector<std::string> file_names;
		for (directory_iterator itr(path); itr != end_itr; ++itr) {
			std::string file_path = itr->path().string();
			file_names.push_back(file_path);

			cv::Mat img = cv::imread(file_path, -1);
			imgs.push_back(img);

			file_counter++;

			if (file_counter % BATCH_SIZE == 0) {
				std::vector< std::vector<Prediction> > batch_predictions = classifier.ClassifyBatch(imgs);

				/* Print the top N predictions. */
				for (size_t i = 0; i < batch_predictions.size(); ++i) {
					std::vector<Prediction> predictions = batch_predictions[i];
					std::cout << "---------- Prediction for "
						<< file_names[file_counter - batch_predictions.size() + i] << " ----------" << std::endl;
					for (size_t j = 0; j < predictions.size(); ++j) {
						Prediction p = predictions[j];
						std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
							<< p.first << "\"" << std::endl;
					}
				}

				imgs.clear();
			}


		}

		if (imgs.size() > 0) {
			std::vector< std::vector<Prediction> > batch_predictions = classifier.ClassifyBatch(imgs);

			/* Print the top N predictions. */
			for (size_t i = 0; i < batch_predictions.size(); ++i) {
				std::vector<Prediction> predictions = batch_predictions[i];
				std::cout << "---------- Prediction for "
					<< file_names[file_counter - batch_predictions.size() + i] << " ----------" << std::endl;
				for (size_t j = 0; j < predictions.size(); ++j) {
					Prediction p = predictions[j];
					std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
						<< p.first << "\"" << std::endl;
				}
			}
		}
	}
	else {
		cv::Mat img = cv::imread(path, -1);
		CHECK(!img.empty()) << "Unable to decode image " << path;
		std::vector<Prediction> predictions = classifier.Classify(img);

		/* Print the top N predictions. */
		for (size_t i = 0; i < predictions.size(); ++i) {
			Prediction p = predictions[i];
			std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
				<< p.first << "\"" << std::endl;
		}
	}
}
#else
int main(int argc, char** argv) {
	LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
