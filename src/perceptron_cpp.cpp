#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <chrono>

using namespace std;
using namespace std::chrono;

// Function to read data from CSV file
bool readData(const string& filename, vector<vector<double>>& x, vector<int>& y) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return false;
    }
    string line;
    // Skip header
    getline(file, line);
    while (getline(file, line)) {
        stringstream ss(line);
        vector<double> row;
        string token;
        while (getline(ss, token, ',')) {
            row.push_back(stod(token));
        }
        if (!row.empty()) {
            y.push_back(static_cast<int>(row.back()));
            row.pop_back();
            x.push_back(row);
        }
    }
    file.close();
    return true;
}

// Function to compute prediction
int predict(const vector<double>& w, double b, const vector<double>& features) {
    double sum_ = b;
    for (size_t j = 0; j < features.size(); ++j) {
        sum_ += w[j] * features[j];
    }
    return (sum_ > 0) ? 1 : 0;
}

// Function to calculate accuracy
double calculateAccuracy(const vector<vector<double>>& x, const vector<int>& y, const vector<double>& w, double b) {
    int num_samples = x.size();
    int correct = 0;
    for (int i = 0; i < num_samples; ++i) {
        int pred = predict(w, b, x[i]);
        if (pred == y[i]) ++correct;
    }
    return (static_cast<double>(correct) / num_samples) * 100.0;
}

// Function to train the perceptron
void trainPerceptron(vector<vector<double>>& x, vector<int>& y, double learning_rate, int max_epoch,
                     vector<double>& best_weight, double& best_bias, double& max_accuracy, int& best_e) {
    int num_samples = x.size();
    if (num_samples == 0) return;
    int num_features = x[0].size();

    vector<double> w(num_features, 0.0);
    double b = 0.0;
    max_accuracy = 0.0;
    best_e = 0;
    int e = 0;

    while (e < max_epoch) {
        ++e;
        bool updated = false;
        for (int i = 0; i < num_samples; ++i) {
            int pred = predict(w, b, x[i]);
            if (pred != y[i]) {
                updated = true;
                double lr_factor = (y[i] == 1) ? learning_rate : -learning_rate;
                for (int j = 0; j < num_features; ++j) {
                    w[j] += lr_factor * x[i][j];
                }
                b += lr_factor;
            }
        }

        double acc = calculateAccuracy(x, y, w, b);
        cout << "Epoch " << e << ", Accuracy: " << fixed << setprecision(2) << acc << "%" << endl;

        if (acc > max_accuracy) {
            max_accuracy = acc;
            best_weight = w;
            best_bias = b;
            best_e = e;
        }

        if (!updated) break;
    }
}

int main() {
    vector<vector<double>> x;
    vector<int> y;
    string filename = "hard_dataset.csv";

    auto start_time = high_resolution_clock::now();

    if (!readData(filename, x, y)) {
        return 1;
    }

    if (x.empty()) {
        cerr << "No data loaded." << endl;
        return 1;
    }

    double learning_rate = 0.05;
    int max_epoch = 1000;

    vector<double> best_weight;
    double best_bias;
    double max_accuracy;
    int best_e;

    trainPerceptron(x, y, learning_rate, max_epoch, best_weight, best_bias, max_accuracy, best_e);

    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end_time - start_time);

    cout << "Max accuracy: " << fixed << setprecision(2) << max_accuracy << "% at epoch " << best_e << endl;
    cout << "Total time taken: " << duration.count() << " milliseconds" << endl;

    return 0;
}