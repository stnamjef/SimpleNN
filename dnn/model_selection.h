#pragma once
#include <vector>
#include <ctime>
using namespace std;

template<class T>
Vector<double> evaluate_model(T& model, double l_rate, int n_epoch, int batch_size, const Vector<Vector<double>>& X,
	const Vector<int>& Y, int n_fold);

void split_to_folds(int n_row, int n_fold, vector<vector<int>>& folds);

int unique_random(vector<int>& unique, int range);

Vector<Vector<double>> train_X(const Vector<Vector<double>>& X, const vector<vector<int>>& folds, int except);

Vector<int> train_Y(const Vector<int>& Y, const vector<vector<int>>& folds, int except);

Vector<Vector<double>> test_X(const Vector<Vector<double>>& X, const vector<vector<int>>& folds, int include);

Vector<int> test_Y(const Vector<int>& Y, const vector<vector<int>>& folds, int include);

double calc_accuracy(const Vector<int>& actual, const Vector<int> predicts);

template<class T>
Vector<double> evaluate_model(T& model, double l_rate, int n_epoch, int batch_size, const Vector<Vector<double>>& X,
	const Vector<int>& Y, int n_fold)
{
	vector<vector<int>> folds;
	split_to_folds((int)X.size(), n_fold, folds);

	Vector<double> accuracies(n_fold);
	for (int i = 0; i < n_fold; i++)
	{
		model.fit(train_X(X, folds, i), train_Y(Y, folds, i), l_rate, n_epoch, batch_size);
		Vector<int> predicts = model.predict(test_X(X, folds, i));
		accuracies[i] = calc_accuracy(test_Y(Y, folds, i), predicts);
	}
	return accuracies;
}

void split_to_folds(int n_row, int n_fold, vector<vector<int>>& folds)
{
	srand((unsigned)time(NULL));
	folds.resize(n_fold, vector<int>());

	vector<int> unique;
	int fold_size = n_row / n_fold;

	for (int i = 0; i < n_fold; i++)
		for (int j = 0; j < fold_size; j++)
			folds[i].push_back(unique_random(unique, n_row));
}

int unique_random(vector<int>& unique, int range)
{
	bool isOverlap;
	int num;
	do
	{
		num = rand() % range;
		isOverlap = false;
		for (int i = 0; i < unique.size(); i++)
			if (unique[i] == num)
			{
				isOverlap = true;
				break;
			}
	} while (isOverlap);
	unique.push_back(num);
	return num;
}

Vector<Vector<double>> train_X(const Vector<Vector<double>>& X, const vector<vector<int>>& folds, int except)
{
	int size = ((int)folds.size() - 1) * (int)folds[0].size();
	Vector<Vector<double>> xTrain(size, Vector<double>(X[0].size()));

	int i = 0;
	for (int j = 0; j < folds.size(); j++)
	{
		if (j == except)
			continue;
		for (int idx : folds[j])
		{
			xTrain[i] = X[idx];
			i++;
		}
	}
	return xTrain;
}

Vector<int> train_Y(const Vector<int>& Y, const vector<vector<int>>& folds, int except)
{
	int size = ((int)folds.size() - 1) * (int)folds[0].size();
	Vector<int> yTrain(size);

	int i = 0;
	for (int j = 0; j < folds.size(); j++)
	{
		if (j == except)
			continue;
		for (int idx : folds[j])
		{
			yTrain[i] = Y[idx];
			i++;
		}
	}
	return yTrain;
}

Vector<Vector<double>> test_X(const Vector<Vector<double>>& X, const vector<vector<int>>& folds, int include)
{
	int size = (int)folds[0].size();
	Vector<Vector<double>> xTest(size, Vector<double>((int)X[0].size()));

	int i = 0;
	for (int idx : folds[include])
	{
		xTest[i] = X[idx];
		i++;
	}
	return xTest;
}

Vector<int> test_Y(const Vector<int>& Y, const vector<vector<int>>& folds, int include)
{
	int size = (int)folds[0].size();
	Vector<int> yTest(size);

	int i = 0;
	for (int idx : folds[include])
	{
		yTest[i] = Y[idx];
		i++;
	}
	return yTest;
}

double calc_accuracy(const Vector<int>& actual, const Vector<int> predicts)
{
	double correct = 0;
	for (int i = 0; i < actual.size(); i++)
		if (actual[i] == predicts[i])
			correct++;
	return correct / actual.size();
}