#include <iostream>
#include "Vector.h"
#include "Dense.h"
#include "Conv2D.h"
#include "Pool2D.h"
#include "CNN.h"
using namespace std;

int main()
{
	CNN model;
	model.add(new Conv2D({ 5, 5, 6 }, 1, 1, { 28, 28, 1 }));
	model.add(new Pooling2D({ 2, 2, 6 }, 2, { 28, 28, 6 }));
	model.add(new Dense(120));
	model.add(new Dense(84));
	model.add(new Dense(10));

	model.print_network();

	return 0;
}