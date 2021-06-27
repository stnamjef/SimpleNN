#pragma once
#include "common.h"

namespace simple_nn
{
	class Optimizer
	{
	private:
		float lr_;
		float decay_;
	public:
		Optimizer(float lr, float decay) : lr_(lr), decay_(decay) {}
		float lr() { return lr_; }
		float decay() { return decay_; }
	};

	class SGD : public Optimizer
	{
	public:
		SGD(float lr, float decay) : Optimizer(lr, decay) {}
	};
}