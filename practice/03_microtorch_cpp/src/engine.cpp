#include "microtorch_cpp/engine.hpp"

namespace microtorch_cpp {
namespace engine {

Eigen::MatrixXd Tensor::asSameShape(const Eigen::MatrixXd& grad, const Eigen::MatrixXd& target) {
    // Return grad if it already has the same shape as target
    if (grad.rows() == target.rows() && grad.cols() == target.cols()) {
        return grad;
    }

    // If grad is a scalar, broadcast it to the shape of target
    // ex. target = [[2, 2]], grad = [1] -> return [[1, 1]]
    if (grad.size() == 1) {
        return Eigen::MatrixXd::Constant(target.rows(), target.cols(), grad(0, 0));
    }

    // If target is a scalar, sum grad to a scalar
    // ex. target = [2], grad = [[1, 1]] -> return [2]
    if (target.rows() == 1 && target.cols() == 1) {
        return grad.sum() * Eigen::MatrixXd::Ones(1, 1);
    }

    throw std::runtime_error("Incompatible grad shape");
}

Tensor::SharedPtr Tensor::add(const Tensor::SharedPtr& other) {
    Tensor::SharedPtr self = shared_from_this();
    Eigen::MatrixXd result_data;

    if (data_.size() == 1) {
        result_data = other->data().array() + data_(0, 0);
    } else if (other->data().size() == 1) {
        result_data = data_.array() + other->data()(0, 0);
    } else {
        result_data = data_ + other->data();
    }

    auto result = std::make_shared<Tensor>(result_data,
                                           this->requiresGrad() || other->requiresGrad(),
                                           std::make_pair(self, other),
                                           "add");

    std::weak_ptr<Tensor> weak_result = result;
    result->setBackwardFn([self, other, weak_result]() {
        auto res_ptr = weak_result.lock();
        if (!res_ptr)
            return;

        if (self->requiresGrad()) {
            Eigen::MatrixXd dx = asSameShape(res_ptr->grad(), self->data());
            if (self->grad().size() == 0)
                self->setGrad(dx);
            else
                self->setGrad(self->grad() + dx);
        }
        if (other->requiresGrad()) {
            Eigen::MatrixXd dy = asSameShape(res_ptr->grad(), other->data());
            if (other->grad().size() == 0)
                other->setGrad(dy);
            else
                other->setGrad(other->grad() + dy);
        }
    });

    return result;
}

Tensor::SharedPtr Tensor::mult(const Tensor::SharedPtr& other) {
    Tensor::SharedPtr self = shared_from_this();
    Eigen::MatrixXd result_data;

    if (data_.size() == 1) {
        result_data = other->data() * data_(0, 0);
    } else if (other->data().size() == 1) {
        result_data = data_ * other->data()(0, 0);
    } else {
        result_data = data_.cwiseProduct(other->data());
    }

    auto result = std::make_shared<Tensor>(result_data,
                                           this->requiresGrad() || other->requiresGrad(),
                                           std::make_pair(self, other),
                                           "mult");

    std::weak_ptr<Tensor> weak_result = result;
    result->setBackwardFn([self, other, weak_result]() {
        auto res_ptr = weak_result.lock();
        if (!res_ptr)
            return;

        if (self->requiresGrad()) {
            Eigen::MatrixXd local_grad;
            if (other->data().size() == 1) {
                local_grad = res_ptr->grad() * other->data()(0, 0);
            } else {
                local_grad = res_ptr->grad().cwiseProduct(other->data());
            }
            Eigen::MatrixXd dx = asSameShape(local_grad, self->data());

            if (self->grad().size() == 0)
                self->setGrad(dx);
            else
                self->setGrad(self->grad() + dx);
        }
        if (other->requiresGrad()) {
            Eigen::MatrixXd local_grad;
            if (self->data().size() == 1) {
                local_grad = res_ptr->grad() * self->data()(0, 0);
            } else {
                local_grad = res_ptr->grad().cwiseProduct(self->data());
            }

            Eigen::MatrixXd dy = asSameShape(local_grad, other->data());

            if (other->grad().size() == 0)
                other->setGrad(dy);
            else
                other->setGrad(other->grad() + dy);
        }
    });

    return result;
}

Tensor::SharedPtr Tensor::matmul(const Tensor::SharedPtr& other) {
    Tensor::SharedPtr self = shared_from_this();
    Eigen::MatrixXd result_data = data_ * other->data();

    auto result = std::make_shared<Tensor>(result_data,
                                           this->requiresGrad() || other->requiresGrad(),
                                           std::make_pair(self, other),
                                           "matMul");

    std::weak_ptr<Tensor> weak_result = result;

    result->setBackwardFn([self, other, weak_result]() {
        auto res_ptr = weak_result.lock();
        if (!res_ptr)
            return;

        const auto& x = self->data();
        const auto& y = other->data();
        const auto& dO = res_ptr->grad();

        if (self->requiresGrad()) {
            Eigen::MatrixXd dx = dO * y.transpose();
            if (self->grad().size() == 0) {
                self->setGrad(dx);
            } else {
                self->setGrad(self->grad() + dx);
            }
        }

        if (other->requiresGrad()) {
            Eigen::MatrixXd dw = x.transpose() * dO;
            if (other->grad().size() == 0) {
                other->setGrad(dw);
            } else {
                other->setGrad(other->grad() + dw);
            }
        }
    });

    return result;
}

}  // namespace engine
}  // namespace microtorch_cpp