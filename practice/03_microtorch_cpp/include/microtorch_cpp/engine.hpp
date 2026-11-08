#ifndef __MICROTORCH_CPP_ENGINE_HPP__
#define __MICROTORCH_CPP_ENGINE_HPP__

#include <eigen3/Eigen/Dense>
#include <string>
#include <utility>
#include <memory>
#include <functional>
#include <stdexcept>

namespace microtorch_cpp {

namespace engine {

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    using SharedPtr = std::shared_ptr<Tensor>;
    using WeakPtr = std::weak_ptr<Tensor>;
    using UniquePtr = std::unique_ptr<Tensor>;

    Tensor(const Eigen::MatrixXd& data,
           bool requires_grad = true,
           std::pair<Tensor::WeakPtr, Tensor::WeakPtr> parent = {},
           std::string op = "")
        : data_(data), grad_(Eigen::MatrixXd::Zero(data.rows(), data.cols())), requires_grad_(requires_grad),
          parent_(std::move(parent)), op_(std::move(op)) {}

    Tensor(double value,
           bool requires_grad = false,
           std::pair<Tensor::WeakPtr, Tensor::WeakPtr> parent = {},
           std::string op = "")
        : Tensor(Eigen::MatrixXd::Constant(1, 1, value), requires_grad, std::move(parent), std::move(op)) {}

    static Eigen::MatrixXd asSameShape(const Eigen::MatrixXd& grad, const Eigen::MatrixXd& target);

    // Accessors
    const Eigen::MatrixXd& data() { return data_; }
    const Eigen::MatrixXd& grad() { return grad_; }
    void setData(const Eigen::MatrixXd& data) { data_ = data; }
    void setGrad(const Eigen::MatrixXd& grad) { grad_ = grad; }
    bool requiresGrad() const { return requires_grad_; }
    std::pair<Tensor::WeakPtr, Tensor::WeakPtr> parent() const { return parent_; }
    std::string op() const { return op_; }
    std::function<void()> backward_fn() const { return backward_fn_; }
    void setBackwardFn(std::function<void()> backward_fn) { backward_fn_ = std::move(backward_fn); }

    // operations
    Tensor::SharedPtr add(const Tensor::SharedPtr& other);
    Tensor::SharedPtr add(double scalar) { return this->add(std::make_shared<Tensor>(scalar, false)); }
    Tensor::SharedPtr mult(const Tensor::SharedPtr& other);
    Tensor::SharedPtr mult(double scalar) { return this->mult(std::make_shared<Tensor>(scalar, false)); }
    Tensor::SharedPtr div(const Tensor::SharedPtr& other) { return this->mult(other->pow(-1.0)); }
    Tensor::SharedPtr div(double scalar) { return this->mult(std::make_shared<Tensor>(scalar, false)->pow(-1.0)); }
    Tensor::SharedPtr matmul(const Tensor::SharedPtr& other);
    Tensor::SharedPtr pow(double exponent);

    Tensor::SharedPtr operator+(const Tensor::SharedPtr& other) { return this->add(other); }
    Tensor::SharedPtr operator+(double scalar) { return this->add(scalar); }
    friend Tensor::SharedPtr operator+(double scalar, const Tensor::SharedPtr& tensor) { return tensor->add(scalar); }

    Tensor::SharedPtr operator-(const Tensor::SharedPtr& other) { return this->add(other->mult(-1.0)); }
    Tensor::SharedPtr operator-(double scalar) { return this->add(-scalar); }
    friend Tensor::SharedPtr operator-(double scalar, const Tensor::SharedPtr& tensor) { return tensor->add(-scalar); }
    Tensor::SharedPtr operator-() { return this->mult(-1.0); }

    Tensor::SharedPtr operator*(const Tensor::SharedPtr& other) { return this->mult(other); }
    Tensor::SharedPtr operator*(double scalar) { return this->mult(scalar); }
    friend Tensor::SharedPtr operator*(double scalar, const Tensor::SharedPtr& tensor) { return tensor->mult(scalar); }

    Tensor::SharedPtr operator/(const Tensor::SharedPtr& other) { return this->div(other); }
    Tensor::SharedPtr operator/(double scalar) { return this->div(scalar); }
    friend Tensor::SharedPtr operator/(double scalar, const Tensor::SharedPtr& tensor) {
        return scalar * tensor->pow(-1.0);
    }

private:
    Eigen::MatrixXd data_;
    Eigen::MatrixXd grad_;
    bool requires_grad_;
    std::pair<Tensor::WeakPtr, Tensor::WeakPtr> parent_;
    std::string op_;
    std::function<void()> backward_fn_;
};

}  // namespace engine
}  // namespace microtorch_cpp

#endif  // __MICROTORCH_CPP_ENGINE_HPP__
