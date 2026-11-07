#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "microtorch_cpp/engine.hpp"

using namespace microtorch_cpp::engine;

// ==========================================
// 測試 Group 1: 輔助函數 asSameShape 驗證
// ==========================================
TEST(TensorShapeTest, AsSameShape_SameDimension) {
    Eigen::MatrixXd grad = Eigen::MatrixXd::Ones(2, 2);
    Eigen::MatrixXd target = Eigen::MatrixXd::Zero(2, 2);
    Eigen::MatrixXd result = Tensor::asSameShape(grad, target);
    EXPECT_TRUE(result.isApprox(grad));
}

TEST(TensorShapeTest, AsSameShape_BroadcastScalarToMatrix) {
    Eigen::MatrixXd grad = (Eigen::MatrixXd(1, 1) << 3.0).finished();
    Eigen::MatrixXd target = Eigen::MatrixXd::Zero(2, 2);
    Eigen::MatrixXd result = Tensor::asSameShape(grad, target);
    Eigen::MatrixXd expected = Eigen::MatrixXd::Constant(2, 2, 3.0);
    EXPECT_TRUE(result.isApprox(expected));
}

TEST(TensorShapeTest, AsSameShape_ReduceMatrixToScalar) {
    Eigen::MatrixXd grad = (Eigen::MatrixXd(2, 2) << 1.0, 2.0, 3.0, 4.0).finished();
    Eigen::MatrixXd target = Eigen::MatrixXd::Zero(1, 1);
    Eigen::MatrixXd result = Tensor::asSameShape(grad, target);
    Eigen::MatrixXd expected = (Eigen::MatrixXd(1, 1) << 10.0).finished();  // 1+2+3+4
    EXPECT_TRUE(result.isApprox(expected));
}

TEST(TensorShapeTest, AsSameShape_IncompatibleThrows) {
    Eigen::MatrixXd grad = Eigen::MatrixXd::Ones(1, 3);
    Eigen::MatrixXd target = Eigen::MatrixXd::Zero(2, 2);
    // 預期拋出 runtime_error
    EXPECT_THROW(Tensor::asSameShape(grad, target), std::runtime_error);
}

TEST(TensorAddTest, MatrixMatrixAdd) {
    auto x = std::make_shared<Tensor>((Eigen::MatrixXd(2, 2) << 1, 2, 3, 4).finished(), true);
    auto y = std::make_shared<Tensor>((Eigen::MatrixXd(2, 2) << 5, 6, 7, 8).finished(), true);

    auto out = x->add(y);
    Eigen::MatrixXd expected = (Eigen::MatrixXd(2, 2) << 6, 8, 10, 12).finished();
    EXPECT_TRUE(out->data().isApprox(expected));

    out->setGrad(Eigen::MatrixXd::Ones(2, 2));
    out->backward_fn()();

    EXPECT_TRUE(x->grad().isApprox(Eigen::MatrixXd::Ones(2, 2)));
    EXPECT_TRUE(y->grad().isApprox(Eigen::MatrixXd::Ones(2, 2)));
}

TEST(TensorAddTest, MatrixScalarAdd_Broadcasting) {
    auto x = std::make_shared<Tensor>((Eigen::MatrixXd(2, 2) << 1, 2, 3, 4).finished(), true);
    auto y = std::make_shared<Tensor>((Eigen::MatrixXd(1, 1) << 10.0).finished(), true);  // 純量

    auto out = x->add(y);  // Edge case: 前向純量廣播
    Eigen::MatrixXd expected = (Eigen::MatrixXd(2, 2) << 11, 12, 13, 14).finished();
    EXPECT_TRUE(out->data().isApprox(expected));

    out->setGrad(Eigen::MatrixXd::Ones(2, 2));
    out->backward_fn()();

    EXPECT_TRUE(x->grad().isApprox(Eigen::MatrixXd::Ones(2, 2)));

    Eigen::MatrixXd expected_dy = (Eigen::MatrixXd(1, 1) << 4.0).finished();
    EXPECT_TRUE(y->grad().isApprox(expected_dy));
}

TEST(TensorMultTest, MatrixMatrixMult) {
    auto x = std::make_shared<Tensor>((Eigen::MatrixXd(2, 2) << 1, 2, 3, 4).finished(), true);
    auto y = std::make_shared<Tensor>((Eigen::MatrixXd(2, 2) << 2, 2, 2, 2).finished(), true);

    auto out = x->mult(y);
    Eigen::MatrixXd expected = (Eigen::MatrixXd(2, 2) << 2, 4, 6, 8).finished();
    EXPECT_TRUE(out->data().isApprox(expected));

    out->setGrad(Eigen::MatrixXd::Ones(2, 2));
    out->backward_fn()();

    // dx = dout * y
    EXPECT_TRUE(x->grad().isApprox(y->data()));
    // dy = dout * x
    EXPECT_TRUE(y->grad().isApprox(x->data()));
}

TEST(TensorMultTest, MatrixScalarMult_Broadcasting) {
    auto x = std::make_shared<Tensor>((Eigen::MatrixXd(2, 2) << 1, 2, 3, 4).finished(), true);
    auto y = std::make_shared<Tensor>((Eigen::MatrixXd(1, 1) << 2.0).finished(), true);

    auto out = x->mult(y);
    Eigen::MatrixXd expected = (Eigen::MatrixXd(2, 2) << 2, 4, 6, 8).finished();
    EXPECT_TRUE(out->data().isApprox(expected));

    out->setGrad(Eigen::MatrixXd::Ones(2, 2));
    out->backward_fn()();

    EXPECT_TRUE(x->grad().isApprox(Eigen::MatrixXd::Constant(2, 2, 2.0)));

    Eigen::MatrixXd expected_dy = (Eigen::MatrixXd(1, 1) << 10.0).finished();
    EXPECT_TRUE(y->grad().isApprox(expected_dy));
}

TEST(TensorMatmulTest, VectorVectorDotProduct) {
    auto x = std::make_shared<Tensor>((Eigen::MatrixXd(1, 3) << 1, 2, 3).finished(), true);
    auto y = std::make_shared<Tensor>((Eigen::MatrixXd(3, 1) << 4, 5, 6).finished(), true);

    auto out = x->matmul(y);
    Eigen::MatrixXd expected = (Eigen::MatrixXd(1, 1) << 32.0).finished();  // 4 + 10 + 18
    EXPECT_TRUE(out->data().isApprox(expected));

    out->setGrad((Eigen::MatrixXd(1, 1) << 2.0).finished());
    out->backward_fn()();

    // dX = dO * Y^T
    Eigen::MatrixXd expected_dx = (Eigen::MatrixXd(1, 3) << 8, 10, 12).finished();
    // dY = X^T * dO
    Eigen::MatrixXd expected_dy = (Eigen::MatrixXd(3, 1) << 2, 4, 6).finished();

    EXPECT_TRUE(x->grad().isApprox(expected_dx));
    EXPECT_TRUE(y->grad().isApprox(expected_dy));
}