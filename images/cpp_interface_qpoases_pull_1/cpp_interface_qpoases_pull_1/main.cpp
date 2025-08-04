#include <iostream>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <proxsuite/proxqp/dense/dense.hpp>
#include <proxsuite/linalg/veg/util/dbg.hpp>
#include <proxsuite/proxqp/utils/random_qp_problems.hpp>
#include <dqrobotics/solvers/DQ_PROXQPSolver.h>
#include <dqrobotics/solvers/DQ_QPOASESSolver.h>


using namespace proxsuite;

int main()
{

    double sparsity_factor{0.15};
    double eps_abs{1e-9};
    proxqp::utils::rand::set_seed(1);

    std::vector<double> error_list;
    int i=0;
    for (proxqp::isize dim = 10; dim <= 200; dim += 10)
    {
        std::cout<<"---------------"<<std::endl;
        std::cout<<"Dimension: "<<dim<<std::endl;
        proxqp::isize n_eq(dim/2);
        proxqp::isize n_in(dim);
        double strong_convexity_factor{1.e-2};
        proxqp::dense::Model<double> qp_random = proxqp::utils::dense_box_constrained_qp(
            dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

        MatrixXd H = qp_random.H;
        VectorXd f = qp_random.g;
        MatrixXd Aeq = qp_random.A;
        VectorXd beq = qp_random.b;
        MatrixXd A = qp_random.C;
        VectorXd b = qp_random.u;
        VectorXd l = -1*VectorXd::Ones(b.size())*INFINITY;

        proxsuite::proxqp::dense::QP<double> qp(dim, n_eq, n_in);
        qp.init(H,f,Aeq,beq,A,l,b); // initialize the model
        qp.solve();
        auto u_proxqp = qp.results.x;
        std::cout<<"u_proxqp : "<<(u_proxqp).transpose()<<std::endl;

        DQ_robotics::DQ_QPOASESSolver qpoases_solver;
        qpoases_solver.set_equality_constraints_tolerance(eps_abs);

        auto u_qpoases = qpoases_solver.solve_quadratic_program(H, f, A, b, Aeq, beq);
        std::cout<<"u_qpoases: "<<(u_qpoases).transpose()<<std::endl;

        error_list.emplace_back((u_proxqp-u_qpoases).norm());

        std::cout<<"error: "<<error_list.at(i)<<std::endl;
        std::cout<<"---------------"<<std::endl;

        i++;
    }
    VectorXd vec_error_list = Eigen::Map<VectorXd>(error_list.data(), error_list.size());
    std::cout<<"error list: "<<std::endl;
    std::cout<<vec_error_list.transpose()<<std::endl;
    return 0;
}
