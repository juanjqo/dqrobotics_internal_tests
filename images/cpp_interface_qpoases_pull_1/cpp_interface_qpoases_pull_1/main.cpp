#include <iostream>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <proxsuite/proxqp/dense/dense.hpp>
#include <proxsuite/linalg/veg/util/dbg.hpp>
#include <proxsuite/proxqp/utils/random_qp_problems.hpp>
#include <dqrobotics/solvers/DQ_PROXQPSolver.h>
#include <dqrobotics/solvers/DQ_QPOASESSolver.h>

using namespace std;
using namespace proxsuite;

int main()
{

    double sparsity_factor{0.15};
    double eps_abs{1e-9};
    proxqp::utils::rand::set_seed(1);
    for (proxqp::isize dim = 10; dim < 300; dim += 10)
    {
        std::cout<<"---------------"<<std::endl;
        std::cout<<"Dimension: "<<dim<<std::endl;
        proxqp::isize n_eq(dim);
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
        MatrixXd Aeq_;
        VectorXd beq_;

        DQ_robotics::DQ_PROXQPSolver proxqp_solver;
        auto u_proxqp = proxqp_solver.solve_quadratic_program(H, f, A, b, Aeq, beq);
        std::cout<<"u_proxqp : "<<(u_proxqp).transpose()<<std::endl;

        DQ_robotics::DQ_QPOASESSolver qpoases_solver;

        auto u_qpoases = qpoases_solver.solve_quadratic_program(H, f, A, b, Aeq, beq);
        std::cout<<"u_qpoases: "<<(u_qpoases).transpose()<<std::endl;

        std::cout<<"error: "<<(u_proxqp-u_qpoases).norm()<<std::endl;
        std::cout<<"---------------"<<std::endl;



        /*
        proxqp::dense::QP<double> qp{ dim, n_eq, n_in }; // creating QP object
        qp.settings.eps_abs = eps_abs;
        qp.settings.eps_rel = 0;

        qp.init(qp_random.H,
                qp_random.g,
                qp_random.A,
                qp_random.b,
                qp_random.C,
                -1*Eigen::VectorXd::Ones(dim)*INFINITY,
                qp_random.u);
        qp.solve();
        double pri_res = std::max(
            (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
            (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
             helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
                .lpNorm<Eigen::Infinity>());
        double dua_res = (qp_random.H * qp.results.x + qp_random.g +
                     qp_random.A.transpose() * qp.results.y +
                     qp_random.C.transpose() * qp.results.z)
                        .lpNorm<Eigen::Infinity>();
        std::cout << "------solving qp with dim: " << dim << " neq: " << n_eq
                  << " nin: " << n_in << std::endl;
        std::cout << "primal residual: " << pri_res << std::endl;
        std::cout << "dual residual: " << dua_res << std::endl;
        std::cout << "total number of iteration: " << qp.results.info.iter
                  << std::endl;
*/
    }
    return 0;
}
