#include <eigen3/Eigen/Core>
#include <cmath>
#include "integrator.hpp"
#include "interpolate.hpp"

Eigen::VectorXd RKF45::step(const double tk, const Eigen::VectorXd& yk,
                             const Eigen::VectorXd& f0, double h, double tol)
{
    Eigen::VectorXd yk1 = yk;
    Eigen::VectorXd zk1 = yk;

    // Clamp h to remaining interval once, before the retry loop
    if (tk + h > t_stop) {
        h = t_stop - tk;
    }

    double err;
    double s;
    do {
        if (h < this->min_step) {
            this->err_flag = true;
            this->current_step = h;
            return yk1;
        }

        // k1 reuses the pre-computed derivative f0 — no extra func() call
        Eigen::VectorXd k1 = h * f0;
        Eigen::VectorXd k2 = h * func(tk + 0.25*h,      yk + 0.25*k1);
        Eigen::VectorXd k3 = h * func(tk + 0.375*h,     yk + 3.0/32.0*k1 + 9.0/32.0*k2);
        Eigen::VectorXd k4 = h * func(tk + 12.0/13.0*h, yk + 1932.0/2197.0*k1 - 7200.0/2197.0*k2 + 7296.0/2197.0*k3);
        Eigen::VectorXd k5 = h * func(tk + h,           yk + 439.0/216.0*k1 - 8.0*k2 + 3680.0/513.0*k3 - 845.0/4104.0*k4);
        Eigen::VectorXd k6 = h * func(tk + 0.5*h,       yk - 8.0/27.0*k1 + 2.0*k2 - 3544.0/2565.0*k3 + 1859.0/4104.0*k4 - 11.0/40.0*k5);

        // RK4 estimate
        yk1 = yk + 25.0/216.0*k1 + 1408.0/2565.0*k3 + 2197.0/4104.0*k4 - 0.2*k5;

        // RK5 estimate
        zk1 = yk + 16.0/135.0*k1 + 6656.0/12825.0*k3 + 28561.0/56340.0*k4 - 9.0/50.0*k5 + 2.0/55.0*k6;

        err = (zk1 - yk1).norm();

        if (err <= tol) {
            this->current_step = h;
            return yk1;
        }

        // Reduce step size and retry
        s = pow(tol / (2.0 * err), 0.25);
        h = s * h;

    } while (h >= this->min_step);

    this->err_flag = true;
    this->current_step = h;
    return yk1;
}

IntegratorResult RKF45::integrate()
{
    // Pre-allocate with a reasonable estimate to reduce reallocations
    const int est_steps = std::max(64, static_cast<int>((t_stop - t_start) / 1.0) * 2);
    std::vector<double> t;
    std::vector<Eigen::VectorXd> sol;
    t.reserve(est_steps);
    sol.reserve(est_steps);

    Eigen::VectorXd yk = this->y0;
    double h = 1.0;
    current_step = h;

    double tk = t_start;
    t.push_back(tk);
    sol.push_back(yk);

    while (tk < t_stop - 1E-12) {
        // Compute f0 once per accepted step
        Eigen::VectorXd f0 = func(tk, yk);

        yk = step(tk, yk, f0, current_step, tol);

        if (err_flag) break;

        tk += current_step;
        t.push_back(tk);
        sol.push_back(yk);

        // Grow modestly for the next step
        current_step = std::min(current_step * 1.5, t_stop - tk);
    }

    // Convert to Eigen matrices
    const int n = static_cast<int>(t.size());
    Eigen::MatrixXd sol_matrix(y0.size(), n);
    for (int i = 0; i < n; i++) {
        sol_matrix.col(i) = sol[i];
    }
    Eigen::VectorXd t_vec = Eigen::Map<Eigen::VectorXd>(t.data(), n);

    return IntegratorResult{t_vec, sol_matrix};
}

IntegratorResult RKF45::integrate(Eigen::VectorXd& teval)
{
    const int n_eval  = static_cast<int>(teval.size());
    const int n_state = static_cast<int>(y0.size());

    Eigen::MatrixXd sol_eval(n_state, n_eval);

    Eigen::VectorXd yk = y0;
    double tk = t_start;

    double h = 1.0;
    current_step = h;

    int eval_idx = 0;

    // Handle evaluation times at t_start
    while (eval_idx < n_eval && std::abs(teval(eval_idx) - tk) < 1E-12) {
        sol_eval.col(eval_idx) = yk;
        eval_idx++;
    }

    while (tk < t_stop - 1E-12 && eval_idx < n_eval) {
        // Derivative at start of step (computed once, reused if step is rejected)
        Eigen::VectorXd fk = func(tk, yk);

        Eigen::VectorXd yk1 = step(tk, yk, fk, current_step, tol);

        if (err_flag) break;

        double tk1 = tk + current_step;
        Eigen::VectorXd fk1 = func(tk1, yk1);

        // Fill teval points within [tk, tk1]
        while (eval_idx < n_eval && teval(eval_idx) <= tk1 + 1E-12) {
            double t_req = teval(eval_idx);

            if (std::abs(t_req - tk1) < 1E-12) {
                sol_eval.col(eval_idx) = yk1;
            } else {
                sol_eval.col(eval_idx) = hermite_interp(tk, yk, fk, tk1, yk1, fk1, t_req);
            }
            eval_idx++;
        }

        tk = tk1;
        yk = yk1;
        current_step = std::min(current_step * 1.5, t_stop - tk);
    }

    return IntegratorResult{teval, sol_eval};
}