#include <picasso/objective.hpp>
// #include <R.h>

namespace picasso {
GLMObjective::GLMObjective(const double *xmat, const double *y, int n, int d)
    : ObjFunction(xmat, y, n, d) {
  a = 0.0;
  g = 0.0;

  p.resize(n);
  w.resize(n);

  r.resize(n);
  wXX.resize(d);
}

GLMObjective::GLMObjective(const double *xmat, const double *y, int n, int d,
                           bool include_intercept)
    : ObjFunction(xmat, y, n, d) {
  a = 0.0;
  g = 0.0;

  p.resize(n);
  w.resize(n);
  r.resize(n);

  wXX.resize(d);

}

double GLMObjective::coordinate_descent(RegFunction *regfunc, int idx) {
  g = 0.0;
  a = 0.0;

  double tmp;

  static int subsampleidx = 0;
  int subsample_portion = 1;
  subsampleidx++;
  subsampleidx = subsampleidx%subsample_portion;
  // Sub hessian
  for (int i = (n*subsampleidx)/subsample_portion; i < (n*subsampleidx+n)/subsample_portion; i++) {
    tmp = w[i] * X[idx][i] * X[idx][i];
    g += tmp * model_param.beta[idx] + r[i] * X[idx][i];
    a += tmp;
  }
  g = g / n * subsample_portion;
  a = a / n * subsample_portion;

  // g = (<wXX, model_param.beta> + <r, X>)/n
  // a = sum(wXX)/n
  // Full hessian
  // for (int i = 0; i < n; i++) {
  //   tmp = w[i] * X[idx][i] * X[idx][i];
  //   g += tmp * model_param.beta[idx] + r[i] * X[idx][i];
  //   a += tmp;
  // }
  // g = g / n;
  // a = a / n;

  tmp = model_param.beta[idx];
  model_param.beta[idx] = regfunc->threshold(g) / a;

  tmp = model_param.beta[idx] - tmp;
  if (fabs(tmp) > 1e-8) {
    // Xb += delta*X[idx*n]
    for (int i = 0; i < n; i++) Xb[i] = Xb[i] + tmp * X[idx][i];
    // sum_r = 0.0;
    // r -= delta*w*X
    for (int i = 0; i < n; i++) {
      r[i] = r[i] - w[i] * X[idx][i] * tmp;
      // sum_r += r[i];
    }
  }
  return (model_param.beta[idx]);
}

double GLMObjective::coordinate_descent_l1_newton(double lambda, int idx) {
  g = 0.0;
  a = 0.0;

  double tmp;
  // g = (<wXX, model_param.beta> + <r, X>)/n
  // a = sum(wXX)/n
  for (int i = 0; i < n; i++) {
    tmp = w[i] * X[idx][i] * X[idx][i];
    g += tmp * model_param.beta[idx] + r[i] * X[idx][i];
    a += tmp;
  }
  g = g / n;
  a = a / n;

  tmp = model_param.beta[idx];

  // soft threshold
  {
    if (g > lambda)
      g = g - lambda;
    else if (g < -lambda)
      g = g + lambda;
    else
      g = 0;
  }

  model_param.beta[idx] = g / a;

  tmp = model_param.beta[idx] - tmp;
  // Rprintf("\t tmp: %lf\n", tmp);
  if (fabs(tmp) > 1e-8) {
    // Xb += delta*X[idx*n]
    for (int i = 0; i < n; i++) Xb[i] = Xb[i] + tmp * X[idx][i];
    // sum_r = 0.0;
    // r -= delta*w*X
    for (int i = 0; i < n; i++) {
      r[i] = r[i] - w[i] * X[idx][i] * tmp;
      // sum_r += r[i];
    }
  }
  return (model_param.beta[idx]);
}

void GLMObjective::intercept_update() {
  sum_r = 0.0;
  for (int i = 0; i < n; i++) sum_r += r[i];
  double tmp = sum_r / sum_w;
  model_param.intercept += tmp;

  // sum_r = 0.0;
  for (int i = 0; i < n; i++) {
    r[i] = r[i] - tmp * w[i];
    // sum_r += r[i];
  }
}

/*
void GLMObjective::set_model_param(ModelParam &other_param,
                                   const std::vector<double> &old_Xb) {
  model_param = other_param;


  for (int i = 0; i < n; i++) {
    Xb[i] = 0.0;
    for (int j = 0; j < d; j++) Xb[i] += X[j][i] * model_param.beta[j];
  }
  for (int i = 0; i < n; i++) Xb[i] = old_Xb[i];
}
*/

void GLMObjective::update_auxiliary() {
  update_key_aux();
  sum_w = 0.0;
  for (int i = 0; i < n; i++) {
    r[i] = Y[i] - p[i];
    sum_w += w[i];
  }

  /*
    for (int idx = 0; idx < d; idx++) {
      wXX[idx] = 0.0;
      for (int i = 0; i < n; i++) wXX[idx] += w[i] * X[idx][i] * X[idx][i];
    }*/
}

void GLMObjective::update_gradient(int idx) {
  gr[idx] = 0.0;
  for (int i = 0; i < n; i++) gr[idx] += r[i] * X[idx][i] / n;
  // if(idx==5) Rprintf("grad[j] = %lf; Y[5] = %lf; p[5] = %lf; X[idx][5] = %lf; n = %d; \n",gr[idx], Y[5], p[5], X[idx][5], n);
}

double GLMObjective::get_local_change(double old, int idx) {
  if (idx >= 0) {
    double tmp = old - model_param.beta[idx];
    double wXX_idx = 0.0;
    for (int i = 0; i < n; i++) wXX_idx += w[i] * X[idx][i] * X[idx][i];
    return (wXX_idx * tmp * tmp / (2 * n));
  } else {
    double tmp = old - model_param.intercept;
    return (sum_w * tmp * tmp / (2 * n));
  }
}

LogisticObjective::LogisticObjective(const double *xmat, const double *y, int n,
                                     int d)
    : GLMObjective(xmat, y, n, d) {
  model_param.intercept = 0.0;
  update_auxiliary();

  for (int i = 0; i < d; i++) update_gradient(i);

  deviance = fabs(eval());
};

LogisticObjective::LogisticObjective(const double *xmat, const double *y, int n,
                                     int d, bool include_intercept)
    : GLMObjective(xmat, y, n, d, include_intercept) {
  model_param.intercept = 0.0;
  update_auxiliary();
  for (int i = 0; i < d; i++) update_gradient(i);

  // update_auxiliary(); (delete by haoming)

  deviance = fabs(eval());
};

void LogisticObjective::update_key_aux() {
  for (int i = 0; i < n; i++) {
    p[i] = 1.0 / (1.0 + exp(-model_param.intercept - Xb[i]));
    // Rprintf("p[i] = %lf\n", p[i]);
    w[i] = p[i] * (1 - p[i]);
  }
}

double LogisticObjective::eval() {
  double v = 0.0;
  for (int i = 0; i < n; i++) v -= Y[i] * (model_param.intercept + Xb[i]);

  for (int i = 0; i < n; i++)
    if (p[i] > 1e-8) v -= (log(p[i]) - model_param.intercept - Xb[i]);

  return (v / n);
}

PoissonObjective::PoissonObjective(const double *xmat, const double *y, int n,
                                   int d)
    : GLMObjective(xmat, y, n, d) {
  model_param.intercept = 0.0;
  update_auxiliary();

  for (int i = 0; i < d; i++) update_gradient(i);

  deviance = fabs(eval());
};

PoissonObjective::PoissonObjective(const double *xmat, const double *y, int n,
                                   int d, bool include_intercept)
    : GLMObjective(xmat, y, n, d, include_intercept) {
  model_param.intercept = 0.0;
  update_auxiliary();
  for (int i = 0; i < d; i++) update_gradient(i);

  // update_auxiliary(); (delete by haoming)

  deviance = fabs(eval());
};

void PoissonObjective::update_key_aux() {
  for (int i = 0; i < n; i++) {
    p[i] = exp(model_param.intercept + Xb[i]);
    w[i] = p[i];
  }
}

double PoissonObjective::eval() {
  double v = 0.0;
  for (int i = 0; i < n; i++)
    v = v + p[i] - Y[i] * (model_param.intercept + Xb[i]);
  return (v / n);
}





GaussianObjective::GaussianObjective(const double *xmat, const double *y, int n,
                                     int d)
    : GLMObjective(xmat, y, n, d) {
  model_param.intercept = 0.0;
  update_auxiliary();

  for (int i = 0; i < d; i++) update_gradient(i);

  deviance = fabs(eval());
};

GaussianObjective::GaussianObjective(const double *xmat, const double *y, int n,
                                     int d, bool include_intercept)
    : GLMObjective(xmat, y, n, d, include_intercept) {
  model_param.intercept = 0.0;
  update_auxiliary();
  for (int i = 0; i < d; i++) update_gradient(i);

  // update_auxiliary(); (delete by haoming)

  deviance = fabs(eval());
};

void GaussianObjective::update_key_aux() {
  // Rprintf("model_param.intercept: %lf; Xb[5] = %lf\n",model_param.intercept, Xb[5]);
  for (int i = 0; i < n; i++) {
    p[i] = model_param.intercept + Xb[i];
    // Rprintf("p[i] = %lf\n", p[i]);
    w[i] = 1;
  }
}

double GaussianObjective::eval() {
  double v = 0.0;
  for (int i = 0; i < n; i++) {
    double pred = model_param.intercept+Xb[i];
    v += pow(Y[i] - pred,2);
  }
  v = v / n;
  return v;
}



double GLMObjective::kkt_val(double lambda) {
  double grad_max = 0;
  for (int i = 0; i < d; i++){
    double grad = gr[i];
    if (model_param.beta[i] == 0 && fabs(grad)<lambda){
        grad = 0;
    } else {
      if (model_param.beta[i] > 0)
        grad = grad + lambda;
      else
        grad = grad - lambda;
    }
    grad = fabs(grad);
    if (grad > grad_max)
      grad_max = grad;
  }
  return(grad_max);
}

double GLMObjective::kkt_val_act(double lambda, std::vector<bool> &actset_is) {
  double grad_max = 0;
  for (int i = 0; i < d; i++)
    if (actset_is[i]){
      double grad = gr[i];
      if (model_param.beta[i] == 0 && fabs(grad)<lambda){
          grad = 0;
      } else {
        if (model_param.beta[i] > 0)
          grad = grad + lambda;
        else
          grad = grad - lambda;
      }
      grad = fabs(grad);
      if (grad > grad_max)
        grad_max = grad;
    }
  return(grad_max);
}





}  // namespace picasso
