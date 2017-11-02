#include <picasso/actnewton.hpp>
#include <picasso/objective.hpp>
#include <picasso/solver_params.hpp>
#include <picasso/solver_params.hpp>

#include <queue>
#define MAXUPDATECORDNUM 60

#include <R.h>

namespace picasso {
namespace solver {
struct CompareByFirst {
    constexpr bool operator()(std::pair<double, int> const & a,
                              std::pair<double, int> const & b) const noexcept
    { return a.first > b.first; }
};



ActNewtonSolver::ActNewtonSolver(ObjFunction *obj, PicassoSolverParams param)
    : m_param(param), m_obj(obj) {
   itercnt_path.clear();
   solution_path.clear();
}

void ActNewtonSolver::solve_lasso() {
  unsigned int d = m_obj->get_dim();
  unsigned int n = m_obj->get_sample_num();
  const std::vector<double> &lambdas = m_param.get_lambda_path();
  itercnt_path.resize(lambdas.size(), 0);

  std::vector<bool> actset_is(d, 0);
  std::vector<int> actset_idx;

  double zeta = 0.05;
  double delta = m_obj->get_deviance() * m_param.prec;

  // Loop for lambdas
  for (int lambda_id = 0; lambda_id < lambdas.size(); lambda_id++){
    ////Rprintf("lambda_id:%d\n",lambda_id);
    double lambda = lambdas[lambda_id];

    // Initialize active set
    for (int  i = 0; i < d; i++) {
        actset_is[i] = m_obj->get_model_coef(i) != 0;
    }
    int maxUpdateCord =         MAXUPDATECORDNUM;
    int numUpdateCord = 0;
    std::priority_queue<std::pair<double, int>,
          std::vector<std::pair<double, int> >,
          CompareByFirst> max_temp_pq;
    for (int k = 0; k < d; k++)
      if(!actset_is[k])
      {
          if (fabs(m_obj->get_grad(k))>=(1-zeta)*lambda) {
              numUpdateCord++;
              max_temp_pq.push(std::make_pair(fabs(m_obj->get_grad(k)), k));
              if(numUpdateCord>maxUpdateCord) max_temp_pq.pop();
          }
      }
    ////Rprintf("LaLaLa size of new active set: %d\n", max_temp_pq.size());
    while(! max_temp_pq.empty())
    {
      std::pair<double, int> pair_temp = max_temp_pq.top();
      max_temp_pq.pop();
        actset_is[pair_temp.second] = true;
    }

    int outer_loop_id = 0;
    // Outer Loop: Multistage Convex Relaxation
    while (m_obj->kkt_val(lambda)>delta*lambda){
      outer_loop_id++;
      ////Rprintf("\touter_loop_id:%d\n",outer_loop_id );

      // initialize actset_idx for faster processing
      actset_idx.clear();
      for (int j = 0; j < d; j++)
        if (actset_is[j])
          actset_idx.push_back(j);

      int inner_loop_id = 0;
      // Inner Loop: Proximal Newton on Active Set
      while (m_obj->kkt_val_act(lambda, actset_is)>delta*lambda){
        inner_loop_id++;
        bool terminate_inner_loop = true;
        ////Rprintf("\t\tinner_loop_id:%d, kkt:%lf, stopping c: %lf\n",inner_loop_id,m_obj->kkt_val_act(lambda, actset_is) , m_param.prec*lambda);
        // update coordinate
        for (int k = 0; k < actset_idx.size(); k++) {
          int idx = actset_idx[k];
          auto old_beta = m_obj->get_model_coef(idx);

          m_obj->coordinate_descent_l1_newton(lambda, idx);
          if (m_obj->get_local_change(old_beta, idx) > delta)
            terminate_inner_loop = false;
        }

        // update intercept
        if (m_param.include_intercept)
        {
            auto old_intcpt = m_obj->get_model_coef(-1);
            m_obj->intercept_update();
            if (m_obj->get_local_change(old_intcpt, -1) > delta)
              terminate_inner_loop = false;
        }

        m_obj->update_auxiliary();
        for (int i = 0; i < d; i++) m_obj->update_gradient(i);
        if(inner_loop_id>m_param.max_iter || terminate_inner_loop) break;
      } // Inner Loop

      m_obj->update_auxiliary();

      // track the number of iterations for each lambda
      itercnt_path[lambda_id] += inner_loop_id;

      // Update active set
      int maxUpdateCord = MAXUPDATECORDNUM;
      int numUpdateCord = 0;
      std::priority_queue<std::pair<double, int>,
              std::vector<std::pair<double, int> >,
              CompareByFirst> max_temp_pq;
      bool new_active_idx = false;
      for (int  k = 0; k < d; k++) {
          actset_is[k] = m_obj->get_model_coef(k) != 0;
      }
      for (int k = 0; k < d; k++)
          if (actset_is[k] == 0) {
              m_obj->update_gradient(k);
              if (fabs(m_obj->get_grad(k)) > lambda*(1-zeta)){
                  numUpdateCord++;
                  max_temp_pq.push(std::make_pair(fabs(m_obj->get_grad(k)), k));
                  if(numUpdateCord>maxUpdateCord) max_temp_pq.pop();
              }
          }
      new_active_idx = numUpdateCord>0;
      ////Rprintf("size of new active set: %d\n", max_temp_pq.size());
      while(! max_temp_pq.empty())
      {
          std::pair<double, int> pair_temp = max_temp_pq.top();
          max_temp_pq.pop();
          actset_is[pair_temp.second] = true;
          // //Rprintf("%d\n",pair_temp.second);
      }

      if(outer_loop_id>m_param.max_iter/100 || !(new_active_idx)) break;
    } // Outer Loop

    // save the solution_path for each lambda
     solution_path.push_back(m_obj->get_model_param());
  } // Loop for lambdas
  // //Rprintf("Training is over! solution_path.size:%d \n", solution_path.size());

}

void ActNewtonSolver::solve() {
  unsigned int d = m_obj->get_dim();
  unsigned int n = m_obj->get_sample_num();

  const std::vector<double> &lambdas = m_param.get_lambda_path();
  itercnt_path.resize(lambdas.size(), 0);

  double dev_thr = m_param.prec*m_obj->get_deviance();

  // actset_indcat[i] == 1 if i is in the active set
  std::vector<int> actset_indcat(d, 0);
  double zeta = 0.02;
  // actset_idx <- which(actset_indcat==1)
  std::vector<int> actset_idx;

  std::vector<double> old_coef(d);
  std::vector<double> grad(d);

  for (int i = 0; i < d; i++) {grad[i] = fabs(m_obj->get_grad(i));
    // //Rprintf("grad[j] = %lf \n",grad[i]);
  }

  RegFunction *regfunc = new RegL1();
  for (int i = 0; i < lambdas.size(); i++) {
    // //Rprintf("lambda[%d]:%f\n", i, lambdas[i]);

    // Initialize active set
    for (int  j = 0; j < d; j++) {
      if (m_obj->get_model_coef(j) != 0)
        actset_indcat[j] = true;
      else
        actset_indcat[j] = false;
    }

    // Multi Update
    // for (int j = 0; j < d; ++j) {
    //   stage_lambdas[j] = lambdas[i];
    //   if (grad[j] > threshold) actset_indcat[j] = 1;
    // }

    // Single update
    // double max_temp = -1;
    // double max_temp_id = -1;
    // for (int j = 0; j < d; ++j) {
    //   stage_lambdas[j] = lambdas[i];
    //
    // if (grad[j] > threshold)
    //   if (max_temp < grad[j] || max_temp_id == -1) {
    //     max_temp = grad[j];
    //     max_temp_id = j;
    //   }
    // }
    // if(max_temp_id != -1)
    //   actset_indcat[max_temp_id] = 1;


    // N cord update
    int maxUpdateCord = MAXUPDATECORDNUM;
    int numUpdateCord = 0;
    std::priority_queue<std::pair<double, int>,
           std::vector<std::pair<double, int> >,
           CompareByFirst> max_temp_pq;
    for (int k = 0; k < d; k++)
    if(!actset_indcat[k])
    {
      if (grad[k] > (1-zeta)*lambdas[i]) {
           numUpdateCord++;
           max_temp_pq.push(std::make_pair(grad[k], k));
           if(numUpdateCord>maxUpdateCord) max_temp_pq.pop();
      }
    }
    //Rprintf("LaLaLa size of new active set: %d\n", max_temp_pq.size());
    while(! max_temp_pq.empty())
    {
      std::pair<double, int> pair_temp = max_temp_pq.top();
      max_temp_pq.pop();
      actset_indcat[pair_temp.second] = true;
    }


    m_obj->update_auxiliary();
    // loop level 0: multistage convex relaxation
    int loopcnt_level_0 = 0;
    int idx;
    double old_beta, old_intcpt, updated_coord, beta;
    while (loopcnt_level_0 < m_param.num_relaxation_round) {
      loopcnt_level_0++;
      //Rprintf("loopcnt_level_0 = %d\n",loopcnt_level_0);

      // loop level 1: active set update
      int loopcnt_level_1 = 0;
      bool terminate_loop_level_1 = true;
      while (loopcnt_level_1 < m_param.max_iter) {
        loopcnt_level_1++;
        //Rprintf("\t loopcnt_level_1 = %d\n",loopcnt_level_1);
        terminate_loop_level_1 = true;

        old_intcpt = m_obj->get_model_coef(-1);
        for (int j = 0; j < d; j++) old_coef[j] = m_obj->get_model_coef(j);

        // initialize actset_idx
        actset_idx.clear();
        for (int j = 0; j < d; j++)
          if (actset_indcat[j]) {
            regfunc->set_param(lambdas[i], 0.0);
            updated_coord = m_obj->coordinate_descent(regfunc, j);
            // //Rprintf("\t updated_coord: %lf",updated_coord);
            if (fabs(updated_coord) > 0) actset_idx.push_back(j);
          }

        // loop level 2: proximal newton on active set
        int loopcnt_level_2 = 0;
        bool terminate_loop_level_2 = true;
        while (loopcnt_level_2 < m_param.max_iter) {
          loopcnt_level_2++;
          //Rprintf("\t\t loopcnt_level_2 = %d; actset_idx.size: %d\n",loopcnt_level_2, actset_idx.size());
          terminate_loop_level_2 = true;

          for (int k = 0; k < actset_idx.size(); k++) {
            idx = actset_idx[k];

            old_beta = m_obj->get_model_coef(idx);
            regfunc->set_param(lambdas[i], 0.0);

            //m_obj->update_gradient(idx); // added by haoming
            m_obj->coordinate_descent(regfunc, idx);

            if (m_obj->get_local_change(old_beta, idx) > dev_thr)
              terminate_loop_level_2 = false;
          }

          if (m_param.include_intercept) {
            old_intcpt = m_obj->get_model_coef(-1);
            m_obj->intercept_update();
            if (m_obj->get_local_change(old_intcpt, -1) > dev_thr)
              terminate_loop_level_2 = false;
          }

          if (terminate_loop_level_2) break;
        }
        // //Rprintf("---------loopcnt cnt level 2:%d\n", loopcnt_level_2);

        itercnt_path[i] += loopcnt_level_2;

        terminate_loop_level_1 = true;
        // check stopping criterion 1: fvalue change
        for (int k = 0; k < actset_idx.size(); ++k) {
          idx = actset_idx[k];
          if (m_obj->get_local_change(old_coef[idx], idx) > dev_thr)
            terminate_loop_level_1 = false;
        }
        if ((m_param.include_intercept) &&
            (m_obj->get_local_change(old_intcpt, -1) > dev_thr))
          terminate_loop_level_1 = false;

        // update p and w
        m_obj->update_auxiliary();

        if (terminate_loop_level_1) break;

        // check stopping criterion 2: active set change
        // Multi Update
        // bool new_active_idx = false;
        // for (int k = 0; k < d; k++)
        //   if (actset_indcat[k] == 0) {
        //     m_obj->update_gradient(k);
        //     grad[k] = fabs(m_obj->get_grad(k));
        //     if (grad[k] > lambdas[i]*(1-zeta)) {
        //       actset_indcat[k] = 1;
        //       new_active_idx = true;
        //     }
        //   }

        //Single Update
        // double max_temp = -1;
        // double max_temp_id = -1;
        // bool new_active_idx = false;
        // for (int k = 0; k < d; k++)
        //   if (actset_indcat[k] == 0) {
        //     m_obj->update_gradient(k);
        //     grad[k] = fabs(m_obj->get_grad(k));
        //     if (grad[k] > stage_lambdas[k])
        //     if (max_temp_id==-1||max_temp<grad[k]){
        //       max_temp = grad[k];
        //       max_temp_id = k;
        //       new_active_idx = true;
        //     }
        //   }
        // if(max_temp_id != -1)
        //   actset_indcat[max_temp_id] = true;

        // N cord update
        int maxUpdateCord = MAXUPDATECORDNUM;
        int numUpdateCord = 0;
        std::priority_queue<std::pair<double, int>,
               std::vector<std::pair<double, int> >,
               CompareByFirst> max_temp_pq;
        bool new_active_idx = false;
        for (int  k = 0; k < d; k++) {
          if (m_obj->get_model_coef(k) != 0)
            actset_indcat[k] = true;
          else
            actset_indcat[k] = false;
        }
        for (int k = 0; k < d; k++)
          if (actset_indcat[k] == 0) {
             m_obj->update_gradient(k);
             grad[k] = fabs(m_obj->get_grad(k));
             if (grad[k] > lambdas[i]*(1-zeta)){
               numUpdateCord++;
               max_temp_pq.push(std::make_pair(grad[k], k));
               if(numUpdateCord>maxUpdateCord) max_temp_pq.pop();
             }
           }
        new_active_idx = numUpdateCord>0;
        //Rprintf("size of new active set: %d\n", max_temp_pq.size());
        while(! max_temp_pq.empty())
        {
          std::pair<double, int> pair_temp = max_temp_pq.top();
          max_temp_pq.pop();
          actset_indcat[pair_temp.second] = true;
          // //Rprintf("%d\n",pair_temp.second);
        }

        if (!new_active_idx) break;
      }

      // //Rprintf("---loop level 1 cnt:%d\n", loopcnt_level_1);

      if (m_param.reg_type == L1) break;

      m_obj->update_auxiliary();

    }

    solution_path.push_back(m_obj->get_model_param());
  }

  delete regfunc;
}  // namespace solver



}  // namespace solver
}  // namespace picasso
